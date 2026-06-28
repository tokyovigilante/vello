// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capture backend for Sumi's hybrid canvas.
//!
//! This is the *only* fork delta over upstream vello: a `pub` module that reuses the
//! crate-internal scheduler to record, rather than draw, the GPU work for a scene.
//!
//! Upstream's [`Renderer`](crate::Renderer) drives the scheduler with a wgpu
//! [`RendererBackend`](crate::schedule::RendererBackend) that issues real draw calls.
//! Sumi instead has its own Slang+Vulkan backend living outside this crate, so it cannot
//! reach the `pub(crate)` scheduler seam. [`render_to_capture`] bridges that gap: it runs
//! the identical scheduler against a [`CaptureBackend`] that records every
//! `render_strips` / `clear_slots` / `apply_filter` call, plus the texture inputs the GPU
//! stage needs (alpha coverage, encoded paints, gradient LUT, [`Config`]). The result,
//! [`CapturedFrame`], is handed to Nim across the FFI and replayed with instanced-quad
//! draws.
//!
//! Staged build-out: this first cut captures **solid fills + alpha coverage only** — the
//! minimal slice that proves the FFI + strip-dump + Vulkan path end to end. Gradients,
//! blurred rects, images, and clip/blend slot passes are layered on in later phases (the
//! scheduler already emits those passes; the GPU side just needs to execute them).

use alloc::vec;
use alloc::vec::Vec;

use vello_common::tile::Tile;

use crate::schedule::{
    ExternalTextureRun, LoadOp, RendererBackend, RootRenderTarget, Scheduler, SchedulerState,
    StripPassRenderTarget,
};
use crate::filter::FilterContext;
use crate::{Config, GpuStrip, RenderError, Scene};
use vello_common::multi_atlas::AtlasConfig;
use vello_common::render_graph::LayerId;

/// Kind of render target a [`CapturedPass`] draws into.
///
/// Mirrors the crate-internal `StripPassRenderTarget`, flattened for the FFI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CaptureTargetKind {
    /// The root output (the canvas surface). `index` is unused.
    Root = 0,
    /// One of the two clip/blend slot textures. `index` is the texture (0 or 1).
    SlotTexture = 1,
    /// A filter atlas layer. `index` is the [`LayerId`].
    FilterLayer = 2,
}

/// Whether a pass loads or clears its target before drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CaptureLoadOp {
    /// Preserve existing target contents.
    Load = 0,
    /// Clear the target to transparent black first.
    Clear = 1,
}

/// One recorded `render_strips` invocation: an opaque batch then an alpha batch into a
/// single target. The GPU side replays this as two instanced-quad draws (opaque with
/// depth-write/no-blend, alpha with depth-test/premultiplied-over).
#[derive(Debug)]
pub struct CapturedPass {
    /// Which target this pass draws into.
    pub target_kind: CaptureTargetKind,
    /// Slot-texture index or filter [`LayerId`], per `target_kind`. `0` for `Root`.
    pub target_index: u32,
    /// Load/clear behaviour for the target at the start of this pass.
    pub load_op: CaptureLoadOp,
    /// Opaque strips, front-to-back.
    pub opaque: Vec<GpuStrip>,
    /// Alpha (translucent) strips, front-to-back.
    pub alpha: Vec<GpuStrip>,
    /// Runs within `alpha` sharing one external-texture binding (empty until images land).
    pub external_runs: Vec<ExternalTextureRun>,
}

/// A scheduler operation, recorded in execution order so the GPU side can replay faithfully.
#[derive(Debug)]
pub enum CaptureOp {
    /// Clear specific slots in a slot texture before they are reused.
    ClearSlots {
        /// The slot texture (0 or 1).
        texture_index: u32,
        /// Slot indices to clear.
        slots: Vec<u32>,
    },
    /// Draw a pair of opaque/alpha strip batches.
    RenderStrips(CapturedPass),
    /// Apply filter effects for a layer once its content has been rendered.
    ApplyFilter {
        /// The layer whose filter to apply.
        layer_id: u32,
    },
}

/// The full recorded GPU workload for one scene render.
///
/// Owns every buffer the Vulkan backend uploads. Pointers handed across the FFI stay valid
/// for the lifetime of this struct, so Sumi keeps it alive until its uploads complete.
#[derive(Debug)]
pub struct CapturedFrame {
    /// Renderer config uniform (target size, strip height, texture-width log2s).
    pub config: Config,
    /// Alpha coverage bytes, uploaded row-major into the `RGBA32Uint` alpha texture
    /// (16 one-byte alphas per texel).
    pub alphas: Vec<u8>,
    /// Serialized encoded-paint texels (`RGBA32Uint`). Empty in the solid-only cut.
    pub encoded_paints: Vec<u8>,
    /// Per-paint texel offsets into `encoded_paints`; `len() == scene paint count + 1`.
    pub paint_idxs: Vec<u32>,
    /// Flattened gradient ramp LUT bytes. Empty in the solid-only cut.
    pub gradient_lut: Vec<u8>,
    /// Scheduler operations in execution order.
    pub ops: Vec<CaptureOp>,
}

/// Records scheduler output instead of issuing GPU draws.
struct CaptureBackend {
    ops: Vec<CaptureOp>,
}

impl RendererBackend for CaptureBackend {
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]) {
        self.ops.push(CaptureOp::ClearSlots {
            texture_index: texture_index as u32,
            slots: slots.to_vec(),
        });
    }

    fn render_strips(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
        load_op: LoadOp,
    ) {
        let (target_kind, target_index) = match target {
            StripPassRenderTarget::Root(_) => (CaptureTargetKind::Root, 0),
            StripPassRenderTarget::SlotTexture(i) => (CaptureTargetKind::SlotTexture, u32::from(i)),
            StripPassRenderTarget::FilterLayer(layer_id) => {
                (CaptureTargetKind::FilterLayer, layer_id)
            }
        };
        let load_op = match load_op {
            LoadOp::Load => CaptureLoadOp::Load,
            LoadOp::Clear => CaptureLoadOp::Clear,
        };
        self.ops.push(CaptureOp::RenderStrips(CapturedPass {
            target_kind,
            target_index,
            load_op,
            opaque: opaque_strips.to_vec(),
            alpha: alpha_strips.to_vec(),
            external_runs: external_texture_runs.to_vec(),
        }));
    }

    fn apply_filter(&mut self, layer_id: LayerId) {
        self.ops.push(CaptureOp::ApplyFilter { layer_id });
    }
}

/// Run the scheduler against `scene` and capture the GPU workload without a GPU.
///
/// `width`/`height` are the render-target size in pixels. `alphas_tex_width` and
/// `encoded_paints_tex_width` are the widths (in texels) Sumi will allocate for the alpha
/// and encoded-paint textures — they MUST be powers of two and MUST match the textures the
/// Vulkan backend creates, because the shader derives row/column from `log2(width)` stored
/// in [`Config`]. `alphas_tex_width` also sets the slot-texture count
/// (`alphas_tex_width / Tile::HEIGHT`).
///
/// The returned [`CapturedFrame`] owns all buffers; replay it with the Slang+Vulkan backend.
pub fn render_to_capture(
    scene: &Scene,
    width: u32,
    height: u32,
    alphas_tex_width: u32,
    encoded_paints_tex_width: u32,
) -> Result<CapturedFrame, RenderError> {
    debug_assert!(
        alphas_tex_width.is_power_of_two(),
        "alphas_tex_width must be a power of two"
    );
    debug_assert!(
        encoded_paints_tex_width.is_power_of_two(),
        "encoded_paints_tex_width must be a power of two"
    );

    let total_slots = (alphas_tex_width / u32::from(Tile::HEIGHT)) as usize;
    let mut scheduler = Scheduler::new(total_slots);
    let mut state = SchedulerState::default();
    let filter_context = FilterContext::new(AtlasConfig::default());

    // Solid-only cut: no paint encoding yet, so a single trailing zero offset and no paints.
    let paint_idxs: Vec<u32> = vec![0];
    let encoded_paints = scene.encoded_paints.borrow();

    let config = Config {
        width,
        height,
        strip_height: u32::from(Tile::HEIGHT),
        alphas_tex_width_bits: alphas_tex_width.trailing_zeros(),
        encoded_paints_tex_width_bits: encoded_paints_tex_width.trailing_zeros(),
        strip_offset_x: 0,
        strip_offset_y: 0,
        // Vulkan/WebGPU share a y-down framebuffer convention; the NDC y-flip the shader
        // applies matches both. Only the WebGL backend sets this. Revisit in the Slang port.
        negate_ndc: 0,
    };

    let mut backend = CaptureBackend { ops: Vec::new() };
    scheduler.do_scene(
        &mut state,
        &mut backend,
        scene,
        RootRenderTarget::UserSurface,
        &paint_idxs,
        &filter_context,
        &encoded_paints,
    )?;
    drop(encoded_paints);

    let alphas = scene.strip_storage.borrow().alphas.clone();

    Ok(CapturedFrame {
        config,
        alphas,
        encoded_paints: Vec::new(),
        paint_idxs,
        gradient_lut: Vec::new(),
        ops: backend.ops,
    })
}
