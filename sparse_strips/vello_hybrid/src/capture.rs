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
/// Re-exported so the out-of-crate FFI can read per-run texture id + strip range
/// from `CapturedPass::external_runs` (Sumi capture fork delta).
pub use crate::schedule::ExternalTextureRun as CapturedExternalTextureRun;
use crate::filter::FilterContext;
use crate::{Config, GpuStrip, RenderError, Scene};
use vello_common::multi_atlas::AtlasConfig;
use vello_common::render_graph::LayerId;

use crate::gradient_cache::GradientRampCache;
use crate::render::common::{
    pack_image_offset, pack_image_params, pack_image_size, pack_radial_kind_and_swapped,
    pack_texture_width_and_extend_mode, pack_tint, GpuBlurredRoundedRect, GpuEncodedImage,
    GpuEncodedPaint, GpuLinearGradient, GpuRadialGradient, GpuSweepGradient,
};
use vello_common::encode::{
    EncodedBlurredRoundedRectangle, EncodedExternalTexture, EncodedGradient, EncodedKind,
    EncodedPaint, RadialKind,
};

/// Image-source flag (bit 14 of `image_params`): 1 = externally-bound texture
/// (sampled directly), 0 = atlas. Mirrors `wgpu.rs`'s private constant.
const EXTERNAL_IMAGE_SOURCE_FLAG: u32 = 1 << 14;
use vello_common::fearless_simd::Level;
use vello_common::peniko::Extend;

/// Encode one gradient paint into its GPU texel form. Replicates the (wgpu-only)
/// `Renderer::encode_gradient_paint` so the capture path can build the same
/// `encoded_paints` buffer the wgpu backend uploads, without depending on the wgpu
/// feature. `gradient_width`/`gradient_start` come from the shared `GradientRampCache`.
fn encode_gradient_paint(
    gradient: &EncodedGradient,
    gradient_width: u32,
    gradient_start: u32,
) -> GpuEncodedPaint {
    let transform = gradient.transform.as_coeffs().map(|x| x as f32);
    let extend_mode = match gradient.extend {
        Extend::Pad => 0,
        Extend::Repeat => 1,
        Extend::Reflect => 2,
    };
    let texture_width_and_extend_mode =
        pack_texture_width_and_extend_mode(gradient_width, extend_mode);

    match &gradient.kind {
        EncodedKind::Linear(_) => GpuEncodedPaint::LinearGradient(GpuLinearGradient {
            texture_width_and_extend_mode,
            gradient_start,
            transform,
        }),
        EncodedKind::Radial(radial) => {
            let (kind, bias, scale, fp0, fp1, fr1, f_focal_x, f_is_swapped, scaled_r0_squared) =
                match radial {
                    RadialKind::Radial { bias, scale } => {
                        (0, *bias, *scale, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
                    }
                    RadialKind::Strip { scaled_r0_squared } => {
                        (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, *scaled_r0_squared)
                    }
                    RadialKind::Focal {
                        focal_data,
                        fp0,
                        fp1,
                    } => (
                        2,
                        *fp0,
                        *fp1,
                        *fp0,
                        *fp1,
                        focal_data.fr1,
                        focal_data.f_focal_x,
                        focal_data.f_is_swapped as u32,
                        0.0,
                    ),
                };
            GpuEncodedPaint::RadialGradient(GpuRadialGradient {
                texture_width_and_extend_mode,
                gradient_start,
                transform,
                kind_and_f_is_swapped: pack_radial_kind_and_swapped(kind, f_is_swapped),
                bias,
                scale,
                fp0,
                fp1,
                fr1,
                f_focal_x,
                scaled_r0_squared,
            })
        }
        EncodedKind::Sweep(sweep) => GpuEncodedPaint::SweepGradient(GpuSweepGradient {
            texture_width_and_extend_mode,
            gradient_start,
            transform,
            start_angle: sweep.start_angle,
            inv_angle_delta: sweep.inv_angle_delta,
            _padding: [0, 0],
        }),
    }
}

/// Encode one external-texture paint into its GPU texel form. Replicates the
/// (wgpu-only) `Renderer::encode_external_texture_paint` so the capture path can
/// build the same `GpuEncodedImage` the wgpu backend uploads, without the wgpu
/// feature. `source_region` becomes the image offset/size; the source-kind flag
/// marks it external (sampled directly rather than from the atlas). The texture
/// identity travels separately in `CapturedPass::external_runs` (Sumi maps the
/// `texture_id` to a bindless sampled-image slot).
fn encode_external_texture_paint(image: &EncodedExternalTexture) -> GpuEncodedPaint {
    let transform = image.transform.as_coeffs().map(|x| x as f32);
    let region = image.source_region;
    let image_params = pack_image_params(
        image.sampler.quality as u32,
        image.sampler.x_extend as u32,
        image.sampler.y_extend as u32,
        0,
    ) | EXTERNAL_IMAGE_SOURCE_FLAG;
    let (tint, tint_mode) = pack_tint(image.tint);
    GpuEncodedPaint::Image(GpuEncodedImage {
        image_params,
        image_size: pack_image_size(region.width(), region.height()),
        image_offset: pack_image_offset(region.x0, region.y0),
        transform,
        tint,
        tint_mode,
        image_padding: 0,
    })
}

/// Encode one blurred-rounded-rectangle paint into its GPU texel form. Replicates the
/// (wgpu-only) `Renderer::encode_blurred_rounded_rect_paint` so the capture path can build
/// the same `encoded_paints` entry the wgpu backend uploads, without the wgpu feature. The
/// Slang shader's `calculate_blurred_rounded_rect` reads these five texels.
fn encode_blurred_rounded_rect_paint(rect: &EncodedBlurredRoundedRectangle) -> GpuEncodedPaint {
    GpuEncodedPaint::BlurredRoundedRect(GpuBlurredRoundedRect {
        transform: rect.transform.as_coeffs().map(|x| x as f32),
        color: rect.color.as_premul_rgba8().to_u32(),
        invert: u32::from(rect.invert),
        params0: [rect.exponent, rect.recip_exponent, rect.scale, rect.std_dev_inv],
        params1: [rect.min_edge, rect.w, rect.h, rect.r1],
        size: [rect.width, rect.height],
        _padding1: [0, 0],
    })
}

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

    // H2a — gradients. Build the GPU paint buffer + ramp LUT exactly as the wgpu
    // backend's `prepare_gpu_encoded_paints` does, but for gradients only: walk the
    // scene's `EncodedPaint`s, ramp each gradient via the shared `GradientRampCache`,
    // encode it to texels, and record its texel offset in `paint_idxs`. This MUST run
    // before `do_scene` — the scheduler packs `paint_idxs[paint_id]` into each strip's
    // paint field. Images, external textures and blurred rounded rects are not encoded
    // yet (their `paint_idxs` entry points at the running offset); the GPU shader still
    // renders those paint types transparent, so leaving them unencoded is harmless.
    let encoded_paints = scene.encoded_paints.borrow();
    let level = Level::try_detect().unwrap_or_else(Level::baseline);
    let mut gradient_cache = GradientRampCache::new(encoded_paints.len() as u32, level);
    let mut gpu_paints: Vec<GpuEncodedPaint> = Vec::with_capacity(encoded_paints.len());
    let mut paint_idxs: Vec<u32> = vec![0; encoded_paints.len() + 1];
    let mut current_idx: u32 = 0;
    for (i, paint) in encoded_paints.iter().enumerate() {
        paint_idxs[i] = current_idx;
        let gpu_paint = match paint {
            EncodedPaint::Gradient(gradient) => {
                let (gradient_start, gradient_width) = gradient_cache.get_or_create_ramp(gradient);
                Some(encode_gradient_paint(gradient, gradient_width, gradient_start))
            }
            EncodedPaint::BlurredRoundedRect(rect) => {
                Some(encode_blurred_rounded_rect_paint(rect))
            }
            EncodedPaint::ExternalTexture(img) => Some(encode_external_texture_paint(img)),
            // Atlas images not encoded yet (Sumi uses external/bindless textures).
            _ => None,
        };
        if let Some(gpu_paint) = gpu_paint {
            // Texel count = serialized byte length / 16 (RGBA32Uint) — keep this in sync
            // with `serialize_to_buffer` rather than hardcoding per-kind sizes.
            current_idx += gpu_paint.as_bytes().len() as u32 / 16;
            gpu_paints.push(gpu_paint);
        }
    }
    paint_idxs[encoded_paints.len()] = current_idx;

    let mut encoded_paints_bytes = vec![0_u8; current_idx as usize * 16];
    GpuEncodedPaint::serialize_to_buffer(&gpu_paints, &mut encoded_paints_bytes);
    let gradient_lut = gradient_cache.take_luts();

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
        encoded_paints: encoded_paints_bytes,
        paint_idxs,
        gradient_lut,
        ops: backend.ops,
    })
}
