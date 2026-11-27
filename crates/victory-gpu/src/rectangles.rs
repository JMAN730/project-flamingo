//! GPU-side expansion of indirect rectangle-list draws (the console's
//! topology 17, `ExecuteIndirect` kind 0). Direct rectangle draws expand
//! their vertices on the CPU ([`crate::primitive::expand_rects`]); indirect
//! draws only know their vertex counts on the GPU, so after
//! [`crate::indirect::clamp_uav`] bounds each record, two compute passes of
//! ours run before `ExecuteIndirect`:
//!
//! 1. `rect_rewrite` reads each live record (skipping records at or past the
//!    title's count, 0-instance draws, and records whose source vertices
//!    `start..start + 3 * rects` reach past any bound per-vertex vertex
//!    buffer), allocates `rects * 6` output vertices from a list-owned
//!    scratch buffer with an atomic counter, and writes a packed 16-byte
//!    `D3D12_DRAW_ARGUMENTS` copy into a list-owned arguments buffer. The
//!    title's arguments are left as the clamp made them.
//! 2. `rect_expand` runs once per per-vertex input slot, writing
//!    `(v0, v1, v2, v0, v2, v3)` into that slot's scratch region with
//!    `v3 = v0 + v2 - v1` for the formats and in-stride components that
//!    [`crate::primitive::expand_rects`] computes.
//!
//! `ExecuteIndirect` then reads our arguments (with the title's count, if
//! any) and the expanded vertices; transitions afterwards return the scratch
//! buffers to unordered access, so the next indirect rectangle draw of the
//! list reuses the whole scratch.
//!
//! shortcut: the per-list vertex scratch has a fixed capacity of 8 MiB
//! ([`RECT_SCRATCH_BYTES`]), shared by the records of one draw; records whose
//! expanded vertices do not fit draw nothing. Raise it if a single indirect
//! draw needs more.
//!
//! shortcut: as in [`crate::primitive`], the fourth corner is computed from
//! vertex inputs rather than vertex shader outputs; formats unknown to
//! [`crate::primitive::components`] and components past the vertex stride
//! keep the first vertex's bytes. A pipeline without per-vertex inputs (one
//! that builds its corners from the vertex id) gets `6 * rects` vertices from
//! 0, as a direct draw does. With inputs, the vertex id is the expanded one
//! from the allocated start, whose order between records may change from run
//! to run. Vertex strides not a multiple of 4 are padded up to 4 bytes in the
//! scratch (and bound with that stride), so threads never share a 32-bit word
//! across vertices.
//!
//! The rewrite pass bounds counts and instances itself, as the clamp does, so
//! it stays safe whether or not the clamp writes its results back in place.

use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::*;

use crate::draw::VertexView;
use crate::host::{HostList, host, transition};
use crate::indirect::uav_barrier;
use crate::pipeline::host::{HostPipeline, Input};
use crate::primitive;
use crate::shadow::Shadow;

/// Per-list GPU-writable vertex scratch capacity (8 MiB), reused per draw.
pub(crate) const RECT_SCRATCH_BYTES: u64 = 8 << 20;

/// Maximum records supported by one 1D `[numthreads(64, 1, 1)]` pass.
const MAX_RECORDS: u32 = 65535 * 64;

/// Maximum thread groups per dispatch dimension (D3D12 spec).
const MAX_GROUPS_PER_DIM: u32 = 65535;

/// One per-vertex input element passed to `rect_expand` (8 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Element {
    /// Byte offset inside the source (and destination) vertex.
    pub(crate) offset: u32,
    /// Low byte: [`primitive::Kind::code`]; bits 8..32: component count inside
    /// the vertex stride (1..=4).
    pub(crate) packed: u32,
}

/// Scratch layout for one per-vertex input slot inside the list's vertex
/// scratch buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotLayout {
    pub(crate) slot: u32,
    /// 16-byte aligned byte offset from the start of the scratch buffer.
    pub(crate) dst_offset: u64,
    /// 4-byte aligned stride in the scratch buffer.
    pub(crate) dst_stride: u32,
    /// Total byte size of this slot's scratch region (`vert_capacity * dst_stride`).
    pub(crate) dst_size: u32,
}

/// Builds the GPU element table for `slot` with vertex `stride`, matching
/// [`crate::primitive::expand_rects`]. Returns `(has_per_vertex, elements)`:
/// `has_per_vertex` is true when the pipeline has at least one per-vertex input
/// on `slot` (even if none of its formats are recognized, in which case the
/// slot is still expanded with `v3 = v0`).
pub(crate) fn slot_elements(inputs: &[Input], slot: u32, stride: u32) -> (bool, Vec<Element>) {
    let mut has_per_vertex = false;
    let mut out = Vec::new();
    let stride = stride as usize;
    for e in inputs {
        if e.slot != slot || e.per_instance {
            continue;
        }
        has_per_vertex = true;
        let Some((n, kind)) = primitive::components(e.format) else {
            continue;
        };
        let inside = stride.saturating_sub(e.offset as usize) / kind.bytes();
        let count = n.min(inside) as u32;
        if count == 0 {
            continue;
        }
        out.push(Element {
            offset: e.offset,
            packed: kind.code() | (count << 8),
        });
    }
    (has_per_vertex, out)
}

/// Lays out disjoint, 16-byte aligned scratch regions for every per-vertex
/// `(slot, src_stride)` inside `capacity_bytes`, sharing one vertex index space
/// `0..vert_capacity`. Returns `(vert_capacity, layouts)`, with `vert_capacity`
/// rounded down to a multiple of 6 (whole rectangles). Returns `None` if a
/// stride is zero or not even one rectangle (6 vertices) fits.
pub(crate) fn scratch_layout(
    slots: &[(u32, u32)],
    capacity_bytes: u64,
) -> Option<(u32, Vec<SlotLayout>)> {
    if slots.is_empty() {
        return Some((0, Vec::new()));
    }
    let mut dst_strides = Vec::with_capacity(slots.len());
    let mut total_stride: u64 = 0;
    for &(_, src_stride) in slots {
        if src_stride == 0 {
            return None;
        }
        let dst_stride = src_stride.checked_next_multiple_of(4)?;
        dst_strides.push(dst_stride);
        total_stride = total_stride.checked_add(u64::from(dst_stride))?;
    }
    // Without alignment padding at most this many vertices fit; the padding
    // between slots (under 16 bytes each) then costs at most a few
    // rectangles, so step down until the padded layout fits.
    let mut vert_capacity =
        ((capacity_bytes / total_stride / 6) * 6).min(u64::from(u32::MAX) / 6 * 6);
    let layouts = loop {
        if vert_capacity < 6 {
            return None;
        }
        if let Some(layouts) = place(slots, &dst_strides, vert_capacity, capacity_bytes) {
            break layouts;
        }
        vert_capacity -= 6;
    };
    Some((vert_capacity as u32, layouts))
}

/// The slots' regions for `verts` vertices each, 16-byte aligned one after
/// the other, if they end within `capacity_bytes`.
fn place(
    slots: &[(u32, u32)],
    dst_strides: &[u32],
    verts: u64,
    capacity_bytes: u64,
) -> Option<Vec<SlotLayout>> {
    let mut cursor = 0u64;
    let mut layouts = Vec::with_capacity(slots.len());
    for (&(slot, _), &dst_stride) in slots.iter().zip(dst_strides) {
        let dst_offset = cursor.next_multiple_of(16);
        let dst_size = verts.checked_mul(u64::from(dst_stride))?;
        let end = dst_offset.checked_add(dst_size)?;
        if end > capacity_bytes {
            return None;
        }
        layouts.push(SlotLayout {
            slot,
            dst_offset,
            dst_stride,
            dst_size: u32::try_from(dst_size).ok()?,
        });
        cursor = end;
    }
    Some(layouts)
}

/// The vertex index bound shared across all bound per-vertex slots: a record's
/// `start..start + 3 * rects` lies inside every slot iff it lies inside
/// `0..max_source_vertices`. Empty `bounds` (no per-vertex slots) returns
/// `u32::MAX`.
pub(crate) fn max_source_vertices(bounds: &[(u64, u32)]) -> u32 {
    bounds
        .iter()
        .map(|&(size, stride)| {
            if stride == 0 {
                0
            } else {
                (size / u64::from(stride)).min(u64::from(u32::MAX)) as u32
            }
        })
        .min()
        .unwrap_or(u32::MAX)
}

/// 2D dispatch grid `(x, y)` covering `records` thread groups with each
/// dimension within D3D12's 65,535-group limit.
pub(crate) fn dispatch_grid(records: u32) -> (u32, u32) {
    if records == 0 {
        return (0, 0);
    }
    let x = records.min(MAX_GROUPS_PER_DIM);
    let y = records.div_ceil(x);
    (x, y)
}

const REWRITE_SOURCE: &str = include_str!("../shaders/rect_rewrite.hlsl");

const EXPAND_SOURCE: &str = include_str!("../shaders/rect_expand.hlsl");

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

struct Shaders {
    rewrite: Pass,
    expand: Pass,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Shaders {}
// SAFETY: as above.
unsafe impl Sync for Shaders {}

fn shaders() -> Option<&'static Shaders> {
    static SHADERS: std::sync::OnceLock<Option<Shaders>> = std::sync::OnceLock::new();
    SHADERS
        .get_or_init(|| match make() {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(error = %e, "no rectangle expander: indirect rectangles dropped");
                None
            }
        })
        .as_ref()
}

fn compile_pass(device: &ID3D12Device, name: &str, source: &str) -> windows::core::Result<Pass> {
    let bytes = &crate::hlsl::compile_compute(name, source)?[..];
    // SAFETY: plain object creation; the root signature is inside the shader.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Pass { root, pso })
    }
}

fn make() -> windows::core::Result<Shaders> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let rewrite = compile_pass(device, "rect_rewrite", REWRITE_SOURCE)?;
    let expand = compile_pass(device, "rect_expand", EXPAND_SOURCE)?;
    Ok(Shaders { rewrite, expand })
}

/// Rewritten indirect arguments and expanded vertex buffer views produced by
/// [`expand`], ready to transition around `ExecuteIndirect`.
pub(crate) struct Prepared {
    pub(crate) args_buf: ID3D12Resource,
    vert_buf: Option<ID3D12Resource>,
    views: Vec<(u32, D3D12_VERTEX_BUFFER_VIEW)>,
    counter: Option<ID3D12Resource>,
}

impl Prepared {
    /// Transitions the rewritten arguments, expanded vertex scratch and
    /// optional count buffer for `ExecuteIndirect`, and binds the expanded
    /// per-vertex buffer views (overriding the title's slots after `prepare`).
    pub(crate) fn before_draw(&self, list: &HostList) {
        let mut barriers = Vec::with_capacity(3);
        barriers.push(transition(
            &self.args_buf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ));
        if let Some(vb) = &self.vert_buf {
            barriers.push(transition(
                vb,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            ));
        }
        if let Some(counter) = &self.counter {
            list.use_shadow(counter, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        }
        // SAFETY: an open list and live buffers owned by the list or shadows.
        unsafe {
            list.list.ResourceBarrier(&barriers);
            for &(slot, view) in &self.views {
                list.list.IASetVertexBuffers(slot, Some(&[view]));
            }
        }
    }

    /// Restores the scratch buffers and optional count buffer to unordered
    /// access so subsequent passes and draws can reuse them.
    pub(crate) fn after_draw(&self, list: &HostList) {
        let mut barriers = Vec::with_capacity(3);
        barriers.push(transition(
            &self.args_buf,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ));
        if let Some(vb) = &self.vert_buf {
            barriers.push(transition(
                vb,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ));
        }

        // SAFETY: an open list and live buffers owned by the list or shadows.
        unsafe { list.list.ResourceBarrier(&barriers) };
    }
}

struct SlotWork {
    slot: u32,
    shadow: Arc<Shadow>,
    address: u64,
    size: u32,
    src_stride: u32,
    elements: Vec<Element>,
}

/// Clamps the title's indirect draw arguments (`kind == 0`) and records our
/// rectangle-expansion compute passes on `list`. On `Err(reason)` nothing was
/// recorded onto the host list; on `Ok(prepared)` the compute pipeline and root
/// signature were replaced and the caller binds its graphics state, calls
/// [`Prepared::before_draw`], executes indirect with stride 16 on
/// `prepared.args_buf`, and calls [`Prepared::after_draw`].
pub(crate) fn expand(
    list: &HostList,
    pipeline: &HostPipeline,
    vertices: &[VertexView; 32],
    (args, arguments, stride, max): (&Arc<Shadow>, u64, u32, u32),
    counter: Option<(&Arc<Shadow>, u64)>,
) -> Result<Prepared, &'static str> {
    if stride < 16 || !stride.is_multiple_of(4) || !args.gpu(arguments).is_multiple_of(4) {
        return Err("rectangle arguments unaligned or short");
    }
    if max > MAX_RECORDS {
        return Err("too many indirect rectangle records");
    }
    if let Some((s, at)) = counter
        && !s.gpu(at).is_multiple_of(4)
    {
        return Err("rectangle count buffer not 4-byte aligned");
    }

    let mut work = Vec::new();
    let mut bounds = Vec::new();
    for slot in 0..32u32 {
        let v = vertices[slot as usize];
        let (has_per_vertex, elements) = slot_elements(&pipeline.inputs, slot, v.stride);
        if !has_per_vertex {
            continue;
        }
        if v.address == 0 || v.stride == 0 || v.size == 0 {
            return Err("rectangle vertex buffer missing");
        }
        let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
            return Err("rectangle vertex buffer outside title memory");
        };
        let effective = u64::from(v.size)
            .min(shadow.left(v.address))
            .min(u64::from(u32::MAX - 3));
        bounds.push((effective, v.stride));
        work.push(SlotWork {
            slot,
            shadow,
            address: v.address,
            size: v.size,
            src_stride: v.stride,
            elements,
        });
    }

    let slot_strides: Vec<(u32, u32)> = work.iter().map(|w| (w.slot, w.src_stride)).collect();
    let Some((vert_capacity, layouts)) = scratch_layout(&slot_strides, RECT_SCRATCH_BYTES) else {
        return Err("rectangle vertex stride exceeds scratch capacity");
    };
    let max_src_verts = max_source_vertices(&bounds);

    let Some(s) = shaders() else {
        return Err("rectangle expander shader unavailable");
    };
    let args_bytes = u64::from(max).max(1) * 16;
    let Some(dst_args_buf) = list.rect_args(args_bytes) else {
        return Err("rectangle rewritten-arguments allocation failed");
    };
    let Some(alloc_gpu) = list.rect_counter() else {
        return Err("rectangle counter allocation failed");
    };
    let vert_buf = if work.is_empty() {
        None
    } else {
        Some(
            list.rect_vertices(RECT_SCRATCH_BYTES)
                .ok_or("rectangle vertex scratch allocation failed")?,
        )
    };

    // Upload per-slot element tables before recording any GPU work.
    let mut elem_gpus = Vec::with_capacity(work.len());
    for w in &work {
        let byte_len = (w.elements.len().max(1) * size_of::<Element>()) as u64;
        let Some((gpu, ptr)) = list.scratch(byte_len) else {
            return Err("rectangle element upload allocation failed");
        };
        if !w.elements.is_empty() {
            // SAFETY: `ptr` is freshly allocated 256-byte-aligned upload memory
            // for this list of at least `byte_len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    w.elements.as_ptr().cast::<u8>(),
                    ptr,
                    w.elements.len() * size_of::<Element>(),
                );
            }
        }
        elem_gpus.push(gpu);
    }

    // Refresh title buffers at submission time.
    let args_len = u64::from(stride) * u64::from(max.max(1));
    args.refresh_in_place(
        list,
        arguments,
        args_len,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    if let Some((c, at)) = counter {
        c.refresh_in_place(list, at, 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    for w in &work {
        w.shadow.refresh_as(
            list,
            w.address,
            u64::from(w.size),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
    }

    let src_args_gpu = args.gpu(arguments);
    if !crate::indirect::clamp_uav(list, src_args_gpu, (0, stride, max)) {
        return Err("rectangle clamp unavailable");
    }

    // SAFETY: live D3D12 resources.
    let dst_args_gpu = unsafe { dst_args_buf.GetGPUVirtualAddress() };
    let count_gpu = counter.map_or(src_args_gpu, |(c, at)| c.gpu(at));
    let rewrite_consts = [
        max,
        stride,
        u32::from(counter.is_some()),
        max_src_verts,
        vert_capacity,
        u32::from(!work.is_empty()),
        0,
        0,
    ];
    let l = &list.list;
    // SAFETY: an open list; our shaders and live list-owned/shadow buffers.
    unsafe {
        l.SetComputeRootSignature(&s.rewrite.root);
        l.SetPipelineState(&s.rewrite.pso);
        l.SetComputeRoot32BitConstants(0, 8, rewrite_consts.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, src_args_gpu);
        l.SetComputeRootUnorderedAccessView(2, count_gpu);
        l.SetComputeRootUnorderedAccessView(3, alloc_gpu);
        l.SetComputeRootUnorderedAccessView(4, dst_args_gpu);
        l.Dispatch(max.div_ceil(64), 1, 1);
    }

    let mut views = Vec::with_capacity(work.len());
    if let Some(vb) = &vert_buf {
        // SAFETY: live buffer.
        let dst_vb_base = unsafe { vb.GetGPUVirtualAddress() };
        let (grid_x, grid_y) = dispatch_grid(max);
        // SAFETY: an open list; our shaders and live buffers.
        unsafe {
            // Make rewritten arguments visible to the vertex-expansion dispatches.
            l.ResourceBarrier(&[uav_barrier()]);
            l.SetComputeRootSignature(&s.expand.root);
            l.SetPipelineState(&s.expand.pso);
            l.SetComputeRootUnorderedAccessView(2, src_args_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_args_gpu);
            for ((w, layout), elem_gpu) in work.iter().zip(&layouts).zip(elem_gpus) {
                let raw_src_gpu = w.shadow.gpu(w.address);
                let src_bias = (raw_src_gpu & 3) as u32;
                let src_vb_gpu = raw_src_gpu & !3;
                let dst_vb_gpu = dst_vb_base + layout.dst_offset;
                let expand_consts = [
                    max,
                    stride,
                    grid_x,
                    src_bias,
                    w.src_stride,
                    layout.dst_stride,
                    w.elements.len() as u32,
                    0,
                ];
                l.SetComputeRoot32BitConstants(0, 8, expand_consts.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, elem_gpu);
                l.SetComputeRootUnorderedAccessView(4, src_vb_gpu);
                l.SetComputeRootUnorderedAccessView(5, dst_vb_gpu);
                l.Dispatch(grid_x, grid_y, 1);
                views.push((
                    layout.slot,
                    D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: dst_vb_gpu,
                        SizeInBytes: layout.dst_size,
                        StrideInBytes: layout.dst_stride,
                    },
                ));
            }
        }
    }

    Ok(Prepared {
        args_buf: dst_args_buf,
        vert_buf,
        views,
        counter: counter.map(|(c, _)| c.buffer.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(slot: u32, offset: u32, format: u32, per_instance: bool) -> Input {
        Input {
            slot,
            offset,
            format,
            per_instance,
        }
    }

    #[test]
    fn slot_elements_filters_and_clips_to_stride() {
        let inputs = [
            input(0, 0, 16, false),  // R32G32_FLOAT (2x4 = 8 bytes)
            input(0, 8, 28, false),  // R8G8B8A8_UNORM (4x1 = 4 bytes)
            input(0, 4, 2, false),   // R32G32B32A32_FLOAT clipped to 2 comps in 12-byte stride
            input(0, 12, 28, false), // past stride -> dropped
            input(0, 0, 42, false),  // R32_UINT (unknown to primitive::components) -> dropped
            input(1, 0, 16, true),   // per-instance -> ignored
            input(2, 0, 42, false),  // unknown only -> still has_per_vertex = true, 0 elements
        ];
        let (has0, elems0) = slot_elements(&inputs, 0, 12);
        assert!(has0);
        assert_eq!(
            elems0,
            vec![
                Element {
                    offset: 0,
                    packed: primitive::Kind::F32.code() | (2 << 8),
                },
                Element {
                    offset: 8,
                    packed: primitive::Kind::Unorm8.code() | (4 << 8),
                },
                Element {
                    offset: 4,
                    packed: primitive::Kind::F32.code() | (2 << 8),
                },
            ]
        );
        let (has1, elems1) = slot_elements(&inputs, 1, 16);
        assert!(!has1);
        assert!(elems1.is_empty());

        let (has2, elems2) = slot_elements(&inputs, 2, 4);
        assert!(has2);
        assert!(elems2.is_empty());
    }

    #[test]
    fn scratch_layout_handles_multi_slot_and_alignment() {
        assert_eq!(scratch_layout(&[], 1024), Some((0, Vec::new())));
        assert_eq!(scratch_layout(&[(0, 0)], 1024), None);
        assert_eq!(scratch_layout(&[(0, 16)], 64), None); // 6*16=96 > 64

        // Two slots: stride 6 (padded to 8) and stride 12 (already multiple of 4).
        let (cap, layouts) = scratch_layout(&[(0, 6), (3, 12)], 1024).unwrap();
        assert_eq!(cap % 6, 0);
        assert!(cap >= 6);
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].slot, 0);
        assert_eq!(layouts[0].dst_offset, 0);
        assert_eq!(layouts[0].dst_stride, 8);
        assert_eq!(layouts[0].dst_size, cap * 8);
        assert_eq!(layouts[1].slot, 3);
        assert_eq!(layouts[1].dst_offset % 16, 0);
        assert!(layouts[1].dst_offset >= u64::from(layouts[0].dst_size));
        assert_eq!(layouts[1].dst_stride, 12);
        assert_eq!(layouts[1].dst_size, cap * 12);
        assert!(layouts[1].dst_offset + u64::from(layouts[1].dst_size) <= 1024);

        // Two slots that fit exactly, with no padding needed between them:
        // 6 vertices of 8 bytes, then 6 of 12 (48 is 16-byte aligned).
        let (cap, layouts) = scratch_layout(&[(0, 8), (1, 12)], 120).unwrap();
        assert_eq!(cap, 6);
        assert_eq!(layouts[1].dst_offset, 48);
        // Padding is counted where it is needed: 6 * 4 = 24 bytes, then 16-byte
        // alignment to 32, then 6 * 4 more; 55 bytes are too few, 56 enough.
        assert_eq!(scratch_layout(&[(0, 4), (1, 4)], 55), None);
        let (cap, layouts) = scratch_layout(&[(0, 4), (1, 4)], 56).unwrap();
        assert_eq!((cap, layouts[1].dst_offset), (6, 32));
    }

    #[test]
    fn max_source_vertices_takes_minimum_across_slots() {
        assert_eq!(max_source_vertices(&[]), u32::MAX);
        assert_eq!(max_source_vertices(&[(120, 12), (80, 16)]), 5);
        assert_eq!(max_source_vertices(&[(100, 0), (80, 16)]), 0);
    }

    #[test]
    fn dispatch_grid_respects_d3d12_limit() {
        assert_eq!(dispatch_grid(0), (0, 0));
        assert_eq!(dispatch_grid(1), (1, 1));
        assert_eq!(dispatch_grid(65535), (65535, 1));
        let (x, y) = dispatch_grid(65536);
        assert!(x <= 65535 && y <= 65535);
        assert!(u64::from(x) * u64::from(y) >= 65536);
    }

    /// Both rectangle-expansion compute shaders compile (system compiler or
    /// DXC) and the host device accepts them.
    #[test]
    fn rectangle_expander_builds() {
        if host().is_none() {
            return; // no host GPU here
        }
        assert!(shaders().is_some());
    }

    /// Both HLSL shaders compile with the system's compiler when it supports
    /// `cs_5_1` raw buffers (`RWByteAddressBuffer::Load`), and with DXC when
    /// that is present.
    #[test]
    fn hlsl_compiles() {
        use crate::hlsl::{dxc_compile, system_compile};
        let probe = r#"
            RWByteAddressBuffer b : register(u0);
            [RootSignature("UAV(u0)")]
            [numthreads(1, 1, 1)]
            void main() { b.Store(0, b.Load(0)); }
        "#;
        for (name, source) in [
            ("rect_rewrite", REWRITE_SOURCE),
            ("rect_expand", EXPAND_SOURCE),
        ] {
            // A compiler that lacks raw buffer loads (Wine's vkd3d-shader)
            // or a missing DXC is skipped.
            if system_compile("probe", probe).is_ok()
                && let Err((e, msg)) = system_compile(name, source)
            {
                panic!("{name} HLSL failed ({e}): {msg}");
            }
            if dxc_compile("probe", probe).is_ok()
                && let Err(e) = dxc_compile(name, source)
            {
                panic!("{name} HLSL failed with DXC ({e})");
            }
        }
    }

    /// End-to-end GPU test: runs `rect_rewrite` and `rect_expand` on synthetic
    /// vertex data covering F32, F16, Unorm16, Unorm8, unknown formats and
    /// elements past the stride, plus count-buffer and out-of-bounds records,
    /// and verifies the output byte-for-byte against `primitive::expand_rects`.
    #[test]
    fn gpu_expansion_matches_cpu_expand_rects() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some(s) = shaders() else {
            panic!("rectangle expander shaders failed to build on a live host");
        };
        let list = host.new_list().expect("host command list");

        // Synthetic 24-byte vertex layout:
        //  0: R32G32_FLOAT (8 bytes)
        //  8: R16G16_FLOAT (4 bytes)
        // 12: R16G16_UNORM (4 bytes)
        // 16: R8G8B8A8_UNORM (4 bytes)
        // 20: R32_UINT (4 bytes, unknown -> must keep v0's bytes)
        // 24: R8G8B8A8_UNORM (past stride -> ignored)
        const STRIDE: usize = 24;
        let vertex = |x: f32, y: f32, u: u16, v: u16, n0: u16, n1: u16, rgba: [u8; 4], tag: u32| {
            let mut out = Vec::with_capacity(STRIDE);
            out.extend(x.to_le_bytes());
            out.extend(y.to_le_bytes());
            out.extend(u.to_le_bytes());
            out.extend(v.to_le_bytes());
            out.extend(n0.to_le_bytes());
            out.extend(n1.to_le_bytes());
            out.extend(rgba);
            out.extend(tag.to_le_bytes());
            out
        };
        // 3 rectangles (9 vertices total):
        // Rect 0: standard full-screen quad
        // Rect 1: negative/clamped colors and negative half-floats
        // Rect 2: used by record 1 starting at vertex 6
        let src_bytes: Vec<u8> = [
            // Rect 0 (verts 0..3)
            vertex(
                -1.0,
                1.0,
                0x0000,
                0x0000,
                0,
                10000,
                [0, 50, 200, 255],
                0x1122_3344,
            ),
            vertex(
                -1.0,
                -1.0,
                0x0000,
                0x3c00,
                10000,
                30000,
                [100, 100, 100, 255],
                0x5566_7788,
            ),
            vertex(
                1.0,
                -1.0,
                0x3c00,
                0x3c00,
                40000,
                50000,
                [150, 200, 50, 255],
                0x99aa_bbcc,
            ),
            // Rect 1 (verts 3..6)
            vertex(
                0.25,
                0.5,
                0x3800,
                0xc000,
                50000,
                1000,
                [220, 10, 128, 64],
                0xdead_beef,
            ),
            vertex(
                0.25,
                -0.5,
                0x3c00,
                0xbc00,
                10000,
                20000,
                [50, 200, 250, 200],
                0xcafe_babe,
            ),
            vertex(
                2.5,
                -0.5,
                0x4000,
                0x3e00,
                60000,
                5000,
                [200, 20, 10, 100],
                0x0102_0304,
            ),
            // Rect 2 (verts 6..9)
            vertex(
                10.0,
                20.0,
                0x3e00,
                0x4200,
                12345,
                54321,
                [10, 20, 30, 40],
                0xaabb_ccdd,
            ),
            vertex(
                10.0,
                30.0,
                0x4000,
                0x4400,
                22222,
                11111,
                [50, 60, 70, 80],
                0x1234_5678,
            ),
            vertex(
                40.0,
                30.0,
                0x4200,
                0x4000,
                33333,
                22222,
                [90, 80, 70, 60],
                0x8765_4321,
            ),
        ]
        .concat();
        assert_eq!(src_bytes.len(), 9 * STRIDE);

        let inputs = [
            input(0, 0, 16, false),
            input(0, 8, 34, false),
            input(0, 12, 35, false),
            input(0, 16, 28, false),
            input(0, 20, 42, false),
            input(0, 24, 28, false),
        ];
        let raw_elements: Vec<(u32, u32)> = inputs.iter().map(|i| (i.offset, i.format)).collect();
        let (has_pv, gpu_elems) = slot_elements(&inputs, 0, STRIDE as u32);
        assert!(has_pv);

        // 5 indirect records, count buffer = 4 (record 4 must not allocate):
        // 0: 6 verts (2 rects) from 0, 1 instance
        // 1: 3 verts (1 rect) from 6, 3 instances, start_instance 7
        // 2: out of bounds (start 8, 3 verts > 9) -> count' = 0
        // 3: 0 instances (3 verts, 0 instances) -> count' = 0
        // 4: past live count (3 verts, 1 instance) -> count' = 0
        let records: [[u32; 4]; 5] = [
            [6, 1, 0, 0],
            [3, 3, 6, 7],
            [3, 1, 8, 0],
            [3, 0, 0, 0],
            [3, 1, 0, 0],
        ];
        let (src_args_buf, src_args_ptr) = host.shared_buffer(256).expect("src_args");
        let (count_buf, count_ptr) = host.shared_buffer(256).expect("count");
        let (alloc_buf, alloc_ptr) = host.shared_buffer(256).expect("alloc");
        let (dst_args_buf, dst_args_ptr) = host.shared_buffer(256).expect("dst_args");
        let (src_vb_buf, src_vb_ptr) = host.shared_buffer(4096).expect("src_vb");
        let (dst_vb_buf, dst_vb_ptr) = host.shared_buffer(4096).expect("dst_vb");
        let (elem_gpu, elem_ptr) = list.scratch(256).expect("elems");

        // SAFETY: freshly mapped host buffers of sufficient size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                records.as_ptr().cast::<u8>(),
                src_args_ptr,
                size_of_val(&records),
            );
            (count_ptr as *mut u32).write(4);
            (alloc_ptr as *mut u32).write(0);
            std::ptr::write_bytes(dst_args_ptr, 0xaa, 256);
            std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), src_vb_ptr, src_bytes.len());
            std::ptr::write_bytes(dst_vb_ptr, 0x55, 4096);
            std::ptr::copy_nonoverlapping(
                gpu_elems.as_ptr().cast::<u8>(),
                elem_ptr,
                gpu_elems.len() * size_of::<Element>(),
            );

            let rewrite_consts = [5u32, 16, 1, 9, 64, 1, 0, 0];
            let l = &list.list;
            l.SetComputeRootSignature(&s.rewrite.root);
            l.SetPipelineState(&s.rewrite.pso);
            l.SetComputeRoot32BitConstants(0, 8, rewrite_consts.as_ptr().cast(), 0);
            l.SetComputeRootUnorderedAccessView(1, src_args_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(2, count_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(3, alloc_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(4, dst_args_buf.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);

            l.ResourceBarrier(&[uav_barrier()]);

            let expand_consts = [
                5u32,
                16,
                5,
                0,
                STRIDE as u32,
                STRIDE as u32,
                gpu_elems.len() as u32,
                0,
            ];
            l.SetComputeRootSignature(&s.expand.root);
            l.SetPipelineState(&s.expand.pso);
            l.SetComputeRoot32BitConstants(0, 8, expand_consts.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, elem_gpu);
            l.SetComputeRootUnorderedAccessView(2, src_args_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(3, dst_args_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(4, src_vb_buf.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(5, dst_vb_buf.GetGPUVirtualAddress());
            l.Dispatch(5, 1, 1);
        }
        list.close();
        host.execute(&[&list]);

        // SAFETY: execution finished; buffers are mapped and readable.
        let (out_recs, allocated, gpu_out) = unsafe {
            let mut out = [[0u32; 4]; 5];
            std::ptr::copy_nonoverlapping(
                dst_args_ptr,
                out.as_mut_ptr().cast::<u8>(),
                size_of_val(&out),
            );
            let allocated = (alloc_ptr as *const u32).read();
            let gpu_out = std::slice::from_raw_parts(dst_vb_ptr, 4096).to_vec();
            (out, allocated, gpu_out)
        };

        // Total allocated vertices: 12 (record 0) + 6 (record 1) = 18.
        assert_eq!(allocated, 18);
        assert_eq!(out_recs[0][0], 12);
        assert_eq!(out_recs[0][1], 1);
        assert_eq!(out_recs[0][3], 0);
        assert_eq!(out_recs[1][0], 6);
        assert_eq!(out_recs[1][1], 3);
        assert_eq!(out_recs[1][3], 7);
        assert_eq!(out_recs[2][0], 0);
        assert_eq!(out_recs[3][0], 0);
        assert_eq!(out_recs[4], [0, 0, 0, 0]);

        // Compare expanded vertices byte-for-byte with primitive::expand_rects.
        let mut expected0 = vec![0u8; 12 * STRIDE];
        primitive::expand_rects(
            &src_bytes[0..6 * STRIDE],
            &mut expected0,
            STRIDE,
            &raw_elements,
        );
        let start0 = out_recs[0][2] as usize * STRIDE;
        assert_eq!(&gpu_out[start0..start0 + 12 * STRIDE], &expected0[..]);

        let mut expected1 = vec![0u8; 6 * STRIDE];
        primitive::expand_rects(
            &src_bytes[6 * STRIDE..9 * STRIDE],
            &mut expected1,
            STRIDE,
            &raw_elements,
        );
        let start1 = out_recs[1][2] as usize * STRIDE;
        assert_eq!(&gpu_out[start1..start1 + 6 * STRIDE], &expected1[..]);
    }
}
