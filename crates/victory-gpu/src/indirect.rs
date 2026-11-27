//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! [`crate::hlsl`], with its root signature inside).
//! Non-indexed rectangle-list draws follow the clamp with GPU-side
//! rectangle expansion in [`crate::rectangles`].
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};

const SOURCE: &str = include_str!("../shaders/argument_clamp.hlsl");
const MESH_SOURCE: &str = include_str!("../shaders/mesh_draws.hlsl");

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make("argument clamp", SOURCE) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

/// A compute pass of ours from HLSL `source`, named `name` in messages.
fn make(name: &str, source: &str) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    // Wine's d3dcompiler_47 refuses it; `compile_compute` then uses DXC.
    let bytes = crate::hlsl::compile_compute(name, source)?;
    let bytes = &bytes[..];
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
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu`, leaving
/// the buffer in unordered-access state with a UAV barrier after the write
/// (for passes such as [`crate::rectangles`] that read the clamped records
/// in another compute shader). Replaces the list's compute root signature
/// and pipeline (callers bind theirs after). False if the clamp is not
/// available.
pub(crate) fn clamp_uav(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    true
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    spec: (u32, u32, u32),
) -> bool {
    if !clamp_uav(list, gpu, spec) {
        return false;
    }
    list.use_shadow(buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    true
}

fn mesh_pass() -> Option<&'static Clamp> {
    static PASS: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    PASS.get_or_init(|| match make("mesh draw arguments", MESH_SOURCE) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(
                error = %e,
                "no mesh draw arguments: indirect mesh dispatches dropped on this host"
            );
            None
        }
    })
    .as_ref()
}

/// Bytes of a draw's and of a dispatch's arguments.
pub(crate) const DRAW_BYTES: u32 = 16;
pub(crate) const DISPATCH_BYTES: u32 = 12;

/// Records, in `list`, the pass that makes the arguments of an indirect
/// mesh dispatch on a host without mesh shaders: from the first clamped
/// record at host address `args` (in unordered-access state, as
/// [`clamp_uav`] leaves it) it writes, at host address `made`, `parts`
/// draws [`DRAW_BYTES`] apart and after them `parts` dispatches
/// [`DISPATCH_BYTES`] apart. Part k covers the record's thread groups from
/// `k * capacity`, at most `capacity` of them: its dispatch runs that many
/// groups and its draw has `vertices` per instance and one instance per
/// group. Replaces the list's compute root signature and pipeline. False if
/// the pass is not available.
pub(crate) fn mesh_draws(
    list: &HostList,
    args: u64,
    made: u64,
    (parts, capacity, vertices): (u32, u32, u32),
) -> bool {
    let Some(c) = mesh_pass() else {
        return false;
    };
    let l = &list.list;
    let constants = [parts, capacity, vertices, 0];
    // SAFETY: an open list; our root signature, pipeline and live buffers
    // the caller's list holds, both in unordered-access state.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, args);
        l.SetComputeRootUnorderedAccessView(2, made);
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    true
}

pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// The arguments of an indirect mesh dispatch without mesh shaders: the
    /// first record's thread groups are split over the parts, each with its
    /// draw and its dispatch, and the title's records stay as they are.
    #[test]
    fn mesh_draw_arguments_split_the_first_record() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // (first record, what each of 3 parts of 1000 groups runs).
        let cases: [([u32; 3], [u32; 3]); 4] = [
            ([2500, 1, 1], [1000, 1000, 500]),
            ([1000, 1, 1], [1000, 0, 0]),
            ([5000, 1, 1], [1000, 1000, 1000]),
            ([700, 0, 1], [0, 0, 0]),
        ];
        for (first, want) in cases {
            let list = host.new_list().expect("host command list");
            let (args, args_at) = host.shared_buffer(256).expect("records");
            let (made, made_at) = host.shared_buffer(256).expect("made arguments");
            let records: [[u32; 3]; 3] = [first, [7, 1, 1], [9, 1, 1]];
            // SAFETY: freshly mapped host buffers of 256 bytes; live buffers.
            let (args_gpu, made_gpu) = unsafe {
                std::ptr::copy_nonoverlapping(
                    records.as_ptr().cast::<u8>(),
                    args_at,
                    size_of_val(&records),
                );
                std::ptr::write_bytes(made_at, 0xaa, 256);
                (args.GetGPUVirtualAddress(), made.GetGPUVirtualAddress())
            };
            // Three parts of 1000 groups, 768 vertices each.
            assert!(super::mesh_draws(&list, args_gpu, made_gpu, (3, 1000, 768)));
            list.close();
            host.execute(&[&list]);
            // SAFETY: execution finished; the buffers are mapped and readable.
            let (after, draws, dispatches) = unsafe {
                (
                    args_at.cast::<[[u32; 3]; 3]>().read_unaligned(),
                    made_at.cast::<[[u32; 4]; 3]>().read_unaligned(),
                    made_at.add(48).cast::<[[u32; 3]; 3]>().read_unaligned(),
                )
            };
            assert_eq!(after, records, "{first:?}: the title's records");
            assert_eq!(draws, want.map(|n| [768, n, 0, 0]), "{first:?}: draws");
            let empty_or = |n: u32| if n == 0 { [0; 3] } else { [n, 1, 1] };
            assert_eq!(dispatches, want.map(empty_or), "{first:?}: dispatches");
        }
    }
}
