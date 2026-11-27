//! The console's display planes on a host with one swap chain. A title
//! that presents a second plane (an interface drawn at its own size over a
//! game drawn smaller, as in Series S mode) has it blended over the first
//! by a compute pass of ours, into a texture the size of the larger plane,
//! which the present then copies to the back buffer.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const SOURCE: &str = include_str!("../shaders/plane_compose.hlsl");

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn pass(device: &ID3D12Device) -> Option<&'static Pass> {
    static PASS: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
    PASS.get_or_init(|| {
        let made = crate::hlsl::compile_compute("plane compose", SOURCE).and_then(|bytes| {
            // SAFETY: plain object creation; the root signature is inside the shader.
            unsafe {
                let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytes)?;
                let pso =
                    device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes.as_ptr().cast(),
                            BytecodeLength: bytes.len(),
                        },
                        ..Default::default()
                    })?;
                Ok(Pass { root, pso })
            }
        });
        made.inspect_err(|e| {
            tracing::warn!(error = %e, "no plane compose pass: only the first plane is shown");
        })
        .ok()
    })
    .as_ref()
}

/// The composed frame and its views, kept from one present to the next
/// (every submission is waited for, so the next present may rewrite them).
pub(crate) struct Composed {
    texture: ID3D12Resource,
    heap: ID3D12DescriptorHeap,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
}

/// The size of the output for planes `base` and `overlay`: the larger of
/// each dimension.
pub(crate) fn size(base: &D3D12_RESOURCE_DESC, overlay: &D3D12_RESOURCE_DESC) -> (u32, u32) {
    (
        (base.Width as u32).max(overlay.Width as u32),
        base.Height.max(overlay.Height),
    )
}

/// The format a plane is read as: its typed, linear form (the bytes go to
/// the output as they are, as the copy of a single plane does).
fn read_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format.0 {
        27 | 29 => DXGI_FORMAT_R8G8B8A8_UNORM,
        23 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        _ => format,
    }
}

/// Moves `resource` from `from` to `to` in `list`, if they differ.
///
/// # Safety
/// `list` is open and `resource` is live in state `from`.
unsafe fn barrier(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    from: D3D12_RESOURCE_STATES,
    to: D3D12_RESOURCE_STATES,
) {
    if from != to {
        // SAFETY: as the caller promises.
        unsafe { list.ResourceBarrier(&[crate::host::transition(resource, from, to)]) };
    }
}

/// Records in `list` the blend of plane `overlay` over plane `base` (each
/// with the host state it is in, which it is left in) into a texture of
/// `format` (the swap chain's), returned in the copy-source state. `None`
/// (nothing recorded) if the pass or its texture cannot be made, or the
/// host cannot write `format` from a shader (BGRA8 on NVIDIA).
pub(crate) fn compose(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    kept: &mut Option<Composed>,
    (base, base_state): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    (overlay, overlay_state): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    format: DXGI_FORMAT,
) -> Option<ID3D12Resource> {
    if format == DXGI_FORMAT_B8G8R8A8_UNORM {
        // shortcut: no typed UAV store of BGRA8 on every host; one plane.
        return None;
    }
    let pass = pass(device)?;
    // SAFETY: live resources.
    let (b, o) = unsafe { (base.GetDesc(), overlay.GetDesc()) };
    let (width, height) = size(&b, &o);
    if kept
        .as_ref()
        .is_none_or(|k| (k.width, k.height, k.format) != (width, height, format))
    {
        *kept = None;
        *kept = Some(make(device, width, height, format)?);
    }
    let k = kept.as_ref()?;
    let srv = |resource: &ID3D12Resource, desc: &D3D12_RESOURCE_DESC, slot: u32| {
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: read_format(desc.Format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a live resource and a slot of our heap.
        unsafe {
            device.CreateShaderResourceView(resource, Some(&view), cpu(device, &k.heap, slot))
        };
    };
    srv(base, &b, 0);
    srv(overlay, &o, 1);
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    // SAFETY: an open list; our live objects and the planes in the states
    // the caller names.
    unsafe {
        device.CreateUnorderedAccessView(&k.texture, None, Some(&uav), cpu(device, &k.heap, 2));
        barrier(list, base, base_state, READ);
        barrier(list, overlay, overlay_state, READ);
        barrier(
            list,
            &k.texture,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        list.SetDescriptorHeaps(&[Some(k.heap.clone())]);
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        list.SetComputeRootDescriptorTable(0, k.heap.GetGPUDescriptorHandleForHeapStart());
        let size = [width, height];
        list.SetComputeRoot32BitConstants(1, 2, size.as_ptr().cast(), 0);
        list.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
        barrier(
            list,
            &k.texture,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        barrier(list, base, READ, base_state);
        barrier(list, overlay, READ, overlay_state);
    }
    Some(k.texture.clone())
}

/// Slot `slot` of `heap` for the CPU.
fn cpu(
    device: &ID3D12Device,
    heap: &ID3D12DescriptorHeap,
    slot: u32,
) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    // SAFETY: plain queries of live objects.
    unsafe {
        let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let start = heap.GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + (slot * step) as usize,
        }
    }
}

fn make(device: &ID3D12Device, width: u32, height: u32, format: DXGI_FORMAT) -> Option<Composed> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width),
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: plain object creation; out pointers are ours.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                None,
                &mut texture,
            )
            .inspect_err(|e| tracing::warn!(error = %e, width, height, "no composed plane texture"))
            .ok()?;
        let views: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .ok()?;
        tracing::info!(
            width,
            height,
            format = format.0,
            "presenting two planes composed"
        );
        Some(Composed {
            texture: texture?,
            heap: views,
            width,
            height,
            format,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_output_is_as_large_as_the_larger_plane() {
        let desc = |w: u64, h: u32| D3D12_RESOURCE_DESC {
            Width: w,
            Height: h,
            ..Default::default()
        };
        assert_eq!(size(&desc(1920, 1080), &desc(3840, 2160)), (3840, 2160));
        assert_eq!(size(&desc(3840, 2160), &desc(1920, 1080)), (3840, 2160));
    }

    #[test]
    fn planes_are_read_in_their_linear_typed_format() {
        assert_eq!(read_format(DXGI_FORMAT(29)), DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(read_format(DXGI_FORMAT(27)), DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(read_format(DXGI_FORMAT(28)), DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(read_format(DXGI_FORMAT(91)), DXGI_FORMAT_B8G8R8A8_UNORM);
    }
}
