//! Compositing of a `PresentX` call with several planes (after upstream
//! #211, Yaekai). A console puts a UI plane over a scene plane; the title
//! of upstream #841 presents its start screens on the UI plane, over a
//! scene plane that stays black there (a black window when only the first
//! plane was shown). Plane 0 is drawn opaque and scaled to the window,
//! planes 1.. over it with premultiplied alpha.
//!
//! shortcut: whole planes, no source rectangles, placement or per-plane
//! color spaces.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::transition;

/// Planes a present composites at most.
pub(crate) const MAX_PLANES: usize = 8;

/// Host states a plane is sampled in.
const SAMPLED: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);

const SOURCE: &str = r#"
Texture2D tex : register(t0);
SamplerState smp : register(s0);
void vs(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD) {
    uv = float2((id << 1) & 2, id & 2);
    pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target {
    return tex.Sample(smp, uv);
}
"#;

/// A fullscreen-triangle pass that draws a plane into the back buffer.
pub(crate) struct Compositor {
    pub(crate) format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    opaque: ID3D12PipelineState,
    overlay: ID3D12PipelineState,
    rtvs: ID3D12DescriptorHeap,
    srvs: ID3D12DescriptorHeap,
    srv_step: u32,
}

fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: a static source and names; out pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("present_composite"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("present compositor: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn bytes(blob: &ID3DBlob) -> D3D12_SHADER_BYTECODE {
    // SAFETY: the blob's own bytes, alive while it is.
    unsafe {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: blob.GetBufferPointer(),
            BytecodeLength: blob.GetBufferSize(),
        }
    }
}

impl Compositor {
    pub(crate) fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        let vs = compile(s!("vs"), s!("vs_5_1"))?;
        let ps = compile(s!("ps"), s!("ps_5_1"))?;
        // Built here: Wine's compiler embeds no root signature.
        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            ..Default::default()
        };
        let parameter = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
            MaxLOD: f32::MAX,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        };
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 1,
            pParameters: &parameter,
            NumStaticSamplers: 1,
            pStaticSamplers: &sampler,
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: descriptions live through the calls.
        unsafe {
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
            let blob = blob.ok_or_else(windows::core::Error::empty)?;
            let root: ID3D12RootSignature = device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                ),
            )?;
            let pipeline = |blend: bool| -> windows::core::Result<ID3D12PipelineState> {
                let mut target = D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                if blend {
                    target.BlendEnable = true.into();
                    target.SrcBlend = D3D12_BLEND_ONE;
                    target.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    target.BlendOp = D3D12_BLEND_OP_ADD;
                    target.SrcBlendAlpha = D3D12_BLEND_ONE;
                    target.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                    target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                let mut blend_state = D3D12_BLEND_DESC::default();
                blend_state.RenderTarget[0] = target;
                let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
                formats[0] = format;
                device.CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    VS: bytes(&vs),
                    PS: bytes(&ps),
                    BlendState: blend_state,
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: false.into(),
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: formats,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                })
            };
            let opaque = pipeline(false)?;
            let overlay = pipeline(true)?;
            let rtvs = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })?;
            let srvs = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: MAX_PLANES as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            let srv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Self {
                format,
                root,
                opaque,
                overlay,
                rtvs,
                srvs,
                srv_step,
            })
        }
    }

    /// Records drawing `planes` (live host textures, each in its host
    /// state, left in it) into `back` (in the present state, left in it),
    /// plane 0 opaque, the rest over it. `format_of` names the format a
    /// plane is read in (none: it cannot be). The caller runs the list
    /// before this is called again.
    pub(crate) fn record(
        &self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        planes: &[(&ID3D12Resource, u32)],
        format_of: impl Fn(DXGI_FORMAT) -> Option<DXGI_FORMAT>,
    ) {
        let planes = &planes[..planes.len().min(MAX_PLANES)];
        // SAFETY: our heaps, an open list and live resources.
        unsafe {
            let rtv = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(back, None, rtv);
            let cpu = self.srvs.GetCPUDescriptorHandleForHeapStart();
            let gpu = self.srvs.GetGPUDescriptorHandleForHeapStart();
            let mut drawn = Vec::with_capacity(planes.len());
            let mut before = Vec::with_capacity(planes.len() + 1);
            for (i, &(texture, state)) in planes.iter().enumerate() {
                let d = texture.GetDesc();
                let Some(format) = format_of(d.Format) else {
                    continue;
                };
                if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1 {
                    continue;
                }
                // The same texture on two planes: drawn (and moved) once.
                if planes[..i].iter().any(|&(t, _)| t == texture) {
                    continue;
                }
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + i * self.srv_step as usize,
                };
                device.CreateShaderResourceView(
                    texture,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: 1,
                                PlaneSlice: 0,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    }),
                    handle,
                );
                let state = D3D12_RESOURCE_STATES(state as i32);
                if state != SAMPLED {
                    before.push(transition(texture, state, SAMPLED));
                }
                drawn.push((i, texture, state));
            }
            before.push(transition(
                back,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ));
            list.ResourceBarrier(&before);
            // Black under a scene plane that does not cover the window.
            list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            list.SetDescriptorHeaps(&[Some(self.srvs.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            for &(i, _, _) in &drawn {
                list.SetPipelineState(if i == 0 { &self.opaque } else { &self.overlay });
                list.SetGraphicsRootDescriptorTable(
                    0,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: gpu.ptr + i as u64 * u64::from(self.srv_step),
                    },
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            let mut after: Vec<_> = drawn
                .iter()
                .filter(|&&(_, _, state)| state != SAMPLED)
                .map(|&(_, texture, state)| transition(texture, SAMPLED, state))
                .collect();
            after.push(transition(
                back,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            ));
            list.ResourceBarrier(&after);
        }
    }
}
