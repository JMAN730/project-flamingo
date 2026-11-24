# Wave size rules on host D3D12

Research for ticket 40. Sources were read on 2026-11-24. Evidence is limited to primary sources:
the DirectX and HLSL specifications, Microsoft Learn, the DirectX developer blog, the DXIL
validator and compiler tables in Microsoft's DirectXShaderCompiler repository (pinned to commit
`b43c5651`), and vendor slide decks. Device lane counts have no vendor or Microsoft table; the
figures in section 3 that come from a community database are labelled as such.

Confidence marks:

- **specified**: stated in a DirectX-Specs or hlsl-specs document, or in the D3D11.3 functional
  spec that D3D12 shader semantics inherit.
- **documented**: stated on Microsoft Learn, the DirectX developer blog, the compiler wiki, or
  in a vendor's own published material.
- **observed**: read directly from source code (the DXIL validator, the compiler's operation
  table, vkd3d-proton, this repository) or from community-submitted device reports. Not stated
  as a rule by Microsoft or a vendor.
- **unknown**: looked for and not found in a primary source.

Statements marked *reading* are this note's interpretation of a source, not the source's words.
A proposal's status (completed, accepted, under consideration) is given where it matters.

## What the translator does today

From `crates/victory-shader/src/emit/` at `origin/main` (`93f2c1fa`):

- Output is shader model 6.6 DXIL (`dx.version` and `dx.valver` 1.6, `shader.rs`).
- A compute shader gets a wave size when it depends on the console's lane layout
  (`console_lanes`: cross-lane instructions, lane masks read as numbers, implicit derivatives)
  or when its whole thread group fits one console wave and it uses group-shared memory, where
  the console compiler leaves out barriers. The size is the console's: 64 or 32
  (`lower.rs`, `Lower::new`).
- The size is written as entry property tag 11 with one integer (`shader.rs`) and as the
  minimum and maximum expected lane counts in `PSV0` (`container.rs`); without a wave size those
  two fields are 0 and `0xFFFFFFFF`.
- Nothing queries the host's lane counts. No `WaveLaneCountMin`/`WaveLaneCountMax` read exists
  in `crates/`.
- With `console_lanes`, the thread id is rebuilt from the flat id of the wave's first active
  lane, rounded down to the wave size, plus the lane index (`lower.rs`, `thread_id`). The code
  comment names the assumption: each host wave holds a wave-aligned run of flat ids.
- Mesh shaders declare no wave size and reject every cross-lane operation (`lower.rs`,
  `cross_lane`). Pixel shaders allow quad-scoped ones. Pixel and vertex shaders that read lane
  masks as data use the host wave's own ballots.

## Facts

### 1. The fixed `WaveSize` attribute

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 1.1 | `[WaveSize(N)]` was added in shader model 6.6. `N` must be an immediate integer that is a power of two from 4 to 128: 4, 8, 16, 32, 64 or 128. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), Allowed Wave Sizes, HLSL Attribute | specified |
| 1.2 | It is allowed on compute shader entry points only. The spec's issue list first resolved to support every non-raytracing stage, then narrowed three later issues to "compute shader stage" only; the body text and the validator follow the narrow reading. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), HLSL Attribute, Issues 1 and 3 to 5 | specified |
| 1.3 | The DXIL validator rejects a wave size on any entry that is not compute or a work-graph node (rule `SM.WAVESIZEONCOMPUTEORNODE`). Its own comment says other stages might be supported in the future. | [`DxilValidation.cpp`, `ValidateWaveSize`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/lib/DxilValidation/DxilValidation.cpp#L6931-L6991); [`DXIL.rst` rule table](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/docs/DXIL.rst?plain=1#L3471-L3482) | observed |
| 1.4 | In DXIL the size is entry property tag 11 (`kDxilWaveSizeTag`) holding one `i32`. The validator accepts that tag only for shader models 6.6 and 6.7 (rule `SM.WAVESIZENEEDSSM66OR67`), requires exactly one constant operand, and allows one wave size tag per entry point. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), DXIL Metadata; [`DxilValidation.cpp`, `ValidateWaveSize` (metadata)](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/lib/DxilValidation/DxilValidation.cpp#L5233-L5336) | specified, observed |
| 1.5 | The requirement also travels in the container's `PSV0` runtime data as `MinimumExpectedWaveLaneCount` and `MaximumExpectedWaveLaneCount` (0 and `0xFFFFFFFF` when unused). A single wave size sets both to that value. The range proposal says this pair is what tells the runtime, as distinct from the driver. | [`DxilPipelineStateValidation.h`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/include/dxc/DxilContainer/DxilPipelineStateValidation.h#L120-L124); [0013 Wave Size Range](https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range/), Motivation, Runtime information | observed, specified |
| 1.6 | The device must run the shader at the declared size. Every power of two inside the range the driver reports must be selectable, with wave operations working at each. A size the hardware can use in some conditions but cannot offer to compute shaders with wave intrinsics should be left out of the reported range. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), Device Behavior | specified |
| 1.7 | A device that reports shader model 6.6 and the `WaveOps` feature must support the attribute. There is no separate feature bit. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), Device Capability, Issue 2 | specified |
| 1.8 | Without the attribute an application cannot guarantee which size in the reported range a shader runs at. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), introduction | specified |
| 1.9 | Microsoft Learn has no reference page for the attribute; a search of learn.microsoft.com returned only the spec pages above. | search on 2026-11-24 | observed |

### 2. Ranged wave size

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 2.1 | Shader model 6.8 adds `[WaveSize(min, max [, preferred])]`. The proposal's status is Completed. Each value is a power of two from 4 to 128, `min` is below `max`, and `preferred` lies within them. The one-value form still exists. | [0013 Wave Size Range](https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range/); [HLSL Shader Model 6.8](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_8.html), Wave Size Range | specified |
| 2.2 | DXIL encoding: entry property tag 23 (`kDxilRangedWaveSizeTag`) with three `i32` values (min, max, preferred), zero meaning absent. From 6.8 onward the old tag 11 is invalid and a single size is written as (N, 0, 0) under tag 23. Tag 23 is invalid below 6.8. | [0013 Wave Size Range](https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range/), DXIL Additions, Validation Changes; [`DxilValidation.cpp`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/lib/DxilValidation/DxilValidation.cpp#L5233-L5336) | specified, observed |
| 2.3 | `PSV0` is unchanged: min and max go in the existing two fields. The preferred value is not sent to the runtime because it cannot make a shader fail to load. | [0013 Wave Size Range](https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range/), Runtime information | specified |
| 2.4 | Devices supporting shader model 6.8 must respect the range. If the preferred size is available it is used; otherwise the driver chooses among the sizes both sides support. The proposal's stated purpose is that a shader loads wherever some size in its range is supported. | [0013 Wave Size Range](https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range/), Proposed solution, Device Capability | specified |
| 2.5 | The proposal does not list stages. Its examples and tests use compute entry points, and the validator rule in 1.3 applies to the ranged form too. | [0013 Wave Size Range](https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range/); 1.3 | specified (silent), observed |
| 2.6 | Nothing reports which size the driver picked for a pipeline. A later proposal (status: Under Consideration) would add a pipeline flag and a reported wave size for that reason. Inside the shader, `WaveGetLaneCount` returns it. | [0054 numWaves](https://microsoft.github.io/hlsl-specs/proposals/0054-numwaves/), Runtime information; [WaveGetLaneCount](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/wavegetlanecount) | specified (proposal), documented |

### 3. What devices report

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 3.1 | `D3D12_FEATURE_DATA_D3D12_OPTIONS1` carries `WaveOps`, `WaveLaneCountMin` (the baseline lane count; the page tells applications to size workloads by this value only), `WaveLaneCountMax` and `TotalLaneCount`. | [D3D12_FEATURE_DATA_D3D12_OPTIONS1](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options1) | documented |
| 3.2 | The compiler wiki used to call `WaveLaneCountMax` reserved for future use. A Microsoft member called that caveat stale on the compiler's tracker and it was removed. | [DirectXShaderCompiler issue 8535](https://github.com/microsoft/DirectXShaderCompiler/issues/8535); [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics), Caps Flags | documented |
| 3.3 | `WaveGetLaneCount` returns between 4 and 128, counts active, inactive and helper lanes, and can vary with the driver implementation. | [WaveGetLaneCount](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/wavegetlanecount) | documented |
| 3.4 | NVIDIA: waves are 32 wide, per a 2017 talk given jointly by AMD and NVIDIA engineers. No NVIDIA page stating what its D3D12 driver reports was found. | [Wave Programming in D3D12 and Vulkan, GDC 2017](https://gpuopen.com/download/GDC2017-Wave-Programming-D3D12-Vulkan.pdf) | documented (width); reported values undocumented |
| 3.5 | AMD: GCN hardware runs 64-lane waves. RDNA runs 32-lane waves natively and 64-lane waves by issuing each vector instruction twice. The driver's compiler picks per shader, usually 32 for compute and vertex and 64 for pixel, and the heuristics were expected to keep changing. | [RDNA Architecture, AMD](https://gpuopen.com/download/RDNA_Architecture_public.pdf), slides "Wave64 via dual-issue" and "Wave32 vs. Wave64" | documented |
| 3.6 | Intel: execution units run 8, 16 or 32 lanes. The compiler decides: vertex, hull, domain and geometry shaders at 8, pixel and compute at 8, 16 or 32, driven by register pressure; mesh shaders run at 8 or 16 by default. Intel tells developers not to assume a lane count. | [Developing for Intel Graphics, GDC 2021](https://www.intel.cn/content/dam/develop/external/us/en/documents/pdf/july-gdc-2021-developing-for-intel-graphics-today-and-into-the-future.pdf), slides "EU SIMD Explained", mesh shading, "Call to Action" | documented |
| 3.7 | Community-reported, not verified here: the D3d12infoDB database of user-submitted `CheckFeatureSupport` dumps (1,689 submissions on 2026-11-24) shows these `WaveLaneCountMin`/`WaveLaneCountMax` pairs. **NVIDIA:** 32/32 in all 509 reports, GeForce GT 710 through RTX 5090. **AMD:** 32/64 in 511 reports covering RDNA parts (RX 5500 XT through RX 9070 XT and integrated 610M to 890M); 64/64 in 56 reports covering GCN and Vega parts (R7 200 through RX 590, Vega, Radeon VII); 12 further reports under AMD's vendor id show 32/32 with adapter names that are not Radeon product names. **Intel:** 16/16 in 95 reports (HD 530 through UHD 770, Iris Xe); 8/32 in 45 reports (Arc A310 to A770 and newer integrated Arc); 16/32 in 53 reports (Arc B570, B580, 140V, B390). **Microsoft Basic Render Driver (WARP):** 4/4 in 312 reports, 4/128 in 70 newer ones. **Qualcomm Adreno:** 64/128 in 19 reports, 64/64 in 1. | [D3d12infoDB](https://d3d12infodb.boolka.dev/FeatureTable.html), all submissions aggregated | observed (community-reported) |
| 3.8 | *Reading* of 3.7 against 1.6: a declared size of 64 lies inside the reported range only on AMD, Qualcomm and the newer WARP; a declared size of 32 lies inside it on NVIDIA, AMD RDNA and Intel Arc, and outside it on AMD GCN and Vega, on Intel integrated parts reporting 16/16, on Qualcomm, and on the older WARP. | 3.7, 1.6 | observed (community-reported) |
| 3.9 | The reported range is defined against compute shaders (1.6). Whether it bounds the wave size of other stages is not stated. A third party (Chromium's Dawn) records Intel devices whose pixel shaders run 8 lanes while the minimum reports 16, and treats it as a driver issue. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), Device Behavior; [Dawn, subgroup size control](https://dawn.googlesource.com/dawn/+/HEAD/docs/tint/extensions/subgroup_size_control.md) | specified (compute); other stages unknown; Dawn observed (third party) |
| 3.10 | Under vkd3d-proton the two counts are Vulkan's `minSubgroupSize` and `maxSubgroupSize`, and shader model 6.6 is offered only if the Vulkan driver can require a subgroup size for compute (or has a single size). Some of the 32/64 AMD reports in 3.7 come from that path (adapter names ending in "RADV"). | [vkd3d-proton `device.c`](https://github.com/HansKristian-Work/vkd3d-proton/blob/ec20719452e916baa2b9e69f33f18de4561a8313/libs/vkd3d/device.c#L9343-L9345), [`state.c`](https://github.com/HansKristian-Work/vkd3d-proton/blob/ec20719452e916baa2b9e69f33f18de4561a8313/libs/vkd3d/state.c#L3175-L3275) | observed |
| 3.11 | Highest shader model in the same community reports (a lower bound: the tool can only report up to the SDK it was built with): 6.8 or higher in 484 of 509 NVIDIA reports, 488 of 511 AMD 32/64 reports, 12 of 56 AMD 64/64 reports, 41 of 53 Intel 16/32, 29 of 45 Intel 8/32 and 26 of 95 Intel 16/16 reports. | [D3d12infoDB](https://d3d12infodb.boolka.dev/FeatureTable.html) | observed (community-reported) |

### 4. Pipeline creation with a size outside the device's range

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 4.1 | The D3D12 runtime validates a shader's wave size against the range the device reports, and validation fails when a shader in a pipeline state object requires a size outside that range. *Reading:* the check belongs to the runtime and happens when the pipeline state is created, before the driver compiles anything. | [HLSL Wave Size](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html), introduction, Runtime Validation | specified |
| 4.2 | For a range, the shader fails to load only when no size in it is supported. | [0013 Wave Size Range](https://microsoft.github.io/hlsl-specs/proposals/0013-wave-size-range/), Introduction, Runtime information | specified |
| 4.3 | The returned `HRESULT` and any debug-layer message are not given by either spec. The public `d3d12sdklayers.h` has no message id naming wave size or lane count. | specs above; [`d3d12sdklayers.h`](https://github.com/microsoft/DirectX-Headers/blob/main/include/directx/d3d12sdklayers.h) | unknown |
| 4.4 | The check can be made ahead of creation with `CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1)`: the size must satisfy `WaveLaneCountMin <= N <= WaveLaneCountMax`. *Reading* of 1.6, 3.1 and 4.1. | 1.6, 3.1, 4.1 | specified (reading) |
| 4.5 | A shader that uses wave intrinsics on a device without `WaveOps` fails at shader creation. | [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics), Caps Flags | documented |

### 5. Wave intrinsics and helper lanes

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 5.1 | Helper lanes exist only in pixel shaders: lanes of a scheduled quad that are not covered or were discarded. They exist only while a later quad or derivative operation needs them, so a shader without such operations may have none. `IsHelperLane` (6.6) is false in every other stage. | [WaveOpsIncludeHelperLanes](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_7_Wave_Ops_Include_Helper_Lanes.html); [HLSL Shader Model 6.6](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_6.html), IsHelperLane | specified |
| 5.2 | By default helper lanes take no part in wave operations: votes, ballots, reductions, prefix operations, `WaveReadLaneFirst` and `WaveIsFirstLane` consider active non-helper lanes only. Ballot bits of inactive and helper lanes are zero, as are bits at or above the lane count. Values these operations return to a helper lane are undefined. | [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics), Shading Language Intrinsics; [WaveOpsIncludeHelperLanes](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_7_Wave_Ops_Include_Helper_Lanes.html) | documented, specified |
| 5.3 | Exceptions: `WaveGetLaneCount` and `WaveGetLaneIndex` work on helper lanes, and quad reads may read from and return to helper lanes. | [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics), Wave Query, Quad-Wide Shuffle | documented |
| 5.4 | `WaveReadLaneAt` is undefined on a helper lane and when the lane it names is inactive or a helper. An index at or above the lane count is not addressed. | [WaveReadLaneAt](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/wavereadlaneat); [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics) | documented; out-of-range index unknown |
| 5.5 | Shader model 6.7 adds `[WaveOpsIncludeHelperLanes]`, valid on pixel shader entry points only. Helper lanes then take part in every wave operation and are kept alive until the last one. It is stored as a feature flag and devices must honour it. It never creates helper lanes. | [WaveOpsIncludeHelperLanes](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_7_Wave_Ops_Include_Helper_Lanes.html) | specified |
| 5.6 | The Learn overview's glossary says an active lane in a pixel shader "may include" helper lanes, which reads against 5.2. The 6.7 spec and the wiki are the later and more specific texts. | [HLSL Shader Model 6.0](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12), Terminology | documented (conflict) |
| 5.7 | Uniform results: a reduction returns the same value to every lane of the wave. Sums and products have no defined order of operations, so floating-point results need not match a fixed order. Prefix operations cover active non-helper lanes with a lower lane index; the lowest such lane gets the identity. | [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics), Wave Reduction, Wave Scan and Prefix | documented |
| 5.8 | Wave operations are valid DXIL in every stage; quad operations in compute, mesh, amplification, pixel and node. | [`hctdb.py`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/utils/hct/hctdb.py#L776-L808); [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics) | observed, documented |

### 6. Several waves in one thread group

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 6.1 | Every wave intrinsic is scoped to the current wave. A thread group larger than the wave runs as several waves, and no intrinsic before shader model 6.10 reduces, counts, prefixes or reads across them. | [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics); [0048 Group Wave Index](https://microsoft.github.io/hlsl-specs/proposals/0048-group-wave-index/), Motivation | documented, specified |
| 6.2 | Lanes of one wave run together without barriers. Across waves of a group that does not hold: Intel warns of races whenever the lane count differs from the thread-group size, and AMD's driver compiler removes barriers it judges unnecessary, so barriers must be semantically right. | [D3D12_FEATURE_DATA_D3D12_OPTIONS1](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options1), Remarks; [Intel GDC 2021](https://www.intel.cn/content/dam/develop/external/us/en/documents/pdf/july-gdc-2021-developing-for-intel-graphics-today-and-into-the-future.pdf); [RDNA Architecture](https://gpuopen.com/download/RDNA_Architecture_public.pdf) | documented |
| 6.3 | Which threads share a wave is not specified. Microsoft's own proposal calls `SV_GroupIndex / WaveGetLaneCount()` an unsafe way to find the wave, not guaranteed across hardware and thread-group dimensions. | [0048 Group Wave Index](https://microsoft.github.io/hlsl-specs/proposals/0048-group-wave-index/), Motivation | specified |
| 6.4 | What is specified about lane order in compute, from shader model 6.6 on: lanes with the same `WaveGetLaneIndex() / 4` form one quad. For a one-dimensional group (X divisible by 4, Y and Z equal to 1), `SV_GroupIndex % 4` equals `WaveGetLaneIndex() % 4`. For a two-dimensional group (X and Y divisible by 2), quads follow `SV_GroupThreadID` and no relation between `SV_GroupIndex` and the lane index may be assumed. The order of quads within a wave is implementation dependent. Before 6.6 the mapping of threads to quads was implementation dependent altogether. | [Derivative Operations in Compute, Mesh and Amplification Shaders](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Derivatives.html), Thread Groups and Quads, Issue 4; [HLSL Shader Model 6.0](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12), Quad-wide Shuffle operations | specified |
| 6.5 | A wave may be partly filled: a lane can be inactive because there is not enough work to fill it. An implementation may launch more waves than the thread count divided by the lane count requires. | [Wave Intrinsics wiki](https://github.com/microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics), Glossary; [0048 Group Wave Index](https://microsoft.github.io/hlsl-specs/proposals/0048-group-wave-index/), Wave Ordering and Launch Flexibility | documented, specified |
| 6.6 | D3D12 has no stated counterpart of Vulkan's "require full subgroups" pipeline flag. Whether a group whose size is a multiple of the wave size always gets full waves holding consecutive thread ids is not stated anywhere found. *Reading:* the translator's `thread_id` assumption is of this kind. | none; [vkd3d-proton `state.c`](https://github.com/HansKristian-Work/vkd3d-proton/blob/ec20719452e916baa2b9e69f33f18de4561a8313/libs/vkd3d/state.c#L3268-L3275) for the Vulkan flag | unknown |
| 6.7 | Shader model 6.10 adds `GetGroupWaveIndex()` and `GetGroupWaveCount()` for compute, mesh, amplification and node shaders (proposal status: Accepted; DXIL opcodes 312 and 313). The count is at least the thread count divided by the lane count, rounded up, and is the same for every group of a dispatch. How indices are dealt to threads stays with the implementation. | [0048 Group Wave Index](https://microsoft.github.io/hlsl-specs/proposals/0048-group-wave-index/); [`hctdb.py`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/utils/hct/hctdb.py#L1074-L1079) | specified, observed |
| 6.8 | Shader model 6.10 shipped with a preview Agility SDK (1.720-preview, April 2026). The announcement lists the two intrinsics as supported on AMD RX 7000 and 9000 and Intel Arc B-series, and as planned on NVIDIA. No retail release was found. | [DirectX developer blog, Shader Model 6.10 preview](https://devblogs.microsoft.com/directx/shader-model-6-10-agilitysdk-720-preview/) | documented |
| 6.9 | The same proposal gives an emulation for earlier shader models: the first lane of each wave takes a number from an atomic counter in group-shared memory and the wave reads it back with `WaveReadLaneAt`. | [0048 Group Wave Index](https://microsoft.github.io/hlsl-specs/proposals/0048-group-wave-index/), Emulation | specified |
| 6.10 | The HLSL execution and memory model has never been fully defined; behaviour rests on device certification. A proposal to formalise it is under consideration. | [0050 Formalized Memory and Execution Model](https://microsoft.github.io/hlsl-specs/proposals/0050-mem-and-exec-model/), Motivation | specified (proposal) |

### 7. Thread-group limits and barrier rules

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 7.1 | Compute thread group: X and Y from 1 to 1024, Z from 1 to 64, and at most 1024 threads in total. | [numthreads](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-attributes-numthreads); [`DxilConstants.h`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/include/dxc/DXIL/DxilConstants.h#L126-L132) | documented, observed |
| 7.2 | Group-shared memory: at most 32,768 bytes per compute or amplification entry and 28,672 bytes per mesh entry, checked by the validator. | [`DxilConstants.h`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/include/dxc/DXIL/DxilConstants.h#L136-L153); [`DxilValidation.cpp`, `getMaxTGSM`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/lib/DxilValidation/DxilValidation.cpp#L4989-L4997); [Mesh Shader spec](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html) | observed, specified |
| 7.3 | Shader model 6.10 replaces that fixed cap with a limit the runtime reports; AMD's first drivers expose only the old default. | [DirectX developer blog, Shader Model 6.10 preview](https://devblogs.microsoft.com/directx/shader-model-6-10-agilitysdk-720-preview/) | documented |
| 7.4 | A thread-group sync holds every thread of the group at the instruction until all have reached it. It cannot be placed in flow control that can vary within a thread group; uniform flow control is allowed. | [D3D11.3 Functional Spec, 22.17.7 sync](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#inst_SYNC) | specified |
| 7.5 | `GroupMemoryBarrierWithGroupSync` in diverging branches of a thread group is undefined behaviour. | [GroupMemoryBarrierWithGroupSync](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/groupmemorybarrierwithgroupsync) | documented |
| 7.6 | The DXIL validator does not check where a sync sits in the control flow. It checks only that the barrier mode is a constant, that it names some memory (group-shared or UAV), and that stages other than compute, mesh, amplification and node use the global UAV fence alone. | [`DxilValidation.cpp`, `Barrier`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/lib/DxilValidation/DxilValidation.cpp#L3130-L3163) | observed |
| 7.7 | A memory fence without sync makes the thread's earlier accesses visible to the group before its later ones and stops reordering across it; it does not wait for other threads. Atomic operations on one address need no fence to be correct with each other. | [D3D11.3 Functional Spec, 22.17.7 sync](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#inst_SYNC) | specified |
| 7.8 | Mesh and amplification shaders may use group-shared memory and the same barriers. | [`DxilValidation.cpp`, `IsCSLike`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/lib/DxilValidation/DxilValidation.cpp#L2980-L2984); [Mesh Shader spec](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html) | observed, specified |

### 8. Mesh shaders

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 8.1 | A mesh or amplification shader may not declare a wave size, fixed or ranged, in any released shader model (1.2, 1.3, 2.5). | 1.2, 1.3, 2.5 | specified, observed |
| 8.2 | Allowing it is an open request on the HLSL specs tracker, filed under the shader model backlog with no version assigned. It needs a new shader model. | [hlsl-specs issue 137](https://github.com/microsoft/hlsl-specs/issues/137) | documented |
| 8.3 | Mesh and amplification groups hold at most 128 threads; a mesh shader outputs at most 256 vertices and 256 primitives. | [Mesh Shader spec](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html); [`DxilConstants.h`](https://github.com/microsoft/DirectXShaderCompiler/blob/b43c5651d744e606552b3274668a160be21db0f8/include/dxc/DXIL/DxilConstants.h#L138-L147) | specified, observed |
| 8.4 | Wave operations and, from 6.6, quad operations are valid in mesh shaders (5.8), at whatever size the driver chooses. Intel documents 8 or 16 lanes by default; AMD's and NVIDIA's choice for mesh shaders is not documented. | 5.8; 3.6 | observed, documented; AMD and NVIDIA unknown |
| 8.5 | The mesh shader spec does not mention wave size or wave intrinsics. | [Mesh Shader spec](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html) | observed |

## What this leaves open

What the rules above permit or forbid for each scheme. No scheme is recommended here.

**Declare the console's 64 and require a 64-lane device.**

- Permitted: 64 is a legal size, the attribute exists at the 6.6 level the translator already
  emits, and a device whose range includes 64 must run the shader at 64 (1.1, 1.6).
- Forbidden: creating the pipeline on a device whose range excludes 64; the runtime refuses it
  (4.1). By the community reports that is every NVIDIA and Intel device (3.7, 3.8).
- Open: the refusal can be predicted from `OPTIONS1` before creation (4.4), which the host
  layer does not do today. The error code is unknown (4.3). A matching size still leaves the
  thread-to-lane layout unspecified apart from the quad rules (6.3, 6.4, 6.6). Covers compute
  only (1.2).

**Declare 32 when provably safe.**

- Permitted: 32 is legal, and a device whose range includes 32 must honour it (1.1, 1.6). For
  console shaders already compiled for 32 lanes this is the existing behaviour.
- Forbidden: the same refusal where 32 is outside the range: by the community reports, AMD GCN
  and Vega (64/64), Intel integrated parts reporting 16/16, Qualcomm, older WARP (3.7, 3.8).
- Open: "provably safe" has to be established by the translator; the host rules only say that
  a 64-thread console wave then spans at least two host waves, that no wave intrinsic crosses
  them (6.1), that their co-execution is no longer implied (6.2), and that which threads share
  a host wave is unspecified (6.3).

**One console wave over two host waves with lane remapping.**

- Permitted: a thread group larger than the declared wave size is ordinary, and cross-wave
  exchange through group-shared memory with barriers is the documented mechanism, including
  Microsoft's own emulation of a wave index (6.9, 7.7). Mesh shaders have the same primitives
  (7.8).
- Bounded by: no wave index before 6.10, and 6.10 is preview with uneven driver support (6.7,
  6.8). Thread-to-wave assignment, wave fill and wave count are the implementation's (6.3, 6.5,
  6.6). A group sync is legal only where all threads of the group arrive, so a cross-lane
  operation inside flow control that diverges across the group has no legal sync point (7.4,
  7.5), and the validator will not flag a misplaced one (7.6). Exchange storage comes out of
  the 32,768-byte (mesh: 28,672-byte) budget that also holds the console shader's own shared
  memory (7.2). Reductions, ballots and prefix results must be recombined from per-wave
  results, in an order the translator defines itself (5.7, 6.1).
- Forbidden: relying on host wave intrinsics alone to see the other half (6.1).
- Open: in mesh shaders the host wave size cannot be fixed (8.1), so the number of host waves
  per console wave is not known at translation time.

**A wave size range.**

- Permitted: `[WaveSize(32, 64)]`, with or without a preferred size, from shader model 6.8, on
  compute entries (2.1, 2.5). The shader then loads on any device whose range overlaps, and a
  preferred size is used where available (2.4).
- Forbidden: the ranged tag below 6.8, and the old tag at 6.8 or above (2.2). The translator
  emits 6.6 today. Mesh shaders remain excluded (8.1).
- Open: the chosen size is known only inside the shader (2.6), so the shader must be correct
  at every size in its range the device offers. A range widens where a shader loads; it does
  not make a 32-lane device run 64 lanes. Host support for 6.8 was not surveyed beyond the
  lower bound in 3.11.

## Unknowns

- The `HRESULT` and debug message when a pipeline's wave size is outside the device range (4.3).
- Vendor statements of what each D3D12 driver reports in `OPTIONS1`; only community reports
  were found (3.7).
- Whether full, consecutive-id waves are guaranteed for a compute group whose size is a
  multiple of a declared wave size, on any vendor (6.6).
- The wave size drivers use for vertex, pixel and mesh shaders, and whether the `OPTIONS1`
  range bounds it (3.9, 8.4).
- `WaveReadLaneAt` with an index at or above the lane count (5.4).
- When, and on which drivers, shader model 6.10 leaves preview (6.8).
