# Host limits on descriptor heaps, and ways past one million views

Research for ticket 41. Sources were read on 2026-11-24. Only Microsoft sources count as
evidence here (Microsoft Learn, the DirectX specs, the public D3D12 headers, the DirectX
developer blog); the one community report is labelled as such.

Confidence marks:

- **specified**: stated in a DirectX spec or fixed by the public `d3d12.idl`.
- **documented**: stated on Microsoft Learn or the DirectX developer blog.
- **observed**: seen directly (a header at a given tag, a page that does not exist, this
  repository's own code comments), not stated as a rule by Microsoft.
- **unknown**: looked for and not found in a primary source.

Statements marked *reading* are this note's interpretation of a source, not the source's words.

## What the host layer does today

From `crates/victory-gpu/src/descriptor/host.rs` and `crates/victory-gpu/src/draw.rs`:

- Translated shaders index the heaps directly, so host slot `i` mirrors title slot `i`.
- Past the mirrored slots lies a dynamic range (`DYNAMIC` = 2^18 slots where the host
  accepts the larger heap, else `MIN_DYNAMIC` = 2^16) for tables a draw needs made afresh.
  A command list holds its dynamic slots until it is reset or released.
- If the heap cannot be created, it is capped at `PORTABLE_VIEWS` (1,000,000) and the title
  slots past the cap are left out. `MAX_VIEWS` is 2^24 because translated shaders send any
  index at or past it to slot 0.
- Sampler heaps are capped at `SAMPLERS` (2048), keeping `SAMPLER_DYNAMIC` (256) slots.
- A draw or dispatch whose fresh table gets no dynamic slots is dropped.

## Facts

### 1. Size of a shader-visible CBV/SRV/UAV heap

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 1.1 | The maximum is 1,000,000 descriptors on resource binding Tier 1 and Tier 2, and "1,000,000+" on Tier 3. | [Hardware Tiers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support); [Resource Binding spec, Levels of Hardware Support](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#levels-of-hardware-support) | documented, specified |
| 1.2 | The headers define `D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1` and `_TIER_2` as 1000000. There is no Tier 3 constant. | [`d3d12.idl`](https://github.com/microsoft/DirectX-Headers/blob/main/include/directx/d3d12.idl) | specified |
| 1.3 | The plus sign on Tier 3 means the runtime lets an application try a larger heap and leaves it to the driver to accept or fail the call. The spec text says no cap reports how large a heap the hardware takes, and tells applications to try and fall back to 1,000,000. | [Resource Binding spec, Levels of Hardware Support](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#levels-of-hardware-support) | specified |
| 1.4 | A later addition does expose a cap: `MaxViewDescriptorHeapSize` in `D3D12_FEATURE_DATA_D3D12_OPTIONS19`. Drivers at DDI version 0102 or newer must report it, and it must be at least 1,000,000. On those drivers, creating a larger shader-visible view heap fails. For older drivers the runtime reports 1,000,000, and on Tier 3 it still forwards larger requests to the driver, which may fail them. | [VulkanOn12 spec, Sampler descriptor heap size increase](https://microsoft.github.io/DirectX-Specs/d3d/VulkanOn12.html#sampler-descriptor-heap-size-increase) | specified |
| 1.5 | The `OPTIONS19` field is absent from the DirectX-Headers tag `v1.608.2` and present from `v1.610.0` onward. *Reading:* it needs a D3D12 runtime of the 1.610 generation or newer (an Agility SDK redistributable or a matching OS runtime); an older runtime rejects the query. | [`d3d12.idl` at `v1.610.0`](https://github.com/microsoft/DirectX-Headers/blob/v1.610.0/include/directx/d3d12.idl) | observed |
| 1.6 | Microsoft Learn has no reference page for the `OPTIONS19` structure (the expected URL returns 404), and the `D3D12_FEATURE` page lists the enumerator without a description. The spec in 1.4 is the only primary description found. | [D3D12_FEATURE](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_feature) | observed |
| 1.7 | The hardware certification test for this limit creates a heap of 1,000,000 SRVs, or 1.1 million on Tier 3, and renders to check every descriptor. *Reading:* 1.1 million is the only size above one million that certification is documented to exercise; nothing found guarantees the roughly 1,245,000 the title asks for. | [HLK: Resource Binding, Max Descriptor Heap Size](https://learn.microsoft.com/en-us/windows-hardware/test/hlk/testref/efad06e8-51d1-40ce-ad5c-573a134b4bb6) | documented |
| 1.8 | `CreateDescriptorHeap` returns `E_OUTOFMEMORY` when memory is insufficient. Passing a null output pointer is a capability test that returns `S_FALSE` for a valid description. Whether that probe consults the driver's size limit is not stated. | [ID3D12Device::CreateDescriptorHeap](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createdescriptorheap) | documented; probe behaviour unknown |
| 1.9 | Per-vendor maxima: no Microsoft source lists what any vendor's driver accepts or reports. | none | unknown |
| 1.10 | Community-reported, not verified: a feature-checker dump posted on the Beyond3D forum shows `MaxViewDescriptorHeapSize` values of 2,097,152 and 1,000,000; the adapter behind each value could not be confirmed (the page refused retrieval; only a search excerpt was seen). | Beyond3D forum, "Direct3D feature levels discussion" | community-reported |
| 1.11 | This repository's code comments record a host that would not take the larger dynamic range on top of the title's slots (named as NVIDIA at `MIN_DYNAMIC`), and 2^16 dynamic slots running out within seconds on a Radeon RX 9070 XT (at `DYNAMIC`). Not re-measured for this note. | `crates/victory-gpu/src/descriptor/host.rs` | observed (project) |
| 1.12 | Descriptor size is hardware specific and must be queried with `GetDescriptorHandleIncrementSize`. Microsoft's worked example assumes about 32 bytes, so a million-entry heap takes about 32 MB. | [Descriptors Overview](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptors-overview); [Shader Visible Descriptor Heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shader-visible-descriptor-heaps) | documented |
| 1.13 | Shader-visible heaps prefer CPU-writable GPU-local (write-combined) memory, which on some configurations is only about 96 MB for all processes. The video memory manager can demote heaps to system memory, at a cost in descriptor access latency. The OS sets no hard limit on an application's descriptor heap footprint. The spec says there is no clean way to report this limit through caps. | [Shader Visible Descriptor Heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shader-visible-descriptor-heaps); [Resource Binding spec, Shader Visible Descriptor Heap Pitfall](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#shader-visible-descriptor-heap-pitfall) | documented, specified |
| 1.14 | Non-shader-visible heaps are not constrained in size, and an application may create any number of heaps. | [Shader Visible Descriptor Heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shader-visible-descriptor-heaps); [Descriptor Heaps Overview](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview) | documented |

### 2. Sampler heap

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 2.1 | A shader-visible sampler heap holds at most 2048 samplers on every tier (`D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE` = 2048). | [Hardware Tiers, Invariable limits](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support); [`d3d12.idl`](https://github.com/microsoft/DirectX-Headers/blob/main/include/directx/d3d12.idl); [HLK: Max Sampler Descriptor Heap Size](https://learn.microsoft.com/en-us/windows-hardware/test/hlk/testref/94fce664-f2b2-42a9-bd12-53593b909831) | documented, specified |
| 2.2 | Static samplers are a separate pool: at most 2032 unique ones across live root signatures. | [Hardware Tiers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support); [Resource Binding spec, Limits Invariant on Hardware](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#limits-invariant-on-hardware) | documented, specified |
| 2.3 | `OPTIONS19` also reports `MaxSamplerDescriptorHeapSize`: at least 4000 on DDI 0102 or newer drivers, 2048 otherwise. Creating a larger shader-visible sampler heap fails. | [VulkanOn12 spec](https://microsoft.github.io/DirectX-Specs/d3d/VulkanOn12.html#sampler-descriptor-heap-size-increase) | specified |
| 2.4 | `MaxSamplerDescriptorHeapSizeWithStaticSamplers` (at least 2048): drawing or dispatching with a root signature that has static samplers while a sampler heap larger than this is bound is undefined behaviour and a debug layer error. | [VulkanOn12 spec](https://microsoft.github.io/DirectX-Specs/d3d/VulkanOn12.html#sampler-descriptor-heap-size-increase) | specified |
| 2.5 | Samplers reachable through descriptor tables per stage: 16 on Tier 1, 2048 on Tier 2 and Tier 3 (the spec table says "full heap" for Tier 2 and 3). | [Hardware Tiers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support) | documented |

### 3. `SetDescriptorHeaps` inside a command list

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 3.1 | Switching heaps within one command list is allowed. | [Descriptor Heaps Overview, Switching heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview) | documented |
| 3.2 | At most one CBV/SRV/UAV heap and one sampler heap are bound at a time, shared by graphics and compute. Each call unsets all previously set heaps, including a type the call does not name. | [SetDescriptorHeaps](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-setdescriptorheaps); [Resource Binding spec, DDI Setting Descriptor Heaps](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#ddi-setting-descriptor-heaps) | documented, specified |
| 3.3 | Descriptor table state is undefined at the start of a command list and after the heaps change. Setting the same heap again does not make it undefined. The spec does not say that other root arguments (root constants, root descriptors) are affected. | [Resource Binding spec, Setting Descriptor Heaps](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#setting-descriptor-heaps) | specified |
| 3.4 | Cost: changing heaps can incur a pipeline flush on some hardware, described elsewhere as a GPU stall or wait for idle so that work depending on the previous heap finishes. No figure is given. | [SetDescriptorHeaps](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-setdescriptorheaps); [Descriptor Heaps Overview](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview) | documented; magnitude unknown |
| 3.5 | Recommendation: one shader-visible heap of each type, set once per frame, with descriptors copied into it from non-shader-visible heaps as needed. If heaps must change, do it when GPU work is light, such as at the start of a command list or at a break where the GPU idles anyway. | same as 3.4; [Shader Visible Descriptor Heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shader-visible-descriptor-heaps) | documented |
| 3.6 | A driver may sub-allocate API heaps smaller than the hardware limit out of one hardware heap; if all of an application's heaps fit the hardware heap's capacity, switching between them causes no wait. Applications must still allow for a wait. | [Shader Visible Descriptor Heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shader-visible-descriptor-heaps); [Resource Binding spec](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#shader-visible-descriptor-heaps---used-during-rendering) | documented, specified |
| 3.7 | In a bundle the heaps can be set at most once and must match the calling command list. | [Descriptor Heaps Overview, Bundles](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview) | documented |
| 3.8 | With directly indexed heaps there is an ordering rule: `SetDescriptorHeaps` must come before the `SetGraphicsRootSignature` or `SetComputeRootSignature` call for a root signature carrying either directly-indexed flag, and may not be called with different heaps after that root signature is set and before a draw or dispatch. | [SM 6.6 Dynamic Resources spec, SetDescriptorHeaps and Set*RootSignature](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html#setdescriptorheaps-and-setrootsignature) | specified |
| 3.9 | Changing the root signature makes all previous root bindings stale; setting the same one again does not. *Reading:* after a heap switch the directly-indexed root signature has to be set again (3.8); because it is the same signature the non-table bindings should survive (this fact), while tables must be set again (3.3). The sources do not spell out this combination. | [Using a Root Signature](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-a-root-signature) | documented; combination unknown |

### 4. Directly indexed heaps (SM 6.6)

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 4.1 | `ResourceDescriptorHeap[i]` and `SamplerDescriptorHeap[i]` index the CBV/SRV/UAV heap and the sampler heap that `SetDescriptorHeaps` bound. There is no descriptor table or root parameter between the index and the heap. | [SM 6.6 Dynamic Resources spec](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html); [In the works: HLSL Shader Model 6.6](https://devblogs.microsoft.com/directx/in-the-works-hlsl-shader-model-6-6/) | specified, documented |
| 4.2 | *Reading:* since nothing maps the index, the application cannot offset or remap it at bind time. The slot a shader reads is the number the shader computes, counted from the start of the bound heap. Moving a descriptor to another slot means changing that number: in the shader, or in the data the shader takes the index from. | follows from 4.1 | reading |
| 4.3 | The root signature must carry `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` (0x400) and/or `D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED` (0x800). Without the matching flag, validation fails at shader compile or pipeline creation. The flags are not allowed on local root signatures. | [SM 6.6 Dynamic Resources spec, Root Signature Changes](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html#root-signature-changes); [D3D12_ROOT_SIGNATURE_FLAGS](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_root_signature_flags) | specified, documented |
| 4.4 | The feature must be supported on devices that have both Resource Binding Tier 3 and Shader Model 6.6. *Reading:* any host that runs these shaders is a Tier 3 host, so the Tier 3 rows of sections 1 and 6 are the ones that apply. | [SM 6.6 Dynamic Resources spec, Device Capability](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html#device-capability) | specified |
| 4.5 | Descriptors and data reached this way are treated as volatile, as if the range had `DESCRIPTORS_VOLATILE` and `DATA_VOLATILE`. Volatile still does not allow a descriptor to change during execution of a draw or dispatch. | [SM 6.6 Dynamic Resources spec, Descriptor and Data Volatility](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html#descriptor-and-data-volatility) | specified |
| 4.6 | An index that varies within a draw or dispatch must be wrapped in `NonUniformResourceIndex`, or the result may be undefined. | [SM 6.6 Dynamic Resources spec](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html); [Resource binding in HLSL](https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl) | specified, documented |
| 4.7 | The SM 6.6 spec says nothing about an index past the end of the heap, a null slot or an uninitialized slot. The general rules apply: out-of-bounds indexing of a descriptor table is a largely undefined memory access that can reset the device; heap memory starts undefined, and referencing uninitialized descriptors is undefined, including device reset. | [Resource Binding spec, Out of Bounds Indexing](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#out-of-bounds-indexing) and [Populating Descriptor Heaps with Descriptors](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#populating-descriptor-heaps-with-descriptors) | specified for tables; for heap indexing unknown |
| 4.8 | GPU-based validation reports uninitialized or incompatible descriptors used in a shader, and indexing beyond the end of the descriptor heap. | [GPU-based validation](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-d3d12-debug-layer-gpu-based-validation) | documented |

### 5. Copying descriptors and staging heaps

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 5.1 | `CopyDescriptors` and `CopyDescriptorsSimple` copy immediately on the CPU. They may be called from several threads as long as nothing makes conflicting writes. | [Copying Descriptors](https://learn.microsoft.com/en-us/windows/win32/direct3d12/copying-descriptors) | documented |
| 5.2 | The source must be a non-shader-visible heap, because a shader-visible heap may live in write-combined or GPU-local memory that is prohibitively slow to read. The spec calls reading a shader-visible heap from the CPU undefined behaviour. The destination may be any CPU-visible heap. | [CopyDescriptors](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-copydescriptors); [D3D12_DESCRIPTOR_HEAP_FLAGS](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_descriptor_heap_flags); [Resource Binding spec, Populating Descriptor Heaps with Descriptors](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#populating-descriptor-heaps-with-descriptors) | documented, specified |
| 5.3 | `CopyDescriptorsSimple` is preferred where it applies, for a better CPU cache miss rate from the linear copy. This is the only cost statement found; no per-descriptor or per-million figure exists. | [CopyDescriptorsSimple](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-copydescriptorssimple) | documented; cost at this scale unknown |
| 5.4 | Overlapping source and destination ranges are invalid. Source and destination need not line up range for range (sparse to contiguous is fine); heap types must match. | [Resource Binding spec, Copying descriptors on the CPU timeline](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#copying-descriptors-on-the-cpu-timeline); [Copying Descriptors](https://learn.microsoft.com/en-us/windows/win32/direct3d12/copying-descriptors) | specified, documented |
| 5.5 | Staging is optional: descriptors may be created straight into a shader-visible heap. Non-shader-visible heaps sit in cacheable system memory and are unconstrained in size. | [D3D12_DESCRIPTOR_HEAP_FLAGS](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_descriptor_heap_flags); [Non Shader Visible Descriptor Heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/non-shader-visible-descriptor-heaps) | documented |
| 5.6 | Heap contents may be changed before, during and after recording a command list that refers to them, but not while a submitted command list might reference that location. | [Descriptor Heaps Overview, Synchronization](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview) | documented |
| 5.7 | Root signature 1.1: a table range without `DESCRIPTORS_VOLATILE` holds static descriptors, which must be initialized when the table is set during recording and must not change until the list has finished executing for the last time. With the flag, they may change until the list is submitted. Breaking the promise is undefined. Static descriptors also lose defined out-of-bounds behaviour for buffer views. | [Root Signature Version 1.1](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-version-1-1) | documented |
| 5.8 | Microsoft's own description of per-draw tables: filling a fresh heap area per draw is the basic strategy; it repeats descriptors and uses heap space quickly. Named refinements are reusing definitions, overlapping tables between draws, pre-filling, and one large array addressed by indices. | [Descriptor Heaps Overview, Management](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps-overview) | documented |
| 5.9 | GPU handles must stay valid until every command list that references them has finished executing. | [Descriptors Overview](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptors-overview) | documented |
| 5.10 | Preview only: Resource Binding Tier 4 adds a command-list `CopyDescriptors` that runs on the GPU timeline between shader-visible heaps and needs descriptor heap barriers. The blog lists it under Agility SDK 1.722 preview, supported by AMD Radeon RX 7000 and 9000 drivers and planned by the other vendors. `D3D12_RESOURCE_BINDING_TIER_4` is not in `d3d12.idl` on the DirectX-Headers main branch today. | [Resource Binding spec](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#copying-descriptors-on-the-gpu-timeline); [Agility SDK 1.620 retail and 1.722 preview](https://devblogs.microsoft.com/directx/agilitysdk-1-620-retail-and-1-722-preview/) | specified, documented; header absence observed |

### 6. Tier differences for null descriptors and sparse heaps

| # | Fact | Source | Confidence |
|---|------|--------|------------|
| 6.1 | Tier 1: every descriptor covered by a bound table must be initialized (real or null) by the time the shader runs, even if not accessed. Tier 2: the same for CBV and UAV, not SRV. Tier 3: no such requirement, provided uninitialized descriptors are never accessed. | [Hardware Tiers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support); [Descriptors Overview, Null descriptors](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptors-overview); [Resource Binding spec](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#setting-descriptor-tables-in-the-root-arguments) | documented, specified |
| 6.2 | A null descriptor is made with a null resource and an otherwise valid view description. Reads then behave as unbound (SRVs return defaults) only if the shader's access type is compatible with the descriptor's type; a mismatch is undefined and can reset the device. Root descriptors should not be null. | [Descriptors Overview, Null descriptors](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptors-overview); [Resource Binding spec, NULL Descriptors](https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#null-descriptors) | documented, specified |
| 6.3 | Tier 3 lifts the per-stage table limits to the full heap for CBVs, SRVs and UAVs (Tier 1: 14 CBVs, 128 SRVs, 8 or 64 UAVs; Tier 2: 14 CBVs, 64 UAVs). | [Hardware Tiers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support) | documented |
| 6.4 | Nothing found describes a driver treating a sparse (mostly unwritten) heap differently from a full one in memory or cost: a heap of N descriptors is described as N times the descriptor size. | section 1.12 and 1.13 sources | unknown |

## What stayed unknown

- What each vendor's driver accepts or reports above 1,000,000 (1.9). Only the host's own
  answer (`MaxViewDescriptorHeapSize`, or a creation attempt) settles it.
- The size of the stall when heaps are switched, and which hardware has it (3.4).
- The cost of descriptor copies at hundreds of thousands per frame (5.3).
- Defined behaviour for a directly indexed slot that is past the heap, null, or uninitialized
  beyond the general rules (4.7).
- Whether non-table root bindings survive a heap switch followed by re-setting the same
  directly-indexed root signature (3.9).

## What this permits

For each candidate strategy: whether the rules above allow it, and which constraint bites.
No strategy is recommended here.

### Segregating single-slot and table dynamic descriptors

Allowed. The rules treat a heap as an array of slots and say nothing about how an application
partitions it. The difference the rules do make is that a descriptor table is a contiguous
region addressed from a base handle, so table copies need runs of consecutive slots while
single descriptors do not.

Constraints: both regions and the mirrored range share one size limit (1.1, 1.4). A slot may
not be rewritten while a submitted list may still reference it (5.6), and not from the moment
its table is set if the host root signature leaves the range static (5.7).

### Shrinking the mirrored range to keep the dynamic range

Allowed as far as heap creation goes: any size up to the limit is valid, and on Tier 3 slots
that are never accessed need not be initialized (6.1).

Constraint: direct indexing (4.1, 4.2). A title index past the mirrored range is not remapped
by anything outside the shader; it reads whatever the host put at that slot (a dynamic table
entry of possibly incompatible type, which is undefined per 6.2) or runs past the heap (4.7).
So it is only sound for indices the title provably does not use, or when the index itself is
rewritten in the translated shader or its inputs.

### Recycling dynamic slots in a ring

Allowed, bounded by lifetime. A slot can be reused once every submitted command list that
may reference it has finished executing (5.6, 5.9).

Constraints: descriptor writes happen immediately on the CPU, while shaders read slots at
execution. Two draws recorded into the same list therefore cannot use one slot for different
contents: the later write is what both would see, which is a changed descriptor for the first
draw (invalid for static ranges per 5.7, and simply wrong for volatile ones). The ring must be
at least as large as the distinct dynamic slots of all lists recorded but not yet finished.
The GPU-timeline copy that would lift this (5.10) is preview-only and Tier 4.

### A second heap switched mid-list

Allowed: switching inside a command list is explicitly acceptable (3.1).

Constraints:
- Only one view heap is bound at a time (3.2), so everything one draw reads, both its tables
  and every slot its shaders index directly, has to be in the heap bound for that draw.
- Table state is undefined after the switch and must be set again (3.3).
- With directly indexed root signatures the heaps must be set before the root signature, and
  cannot change between setting it and the draw (3.8), so the root signature is set again
  after each switch (3.9).
- A switch may stall the GPU on some hardware, with no documented size (3.4). The case
  documented as free of waits needs all heaps to fit one hardware heap (3.6), which two heaps
  of around a million views each are unlikely to do (*reading*).
- Two large shader-visible heaps both count against write-combined memory (1.13).

### Compacting and deduplicating per-draw tables by content

Allowed. A table's base handle may point anywhere in the bound heap, several root parameters
or draws may point at the same or overlapping ranges, and Microsoft's own guidance names
reuse and overlapping tables as refinements of the per-draw strategy (5.8). On Tier 3 only
slots that shaders actually access need valid descriptors (6.1), so a table may be shortened
to what is read, where that can be known.

Constraints:
- A shader-visible heap cannot be read back (5.2), so equality has to be decided from
  CPU-side records of what each slot was made from, not from heap contents.
- A table is contiguous, so sharing works for whole tables or aligned overlapping runs, not
  for arbitrary subsets.
- A shared table lives as long as the longest-lived list referencing it (5.6, 5.9), and its
  contents may not be changed for one user while another is recorded or in flight (5.7).
- Shortening a table is only sound if no shader indexes past the kept entries (4.7); the
  current code copies the whole range for that reason.
