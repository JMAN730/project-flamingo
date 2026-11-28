# D3D12 debug layer crash in DXGIDebug.dll on Windows 10

Research notes for fork issue 817 (2026-11-28). The question: why a run with
`VICTORY_D3D12_DEBUG=1` crashes inside `DXGIDebug.dll` while the debug layer
reports a message, and what setup avoids it.

## What we observed

- Machine: NVIDIA GTX 1080, Windows 10 build 19045. Agility SDK
  `D3D12Core.dll` and `d3d12SDKLayers.dll` in a `D3D12` folder beside the
  executable; VIctory exports `D3D12SDKVersion = 620` and
  `D3D12SDKPath = ".\D3D12\"` (`crates/victory-shader/src/agility.rs`).
- One run: access violation at `DXGIDebug.dll+0xd33c` reading address `0x30`.
  Stack: `DXGIDebug.dll` -> `D3D12SDKLayers.dll` -> `DXGIDebug.dll` (x2) ->
  `D3D12SDKLayers.dll` (x2) -> `victory_gpu::resource::host_texture`
  (inside or right after `CreateCommittedResource`), on the worker thread that
  mirrors descriptors.
- Earlier in the same run the layer logged warnings normally (CheckFeatureSupport
  superseded, CreateCommittedResource ignoring InitialState for buffers), so the
  message path works at least some of the time.
- File versions on the reference machine (read with `Get-Item ... VersionInfo`
  on 2026-11-28):

  | File | Location | Version |
  | --- | --- | --- |
  | `dxgidebug.dll` | `System32` | 10.0.19041.1 (dated 2019-12-06) |
  | `d3d12SDKLayers.dll` | `System32` | 10.0.19041.5794 |
  | `D3D12Core.dll` | `System32` | 10.0.19041.5794 |
  | `D3D12Core.dll` | `target\diag\D3D12` | 1.620.2.0 (2026-11-06 build) |
  | `d3d12SDKLayers.dll` | `target\diag\D3D12` | 1.620.2.0 (2026-11-06 build) |

  The system `dxgidebug.dll` is the original 19041 release-to-manufacturing
  build; it has not been serviced. The SDK layers calling into it are seven
  years newer.

## What our code does

- `crates/victory-gpu/src/host.rs` `create()`: in a diagnostics build with
  `VICTORY_D3D12_DEBUG`, calls `D3D12GetDebugInterface` +
  `EnableDebugLayer` (plus GPU-based validation with `VICTORY_D3D12_GBV`),
  then casts the device to `ID3D12InfoQueue`. No storage filters, no break-on
  settings, no message count limit, no `SetMuteDebugOutput`.
- `Host::log_messages()` (`host.rs`, "Moves the debug layer's messages into
  our log") polls: `GetNumStoredMessages`, `GetMessage(i)` twice per message
  (size, then data), logs it as `D3D12: ...`, then `ClearStoredMessages`. It
  is called from several places (`host.rs`, `host_relay.rs`), so it can run on
  a different thread from the one creating resources.
- We do **not** use `ID3D12InfoQueue1::RegisterMessageCallback`. So the crash
  is not in a callback of ours; the stack shows the layer producing a message
  on our resource-creation thread, not our reader.

## Findings from sources

1. **The D3D12 SDK layers' message plumbing lives partly in `DXGIDebug.dll`,
   and that DLL is not part of the Agility SDK.** The DXGI info queue
   (`IDXGIInfoQueue`) is implemented in `DXGIDebug.dll`
   ([Learn, IDXGIInfoQueue](https://learn.microsoft.com/en-us/windows/win32/api/dxgidebug/nn-dxgidebug-idxgiinfoqueue),
   "req.dll: DXGIDebug.dll"). A Microsoft D3D12 developer, quoted in
   [gfx-rs/wgpu#1659](https://github.com/gfx-rs/wgpu/issues/1659) (2021),
   said of the D3D12 message callback that it "is tied to DXGIDebug.dll, which
   is not one of the redistributable pieces from the Agility SDK", so it needed
   a newer Windows build. The Agility SDK NuGet package ships the runtime
   binaries, `d3dconfig.exe` and `D3D12StateObjectCompiler.exe`; no DXGI debug
   DLL is listed
   ([NuGet 1.620.2](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2)).
   The callback itself first shipped in Windows 10 build 20236, an Insider
   build newer than 19045
   ([DirectX blog: debug layer message callback](https://devblogs.microsoft.com/directx/d3d12-debug-layer-message-callback/)).
   So an Agility SDK 1.620 `d3d12SDKLayers.dll` on Windows 10 19045 always
   talks to the old system `dxgidebug.dll`; Microsoft ships no way to pair
   them.

2. **The only version match Microsoft documents is SDK layers <-> D3D12Core.**
   "A mismatch between D3D12SDKLayers (the debug layer) and D3D12Core.dll will
   cause CreateDevice to fail"; keep the SDK files in a subfolder, not beside
   the exe
   ([Getting Started with the Agility SDK](https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/)).
   We already do this (device creation succeeds and messages flow). The same
   page says nothing about `DXGIDebug.dll` or Graphics Tools; no Microsoft
   page found states a minimum `dxgidebug.dll` for a given SDK layers version.
   Consequence: we cannot fall back to the system `d3d12SDKLayers.dll`
   (19041.5794) while loading Agility `D3D12Core.dll` 1.620; the pair must come
   from the same SDK.

3. **Graphics Tools provides the system debug layer pieces.** On Windows 10
   the "Graphics Tools" optional feature is what makes a debug device possible
   ([Learn, DirectX runtime and graphics diagnostics](https://learn.microsoft.com/en-us/windows/uwp/gaming/use-the-directx-runtime-and-visual-studio-graphics-diagnostic-features);
   [DirectX blog, improved SDK layers](https://devblogs.microsoft.com/directx/gpu-plugins-improved-sdk-layers-and-hang-debugging-bringing-directx-12-tools-to-the-next-level/)).
   It installs with
   `Add-WindowsCapability -Online -Name "Tools.Graphics.DirectX~~~~0.0.1.0"`
   (elevated). No source found says whether reinstalling it on 19045 brings a
   newer `dxgidebug.dll` than 19041.1.

4. **Debug layer crashes and races are a known class, fixed per release.**
   The NuGet changelog lists, among others: 1.619.5 "a race condition" in
   subresource state validation, 1.606.3 "various debug layer stability
   fixes", 1.602.0 a GBV crash with `TRACKING_ONLY` patch mode, and 1.611.1 a
   change to the message-callback registration signature
   ([NuGet changelog](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2)).
   The 1.600.10 post warns that churn makes new bugs "more likely" to "show up
   as bugs in debug layer validation"
   ([Agility SDK 1.600.10](https://devblogs.microsoft.com/directx/agility-sdk-1-600-10-minor-bugfix-release/)).
   An older precedent with the same shape (null read inside
   `d3d12SDKLayers.dll` on NVIDIA) was "a bug in d3d12sdklayers.dll exposed by
   recent NVidia drivers", fixed through Windows Update
   ([DirectX-Graphics-Samples#95](https://github.com/microsoft/DirectX-Graphics-Samples/issues/95)).

5. **Threading.** The message callback
   post says a callback "may execute on a thread other than the one that
   originally issued the API call". The `ID3D12InfoQueue` reference says
   nothing about concurrent `GetMessage`/`ClearStoredMessages` while another
   thread produces messages
   ([Learn, ID3D12InfoQueue](https://learn.microsoft.com/en-us/windows/win32/api/d3d12sdklayers/nn-d3d12sdklayers-id3d12infoqueue)).
   No source found ties concurrent polling to a crash.

## Answer

No Microsoft source names this exact crash. The best-supported cause is the
pairing in finding 1: the Agility SDK 1.620 `d3d12SDKLayers.dll` reports
messages through the system `DXGIDebug.dll` (the faulting stack shows it), which on this machine is the
unserviced 19041.1 build, and Microsoft neither redistributes a newer one nor
documents compatibility between the two. A null-based read (`0x30`) inside
`DXGIDebug.dll` called back and forth with the SDK layers fits an interface or
object the old DLL lacks, but that is inference, not documented. The second
candidate is an ordinary SDK-layers bug under multithreaded resource creation
(finding 4), which also has precedent. Our polling reader is not on the
faulting stack.

## Setup that should avoid it (in order to try)

1. **Run debug-layer sessions on Windows 11** (current `dxgidebug.dll`), or
   compare the crash rate there. This is the only setup in which the SDK
   layers and `DXGIDebug.dll` come from the same era.
2. **Reinstall/update Graphics Tools** on the Windows 10 machine (elevated:
   `Get-WindowsCapability -Online -Name Tools.Graphics.DirectX*`, then remove
   and re-add) and re-check the `dxgidebug.dll` version. If it stays at
   19041.1, nothing on Windows 10 fixes the pairing.
3. **Keep `D3D12Core.dll` and `d3d12SDKLayers.dll` from the same Agility SDK
   package** in `D3D12\` (already the case). Mixing in the System32
   SDK layers fails device creation (finding 2).
4. **Produce fewer messages.** The faulting stack shows a message trip into
   `DXGIDebug.dll`. Push an `ID3D12InfoQueue` storage filter denying the
   known, accepted warnings (CheckFeatureSupport superseded,
   InitialState ignored for buffers) at device creation, so the remaining
   traffic is the errors we want. Unsourced mitigation; it lowers exposure,
   it does not remove the cause.
5. **Try without GPU-based validation and on a newer Agility SDK** if the
   crash reproduces with filters in place (finding 4: per-release
   debug-layer fixes).
6. If none of this holds, the debug layer stays usable for short, targeted
   runs (one failing command list), and DRED (`dred_on()` in `host.rs`) covers
   the device-removed case without the SDK layers.

## Not found

- Any Microsoft statement of the minimum `dxgidebug.dll` for an SDK layers
  version, or that SDK layers can be used without Graphics Tools.
- Any Microsoft known-issue for a `DXGIDebug.dll` access violation.
- Whether `ID3D12InfoQueue` is safe to drain from one thread while another
  thread creates resources.
