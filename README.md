# VIctory

*Read this in: [English](README.md) | [Português](README.pt-BR.md) | [Bahasa Indonesia](languages/README.id.md)*

Runs an Xbox Series X|S title natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can and on first use where it must (shaders). See
[docs/architecture.md](docs/architecture.md).

The Windows executable can also be launched under Proton on x86-64 Linux. See
[Running with the Proton launcher](docs/proton-launcher.md).
For a pinned Linux development environment, run `nix develop` (or `nix-shell`).
See [Linux development with Nix](docs/linux-proton.md#development-with-nix).

## Status (2026-11-23)

**Not playable yet.** The title boots, runs its menus at its own 30 fps and
loads the world, where rendering fails.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | reached and runs with indirect draws; its title card renders, the scene after it is still dark; 0.5-1.8 s per frame |

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** Entering the world on a real GPU can currently hang it, and
> Windows then resets the display driver (every monitor goes black for a
> moment). To look further, use a diagnostics build with
> `VICTORY_D3D12_WARP=1`, which renders on the software adapter.
> A world frame can also outlast Windows' 2 s GPU timeout on slower GPUs
> (`slow host submission` above 2000 ms, then the device is removed as hung);
> raising `TdrDelay` and `TdrDdiDelay` (for example to 60) under
> `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` and rebooting
> avoids that reset.

For an Arch Linux cross-build and a Proton Experimental run on an RTX 4080,
see the [Linux rendering report](docs/arch-proton-rendering.md).

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads, keyboard, keyboard stand-in pad | Trigger rumble |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode; convolution reverb is wet | DSP FFT/IFFT transforms stay silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | Mesh pipelines, shader-written buffers, descriptor lifetimes, indirect execution (arguments clamped), pipelines made by worker threads and kept on disk | Most of the scene is still missing (untranslated shaders drop draws); frames take seconds; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL at first use, cached on disk. Menus: all. World: mesh 8,191 of 8,506, vertex/pixel 10,617 of 12,346, compute 1,643 of 2,539 | Merged geometry stages, the ray-tracing builder's shaders; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Runs through the menus to the loading screen | The world, as above |

In an offline scan of `shaders.rpf`, SM 6.6 texture atomics recovered
520 shader translations. Specializing proven branch outcomes recovered
another 108 by keeping descriptor words off paths that consume buffer data.
Microsoft's DXIL validator accepted all 628 shaders. Game rendering with
these shader changes still needs a Windows D3D12 run.

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (the world uses mesh shaders where the GPU has them, else a
compute and vertex fallback), Rust (pinned in
`rust-toolchain.toml`). On Linux, see
[docs/linux-proton.md](docs/linux-proton.md) for which crates build and test
natively today, cross-building `victory.exe` and running it under Proton.
For Windows 10 runtime setup, adapter selection, and GPU troubleshooting, see
[docs/windows-gpu-support.md](docs/windows-gpu-support.md).

Windows 10's own Direct3D 12 stops at shader model 6.5, so there
`victory.exe` needs the runtime of Microsoft's Agility SDK. Take
`D3D12Core.dll` from the NuGet package `Microsoft.Direct3D.D3D12` 1.620.2
(`build\native\bin\x64`) and put it in a `D3D12` folder beside the
executable before building, for example `target\release\D3D12\D3D12Core.dll`:
a build that finds it there makes `victory.exe` load it. Keep the folder
beside `victory.exe` wherever you copy it. A build made without the DLL uses
the system's runtime, which is enough on Windows 11.

Under Wine or Proton, builds without embedded helper shaders also need DXC's
`dxcompiler.dll` (with its `dxil.dll`): Wine's `d3dcompiler_47` cannot compile
the indirect-argument clamp. The Nix shell sets `VICTORY_BUILD_DXC` to embed
these helpers during cross-builds. Runtime DXC is looked for in
`VICTORY_DXC` (the DLL or its folder), beside the
`dxil.dll` that shader validation finds, then the executable's folder and System32.

To install Rust, use rustup (`winget install --id Rustlang.Rustup -e`, or
the installer from https://rustup.rs). The first `cargo` command in this
repository then fetches the pinned toolchain. Building needs the MSVC C++
build tools: Visual Studio or its Build Tools with the "Desktop development
with C++" workload.

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

On a slow host (a few frames a second or less), set
`VICTORY_TIME_SCALE=auto` (or `auto:<fps>`, 10 by default): the title's
clock then slows while frames come slower than that, so what the title does
by the clock (streaming in and creating objects) keeps pace with what it
does per frame (freeing them). Without it the title can run out of a fixed
pool of objects while loading and stop at a breakpoint. It is off by
default.

Keep the game folder on an SSD if you can. From a hard drive the first
screen stays black, at 30 fps and about 3 draws/frame, for a minute and a
half or more while the title reads its files: the drive stays 100% busy with
small random reads, and nothing shows until they finish. Wait for it rather
than closing the window. A later launch is faster while Windows still holds
the files in its cache.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo test --workspace -- --test-threads=1 # serializes D3D12 tests; no game data
cargo run -p victory -- config           # check your paths
```

Run the title, replacing the example path with the full path to your game's executable:

```powershell
cargo build --release -p victory
.\target\release\victory.exe run 'D:\Xbox Games\Title\title.exe' --execute
```

### Selecting the Series S profile

The title chooses its render configuration from the console model, not from the
display modes VIctory advertises through DXGI. To use the lower-resolution
profile, pass `--console series-s` when starting the title:

```powershell
target\release\victory.exe run <game dir>\<the title's exe> --execute --console series-s
```

In a reported diagnostic run, the Series X profile used a 3840×2160 scene
with 2560×1440 internal targets; the Series S profile used a 1920×1080 scene
with 1280×720 internal targets. The UI can still use a separate 3840×2160
presentation plane, so selecting Series S does not necessarily make the
window or every render target 1080p. Changing the DXGI mode list alone does
not change the scene resolution ([issue #203](https://github.com/Marvisak/VIctory/issues/203));
Series S multi-plane presentation is covered separately by
[PR #211](https://github.com/Marvisak/VIctory/pull/211).
The experimental `VICTORY_PRESENT_TWO_PLANES=1` setting enables composition
for the two full-screen RGBA8 planes used by the Series S menu. It uses the
base plane when the overlay is unavailable or the compositor cannot start.
The diagnostic render-target override takes precedence.

The same report measured about 3.9 GB less committed process memory in the
menus/intro and about 3.6 GB less at the world peak with Series S enabled.
However, peak committed memory was still about 17.7 GB, and whole-system
GPU memory use still reached about 7.3 of 8 GB. Both profiles also lost the
GPU device 30–45 seconds after entering the world. These are measurements
from one Windows 10 / RTX 2070 SUPER 8 GB / 32 GB RAM diagnostic setup
(base commit `d2a4bae`, with PRs #175 and #211 applied), not general
requirements or a guarantee of stability. Series S can reduce memory
pressure, but it does not make a 16 GB RAM / 8 GB VRAM system reliably fit
the world or resolve the current GPU hang.
The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Input: XInput pads. A DualSense or DualSense Edge (USB or Bluetooth) is read
over HID as well and plays as the first pad when no XInput pad answers. With
no pad connected the keyboard is a pad while the title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, O | A, B |
| Backspace, X, Y | B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--validate] [--pipelines] [--mesh-fallback]
                  [--barycentric-fallback] [--list] [--jobs N] [--hash H,...]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

`translate --jobs N` requires a positive worker count. Omit it to use the
default worker count.
`translate --hash H` translates only the containers whose runtime hash (the
16 hex digits the runtime logs a shader by) is one of those given.
With `VICTORY_WAVE64_SPLIT=1`, `translate` lowers Wave64 programs as two
32-lane halves (as the runtime does on a 32-lane GPU); this needs no
device, so it works with or without `--pipelines`.

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

For structured failure reports, see [Dependency diagnostics](docs/dependency-diagnostics.md).

## Debugging

If the title presents one or two black frames and exits with `0x80000003`,
check the archive files before investigating rendering. The loader's import
report only counts bindings, so it can report zero missing imports with a
damaged archive. Check the SHA-256 of the files your run actually uses against
a known-good copy of the **same build**. For example, in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'C:\path\to\title\anim.rpf'
```

If a hash differs, restore that file from your verified backup and rerun with
the same build of VIctory and the same options. A breakpoint can have other
causes. Include the archive verification outcome in your test report.

Set `VICTORY_WAVE64_DOUBLE=1` to let a 32-lane host run a straight-line
64-thread compute group that reads indexed lanes across the two halves.
The initial form handles integer register operations, dword LDS reads and
writes, and raw UAV stores. It uses no extra LDS. Other programs retain
their usual translation. The option has separate shader cache entries.

The experimental `VICTORY_WAVE64_CONSOLE_FLOW=1` option lowers eligible
single-wave compute groups for a host that supports 32-lane waves. It
handles some groups that share local data or exchange values across halves.
The cross-half console-flow retry is limited to groups of at most 64 threads.
Larger groups can use the split-wave path when their operations stay within
each half. Unsupported cases retry the ordinary lowering. The option is off
by default and keys the shader cache separately.

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

The on/off switches below (`...=1`) are off when unset or set to `0`.

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds, with the most stack it has ever committed and its reserve |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_XMEM_REUSE_PLACEHOLDER_BOUNDS=1` | experimental reuse of exact placeholder bounds within an XMem mapping call |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_KEYS_FILE=<path>` | presses the keys named one per line in that file (same names), read while the title runs: each held 0.2 s, 0.2 s apart; append lines as screens appear (the legal-notices screen takes `Down`, `Down`, `Enter` a few seconds apart) |
| `VICTORY_CRASH_BREAKPOINTS=1` | appends first-chance breakpoints to `crashes.log`, including ones a handler may recover from |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_COPY_BACK_GPU_WINS=1` | copy-back with a submission takes every byte the GPU changed, also where the title wrote the same dword since (the rule before the three-way merge) |
| `VICTORY_TEXTURE_CPU_REFRESH=1` | (any build) a texture the host GPU copied or resolved into is uploaded from title memory again when a sampled fingerprint of that memory changed since (the CPU wrote it after the GPU); `=0` only logs it. Each texture logs once at debug: `texture written by the GPU, then by the CPU: re-uploaded` or `…: kept GPU copy` |
| `VICTORY_TEXTURE_RT_UPLOAD=1` | (any build) render-target and UAV-capable textures are uploaded from title memory when read, as other textures are, until the host GPU first renders to, clears, copies into or writes them (then the GPU's contents stay); depth textures are not uploaded. Each texture logs once at debug: `render-target or UAV texture read before any GPU write: uploaded from title memory` (or `…: a depth texture, not uploaded`) |
| `VICTORY_COPY_BACK_LATE_RULE=1` | the three-way merge also on the copy-back that runs after a submission (relay, card copies, lazy readback), where it is off: the title's world stops on it |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_CAPTURE_SECONDS=<from>-<to>` | with `VICTORY_CAPTURE`, also saves every frame presented in that span of seconds after the first presented frame |
| `VICTORY_PLANE_STATS=1` | with `VICTORY_PRESENT_TWO_PLANES=1`, logs RGB statistics for successful two-plane presents listed in `VICTORY_CAPTURE_FRAMES` |
| `VICTORY_DUMP_TARGETS=1` | with `VICTORY_CAPTURE` and `VICTORY_CAPTURE_FRAMES`, saves every render target of those frames |
| `VICTORY_SHOW_TARGET=2560x1440-f28` | shows the last render target of that size and DXGI format each frame instead of the title's frame (copied 1:1 to the top left, only when its format matches the window's), e.g. the world's geometry while its lit image is black |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_GBV_EAGER=1` | with the two above, compile the active GPU validation shader variant at pipeline creation rather than first bind |
| `VICTORY_D3D12_GBV_EAGER=guarded` | as above, explicitly selecting guarded GPU validation (fails host initialization if guarded mode cannot be verified; diagnostics only) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_BREADCRUMBS=1` | without the above: on a GPU failure, the commands of the last submissions that had begun but not finished (two immediate writes per command; light enough not to hide timing-sensitive faults) |
| `VICTORY_CHECK_DESCRIPTORS=1` | at each draw and dispatch, the descriptor kind at every heap entry a translated shader reads (table entries, and heap indices in root constants or read from a buffer) compared with what the shader declares; mismatches logged, and draws with a buffer read as a texture, another texture dimension or another view class dropped; slots a submission in flight reads rewritten with another kind logged |
| `VICTORY_TIME_SCALE=0.25` | the title's clock (`QueryPerformanceCounter`, `GetTickCount`) at a quarter of real time |
| `VICTORY_TRACE_CBV=<hash>[,...]` | for the stages with those runtime hashes (as `translate --hash` takes them), logs each constant view they read: a fresh table's entry (title address, size, scratch address, host slot), one made at a title slot on the buffer's host copy, a fill reused from earlier in the list, and after the list ran the first 64 bytes of each fill next to title memory's |
| `VICTORY_SHADER_TRACE_DYNAMIC=1` | logs how each translated shader computes the heap indices it adds at run time (`dynamic heap indices`) |
| `VICTORY_LOG=victory_gpu::draw=debug` | with frames listed for capture: one line per draw and dispatch of those frames with its targets (named as in the `VICTORY_DUMP_TARGETS` files) and, if it was dropped, why |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `VICTORY_SKIP_HOST_PSOS=<name>,...` | omit a host pipeline whose 32-digit library name matches exactly; get names and worker thread IDs with `VICTORY_TRACE_PIPELINES=1` |
| `VICTORY_SKIP_WAVE64=mesh,compute` | omit draws and dispatches whose pipeline has a program of a listed stage written for 64-lane waves (on hosts with 32-lane waves, such as NVIDIA's, these can hang the GPU) |
| `VICTORY_NO_BARYCENTRICS=1` | pixel shaders that need barycentrics run the way they do on a host without them (translated without `SV_Barycentrics` and `GetAttributeAtVertex`, behind a geometry shader that gives them the values at each vertex and the weights), to compare the two on a host that has them |
| `VICTORY_SLOT67_HR=<name>` | what the device's `DeserializeComputePipelineStateX` (slot 67, used by the AMD upscaler with GFX10 compute programs we do not translate) returns: `E_FAIL` (the default, also without this build), `E_INVALIDARG`, `E_NOTIMPL` or `S_OK` (a pipeline whose dispatches are dropped); the title's reaction to each shows whether it has a fallback |
| `VICTORY_FSPX_REJECT_SLOT62=1` | fail `CreateComputePipelineStateX` (slot 62) with `E_FAIL` for FSPX compute modules instead of making a pipeline whose dispatches are dropped |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `VICTORY_WAIT_TIMEOUT_MS=<ms>` | override the 2000 ms fail-open timeout for mid-list GPU memory waits |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_ADAPTER=<name>` | select a D3D12 adapter by case-insensitive substring (default: the first GPU, in high-performance order, with shader model 6.6, resource binding tier 3 and mesh shaders; the log says what is missing when none has all three) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_SHADER_OPT=<bits>` | which clean-up passes run over each translated shader (own shader cache entries): `0` none, 1 dead code, 2 constants and identities, 4 reuse of values, 8 branch facts, 16 loads within a block; unset runs all. Every pass is exact, so a fault that goes away with `0` is in a pass |
| `VICTORY_SHADER_NO_LAST_CHANCE=1` | works in release builds too (own shader cache entries): the translator's last-chance retry (leftover registers, in `emit` and `resource::track`) is off; programs that need it are refused |
| `VICTORY_SHADER_NO_SECOND_PASS=1` | works in release builds too (own shader cache entries): no second try at refused programs (SCC of a lane mask as wave-any, lane reads from spills) |
| `VICTORY_SHADER_NO_BUFFER_TABLE=1` | works in release builds too (own shader cache entries): buffer loads whose element stride only the view knows (a descriptor picked at run time, such as each lane's own in a waterfall loop) are refused instead of read through the host's buffer view table |
| `VICTORY_NO_ACCC=1` | works in release builds too: `ACCC` containers have no programs |
| `VICTORY_ALIAS_COPY=0` | views the title places in the memory of a texture the host writes are filled from title memory again instead of copied from that texture on the host |
| `VICTORY_REBAR=0` | disable `D3D12_HEAP_TYPE_GPU_UPLOAD` (Resizable BAR) placement for scratch and shadow buffers, keeping them in system memory |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |
| `VICTORY_NO_WATCHDOG=1` | apply the opt-in title watchdog disarm hooks (Series 1.0.2.1) |
| `VICTORY_SKIP_CPU_CHECK=1` | skip the AVX2 pre-flight check before `--execute` (for CPU emulators such as Intel SDE) |
| `VICTORY_PAD_LOG=1` | log DualSense / HID pad discovery and reports |
| `VICTORY_COPY_MODE=plain` | copy title GPU memory without the timed/stats path used for the frame-rate line |
| `VICTORY_WARP=1` | force the shader-test host device onto WARP (distinct from `VICTORY_D3D12_WARP`) |
| `VICTORY_SAVE_STOP=<n>` | test helper: hang the process at save step `n` (used by crash-survival tests) |

For Series S two-plane diagnostics, set `VICTORY_PRESENT_TWO_PLANES=1`,
`VICTORY_PLANE_STATS=1` and an explicit comma-separated `VICTORY_CAPTURE_FRAMES`
list. `VICTORY_CAPTURE` is optional. Each selected two-plane present that
succeeds logs a `PresentX plane RGB samples` record for `primary`, `overlay`
and `composite`. Each record counts 16 positions in a 4×4 grid across that
texture's own extent. `nonzero` counts positions with any nonzero RGB,
`above_black` counts those with any RGB channel at least 8, and `peak` gives
the highest sampled RGB channel. The probes ignore alpha and can miss detail
between sample positions. A failed readback logs `PresentX plane RGB sample
unavailable`. Single-plane fallbacks and failed presents produce no plane
records. The fade detector uses the presented image's samples.

Every build appends unhandled exceptions, panics, and host GPU failures to
`crashes.log` in the title's data folder.

For a startup breakpoint (`0x80000003`), run the diagnostics build with
`VICTORY_CRASH_BREAKPOINTS=1`. Keep the console output along with the new
`crashes.log` entries. A host DLL refusal includes the caller stack in the
console output. The refusal is expected for DLLs outside the host policy,
including `rpcrt4.dll`.

On Series 1.0.2.1, a missing `appdata0.rpf` reproduced the breakpoint at
`+0x2237178` before the first window. Restore the archive from the same
package beside the title executable. `run --execute` checks that it can
read the RPF9 header before starting this build.

To compare hardware and software rendering, repeat with
`VICTORY_D3D12_WARP=1` and check the console for
`host device on the software adapter (WARP)` to confirm the WARP device was
created. Compare the exception's module and offset, stack frames, and
preceding messages across the two runs.

### World runs: `cargo xtask run-world`

World runs share one GPU between sessions, and a run is only worth comparing
with another when both were started the same way. One command starts them:

```powershell
cargo xtask run-world --session <name>                       # build, claim the card, run, write the manifest
cargo xtask run-world --session <name> --capture-frames 1500-1560:20 --capture-seconds 300-330
cargo xtask run-world --session <name> --keys 40:Enter,60:Enter,80:Enter --max-seconds 1800
cargo xtask run-world status                                  # who holds the card
cargo xtask run-world stop                                    # end the run that holds it
cargo xtask run-world list                                    # runs so far: commit, build, length, exit code
```

A run:

1. builds `victory.exe` (`--build diag`, the default, or `release`; `--no-build`
   uses the one already in the target folder, `--exe <path>` another one) and
   checks that the Agility SDK's `D3D12` folder sits beside it;
2. claims the card by creating `C:\victory-data\card.lock` (`--lock`), which
   holds the session name, the run id, the start time and the process ids of
   the launcher and the title. While those processes run, another
   `run-world` refuses to start and says who holds the card. A lock whose
   processes are gone is stale and is cleared. Reading, clearing, writing and
   releasing the lock happen while a second file, `card.lock.guard`, exists for a
   moment, so two runs that start together cannot both get the card;
3. copies the exe as `victory-<run id>.exe`, with its `D3D12` folder, to a
   folder of its own under `victory-data\runs` in your user folder
   (`--bin-dir`), and starts the title from `game_dir` of `victory.toml`
   (`--config <file>` names another config; `--title-exe <name>` when the
   folder has several executables);
4. reads the adapter line of the title's output. If the adapter lacks shader
   model 6.6, the title got the system's D3D12 instead of the Agility SDK:
   the run is ended at once and the command fails;
5. waits for the title, then writes how it ended and gives the card back. A
   run is ended after 45 minutes unless `--max-seconds` says otherwise
   (`0`: no limit), so a forgotten run does not hold the card.

The exe copy goes to the user folder because of where Windows looks for the
Agility SDK: run from a folder other users may write to (`C:\victory-data`
was one), the exe did not load the `D3D12` folder beside it and ran on the
system's D3D12, without shader model 6.6, and the title crashed a minute in.

Everything a run writes goes to `C:\victory-data\runs\<run id>\` (`--runs-dir`),
which must be outside the repository like every other output folder:
`victory.log` (the title's output), `frames\` (captured frames),
`crashes.log` (what the run appended to the title's `crashes.log`, if
anything) and `manifest.json`: the commit and whether the tree had changes,
the build, the arguments, every variable that was set, when the run was
asked for, when the title was started and when it ended (the run's length
counts from the title's start, not from the build before it), the exit code,
what ended the run and where the crash report is.

The defaults are the environment world runs are known to work with on an 8 GB
card: `--console series-s`, `VICTORY_SKIP_PIPELINES=5d113aa6830760b6,caa78b8ca62bc3d6,5552dac6a2b7a1fd`,
`VICTORY_TEXTURE_MIP_DROP=2`, `VICTORY_TEXTURE_MIP_DROP_MIN_SIZE=256`,
`VICTORY_WAVE64_SPLIT=1` and `VICTORY_TIME_SCALE=auto`. The skip list, capture
and scripted keys only exist in the diagnostics build, which is why `diag` is
the default; `--build release` refuses to start while any of them is set.
`--env NAME=VALUE` changes or adds a variable and `--env NAME=` removes a
default. Other `VICTORY_*` variables of the calling shell are **not** passed
on (only the path variables are), so the manifest is the whole story. A
profile file (`--profile <file>`) holds the same settings; flags win over it:

```toml
build = "diag"
console = "series-s"
keys = "40:Enter,60:Enter,80:Enter"   # as VICTORY_KEYS takes them
max_seconds = 1800

[env]
VICTORY_LOG = "victory_gpu::draw=debug,info"
VICTORY_SKIP_PIPELINES = ""            # empty: remove the default

[capture]
frames = "1500-1560:20"                # presented frames, every 20th
seconds = "300-330"                    # seconds after the first presented frame
```

Frames whose number is a power of two are always saved. Presented-frame
numbers do not land on the same scene from run to run (the intro's length
varies), so capture a range of frames or a span of seconds and pick by eye.

`run-world stop` ends the title by the process id in the lock, after checking
that the id still belongs to that run's exe. Never end a run by image name:
other sessions run `victory.exe` tools of their own. Windows can take a
minute to tear down an ended title; `stop` waits for it. `--dry-run` prints
what would run without claiming the card.

Anything else that uses the card (GPU tests, a script of your own) shares it
by writing the same lock file while it runs and deleting it afterwards: JSON
with at least `pid`, and `session` and `run_id` to say whose it is, or one
line `<pid> <session>`. `run-world` treats such a lock as live while that
process runs. With `--exe`, the commit in the manifest is that of the tree
the command ran in, not necessarily the one the exe was built from.

### Reporting a GPU failure

A host device removal, a hang, or a frame that comes out wrong is much easier
to place with these next to it:

* the adapter and the driver, as the title's own window sees them (GPU model,
  the Adrenalin or GeForce version, and the Windows build number - a driver
  update between two reports changes what is worth comparing);
* whether `VICTORY_D3D12_WARP=1` reproduces it: a failure that survives the
  software adapter is ours, one that does not points at the driver;
* the last lines of `crashes.log`, which name the faulting code and the
  commands in flight when `VICTORY_D3D12_DRED=1` was set;
* the variables from the table above that were set for the run, and the build
  profile (`diag` or release);
* how long it took, and whether the display reset (a TDR) or the process
  stopped.

Nothing from the title's data belongs in the report: no shader dumps, no
frames, no memory, per [docs/asset-policy.md](docs/asset-policy.md). Names of
our own functions, error codes and timings are enough to start.

[docs/debugging-notes-from-a-port.md](docs/debugging-notes-from-a-port.md)
collects what got another RAGE port from "the world loads" to "the game
runs": what the title waited for, and how each wait was found.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
