# Windows and GPU compatibility

This guide describes the requirements and behavior in the current `main`
branch. A test or fix from another branch is not evidence that the same
hardware works on `main`; check the commit being tested.

## Check the adapter

The host needs Windows 10/11 x64, an AVX2 CPU, and a Direct3D 12 adapter with
shader model 6.6 and resource binding tier 3. The world also needs mesh shader
support. An adapter without mesh shaders may still start the menus, but it
cannot render the world's mesh draws.

When several adapters are present, VIctory prefers the first adapter that has
all required features. Set `VICTORY_ADAPTER` to a case-insensitive substring
of an adapter name to select one explicitly. If no adapter has every feature,
the startup log reports the missing capabilities of the selected adapter.
Selecting an adapter does not add unsupported features.

`VICTORY_D3D12_WARP=1` selects the software adapter for diagnosis. It is slow
and is not a substitute for testing the physical GPU.

## Windows 10 and the Agility SDK

Windows 10's in-box Direct3D 12 runtime reports shader model 6.5, while this
project needs shader model 6.6. The Windows build can opt into Microsoft's
[Direct3D 12 Agility SDK](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2)
by placing its matching `D3D12Core.dll` at
`target\<profile>\D3D12\D3D12Core.dll` before building. For example, use
`target\debug` for the default development build or `target\release` for a
release build.

The shader crate detects that file during the build, exports the SDK loader
settings, and copies the runtime beside test executables. If the file is not
present, the build leaves the system runtime in use; it does not download the
SDK automatically. The configured SDK version is 620, matching package
1.620.2. Keep the redistributable runtime next to the executable at the
configured `D3D12` path.

## Work not included in main

Status checked on 2026-11-25; these proposals are not guarantees of support:

- The mesh-shader fallback is not in this repository's `main`. The related
  [Marvisak/VIctory#961](https://github.com/Marvisak/VIctory/pull/961) is
  still open in the previous repository; its report says full-title world
  rendering on a GPU without mesh shaders has not been verified end to end.
- [#593](https://github.com/JMAN730/project-flamingo/pull/593) is an
  opt-in Wave64 compute experiment, not default behavior, and reports a
  D3D12 debug-layer error. [#611](https://github.com/JMAN730/project-flamingo/pull/611)
  is a narrower follow-up for indexed lane reads; it does not establish
  general Wave64 or full-title compatibility.

Check the current branch and linked reports before relying on these statuses;
pull requests can change or close independently of `main`.

## Reporting a problem

Include the GPU model, driver version, Windows build, tested VIctory commit,
and relevant `crashes.log` lines. Note whether `VICTORY_D3D12_WARP=1`
reproduces the issue and list any diagnostic environment variables used.
The [GPU failure reporting notes in the README](../README.md#reporting-a-gpu-failure)
describe useful details to collect. Do not attach title assets, shader dumps,
frames, or memory captures; see the [asset policy](asset-policy.md).
