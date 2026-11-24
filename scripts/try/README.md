# VIctory Try

A simple way to try this fork's VIctory on your own PC. Download
**VIctoryTry.exe** from the fork's Releases page, double-click it, and type
two paths: the game files folder and the game's exe. That's all.

It doesn't sign in to anything and doesn't send anything anywhere. It only
downloads things anonymously:
- this fork's source code, and upstream VIctory's new commits;
- the first time only, the tools building needs: Git (a portable copy if you
  have none), Rust and Microsoft's C++ Build Tools (a few GB; Windows asks
  for permission once).

## Upstream's new fixes, added by themselves

Each time it updates, VIctory Try also fetches upstream VIctory
(Marvisak/VIctory). [`upstream.txt`](upstream.txt) names the last upstream
commit this fork already has. Every later upstream commit is applied on top
of the fork, oldest first:
- a fix the fork already has applies as an empty change and is skipped;
- a fix that clashes with the fork's changes is skipped;
- if the build fails with the new fixes added, the fork is built alone, so
  you always get a working build.

The window lists what was added and what was skipped.

## Keeping itself up to date

From 1.3 on, VIctory Try checks the branch's `scripts/try/try.ps1` each time
it updates. If that one is newer than the one inside your VIctoryTry.exe, it
runs the newer one, so fixes to VIctory Try itself need no new download.

## Safety checks

- The game exe can't be VIctoryTry.exe, victory.exe or anything inside
  `%LOCALAPPDATA%\VIctoryTry`. Choosing VIctoryTry.exe as the "game" made
  VIctory run VIctory Try, which started itself again in a loop. A saved path
  like that is rejected and you are asked again.
- A VIctory Try started from inside a game run stops at once.
- Only one VIctory Try runs at a time, because two would build in the same
  folders.
- When a build fails, the window shows the first error.

## What it does

1. Asks for the game files folder (the one with the `.rpf` archives) and the
   game exe, which may be in another folder, and remembers them. VIctory uses
   the folder holding the game's archives as the console's `G:` drive. If the
   folder holds several exe files, it lists them so you can pick one by number.
2. Installs Rust and the C++ Build Tools if they are missing.
3. Downloads the newest source, adds upstream's new fixes, and builds `victory.exe`. The first build takes
   a while; after that it only rebuilds when the source changed.
4. Runs `victory.exe run <game exe> --execute --console series-s` and shows
   the output. From 1.8 the game sees an Xbox Series S by default: it then
   presents its menus on a separate UI plane, which VIctory composites, where
   as a Series X it has shown a black screen so far.
5. When the game stops, you choose what's next:
   - **Enter**: run again;
   - **U**: update to the newest version and run;
   - **G**: change the game paths;
   - **C**: switch between Series S and Series X (remembered);
   - **Q**: quit.

## Requirements

VIctory's own requirements apply:
- Windows 10 or 11 (x64);
- a CPU with AVX2;
- a D3D12 GPU with Shader Model 6.6 and mesh shaders;
- your own copy of the game.

Keep the game outside the `%LOCALAPPDATA%\VIctoryTry` folder.

## Where things go

Everything lives in `%LOCALAPPDATA%\VIctoryTry`:
- `victory\`: the source;
- `target\`: the build;
- `logs\`: the build output and every game run;
- `config.json`: your two paths.

VIctory writes crash details to `%LOCALAPPDATA%\victory\data\crashes.log`. To
remove everything, delete those two folders.

## Windows warnings

The exe isn't signed, so Windows SmartScreen may say "Windows protected your
PC". Click **More info**, then **Run anyway**. The exe is small: it holds
[`try.ps1`](try.ps1), writes it to `%LOCALAPPDATA%\VIctoryTry` and runs it
with Windows PowerShell. You can read the script, and the exe's C source is in
[`build-exe.sh`](build-exe.sh).

## Building the exe

On any OS with Zig (`pip install ziglang` works), run
`scripts/try/build-exe.sh`. The exe lands in `target/try/`. It's a release
download only: never commit it.
