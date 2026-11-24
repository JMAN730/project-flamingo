# VIctory Pilot

Pilot turns a Windows PC into a test machine that works with the developer
by itself. You start it once and leave it running. It then keeps doing four
things:

1. **Builds and tests every new commit** on the branch, including the
   Windows-only crates (`victory-gpu`, `victory-hle`, `victory-loader`) that
   cannot be compiled where the fixes are written.
2. **Runs the game**, once you have chosen it, and watches for crashes, hangs
   (no output for a few minutes) and GPU device removal.
3. **Reports what breaks** as issues on
   [MfDev0/victory](https://github.com/MfDev0/victory/issues), titled
   `[Pilot] ...`. Each report has the machine, the commit, the new
   `crashes.log` entries, the most frequent warnings and the end of the
   output. It also keeps a `[Pilot] status` issue up to date.
4. **Picks up fixes**: when a new commit lands on the branch, Pilot updates,
   rebuilds, retests and reruns the game. If a crash does not come back,
   Pilot closes its report with a note.

A dashboard at `http://localhost:8765/` shows all of it live: what Pilot is
doing, the game's output, the reports, and buttons to choose the game, run,
stop, rebuild, test, or send a note ("the screen went black after the
logo").

## Start

1. Download [`VIctoryPilot.cmd`](https://github.com/MfDev0/victory/raw/refs/heads/arena/59fabb00-victory/scripts/pilot/VIctoryPilot.cmd)
   (right-click, *Save link as...*).
2. Double-click it. If Windows SmartScreen warns about it, choose
   *More info*, then *Run anyway*.
3. The first start installs what is missing with `winget`: Git, Rust, the
   GitHub CLI and the Visual Studio C++ build tools. Windows asks for
   permission. The build tools are a large download.
4. Sign in to GitHub when asked. Press Enter, then paste the code shown in
   the window into the browser page. Reports are sent from your account.
5. The dashboard opens. Keep the Pilot window open; closing it stops Pilot.

You do not need the game for the first part: Pilot builds and tests on its
own. When you have the game, click **Choose game...** and pick the title's
executable. Testing the game starts then.

The game must be your own copy. Keep it outside Pilot's folder.

## What it runs, and what the developer can change

Pilot only runs `git`, `cargo build` and `cargo test` for this repository,
`gh` (for issues on the fork) and the built `victory.exe`. Building runs the
repository's code, so only run Pilot on a branch you trust.

The branch's [`remote.json`](remote.json) sets the next run. The developer
uses it to switch on more logging or another setting without asking you.
Pilot checks every value against an allowlist and ignores anything else:

| Key | Allowed |
|---|---|
| `pause` | `true` stops Pilot from building and running until it is `false` again |
| `profile` | `release`, or `diag` (the diagnostics build) |
| `env` | variables named `VICTORY_*`, and `RUST_BACKTRACE` |
| `args` | `--console series-s` or `series-x`, `--unimplemented trap` or `zero`, `--strict` |
| `test_crates` | crates of this workspace |
| `test_filter` | a test name filter (letters, digits, `_`, `:`) |
| `hang_minutes` | minutes without output before a run counts as hung (2-120) |
| `survive_minutes` | minutes a run must last before an old crash report is closed (2-120) |

When the branch has a newer `pilot.ps1`, Pilot replaces itself and restarts.
It waits until nothing is running.

## Privacy

Reports contain logs only: no game files, shaders, screenshots or memory,
per [docs/asset-policy.md](../../docs/asset-policy.md). Pilot replaces your
user name, your profile folder and the game's folder with placeholders
before sending anything. The issues are public.

Pilot keeps its files under `%LOCALAPPDATA%\VIctoryPilot`: the source,
build output, `logs\` (every build, test and game run) and its settings. To
remove Pilot, delete that folder.

## The developer's side

`scripts/pilot/watch.sh [seconds]` waits until a `[Pilot]` report is new or
updated, prints it with its comments, and exits. A fix is pushed to the
branch, which Pilot picks up by itself.
