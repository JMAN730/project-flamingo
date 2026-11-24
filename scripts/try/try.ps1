# VIctory Try: builds this fork's VIctory on your PC and runs your game with
# it. Started by VIctoryTry.exe (see scripts/try/README.md).
#
# Nothing is sent anywhere: no sign-in, no reports, no accounts. It only
# downloads (anonymously) the fork's source, upstream VIctory's new commits
# and, the first time, Git, Rust and Microsoft's C++ build tools.
#
# Upstream's new fixes are added by themselves: the branch's
# scripts/try/upstream.txt names the last upstream commit the fork already
# has; each later one is applied on top, oldest first. Ones already in the
# fork (empty) or conflicting are skipped, and if the build fails with them
# the fork is built alone.
#
# ASCII only: Windows PowerShell 5.1 reads a script without a BOM as ANSI.

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$TryVersion = '1.9'
$ForkUrl = 'https://github.com/MfDev0/victory.git'
$Branch = 'arena/59fabb00-victory'
$UpstreamUrl = 'https://github.com/Marvisak/VIctory.git'
$MinGitUrl = 'https://github.com/git-for-windows/git/releases/download/v2.56.1.windows.1/MinGit-2.56.1-64-bit.zip'
$TryHome = Join-Path $env:LOCALAPPDATA 'VIctoryTry'
$Src = Join-Path $TryHome 'victory'
$Target = Join-Path $TryHome 'target'
$ConfigPath = Join-Path $TryHome 'config.json'
$LogDir = Join-Path $TryHome 'logs'
New-Item -ItemType Directory -Force -Path $TryHome, $LogDir | Out-Null
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
try { $Host.UI.RawUI.WindowTitle = 'VIctory Try' } catch {}

function Say([string]$msg) { Write-Host ('  ' + $msg) }
function Step([string]$msg) { Write-Host ''; Write-Host ('== ' + $msg) -ForegroundColor Cyan }
function Bad([string]$msg) { Write-Host ('  ' + $msg) -ForegroundColor Red }

function Load-Config {
    # console: series-s shows the menus (two planes, composited), where
    # series-x has stayed black for everyone so far (upstream #841, #211).
    $c = @{ game_dir = ''; game_exe = ''; built_key = ''; added = ''; console = 'series-s' }
    if (Test-Path $ConfigPath) {
        try {
            $j = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
            foreach ($k in @('game_dir', 'game_exe', 'built_key', 'added', 'console')) { if ($j.$k) { $c[$k] = [string]$j.$k } }
        } catch {}
    }
    return $c
}
function Save-Config { try { ($script:Config | ConvertTo-Json) | Set-Content -Path $ConfigPath -Encoding ASCII } catch {} }

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
    if ((Test-Path $cargoBin) -and ($env:Path -notlike "*$cargoBin*")) { $env:Path += ";$cargoBin" }
}

function Find-Exe([string]$name) {
    $c = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    return $null
}

function Download([string]$url, [string]$file) {
    try { Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $file; return $true }
    catch { Bad "download failed: $url ($_)"; return $false }
}

# ------------------------------------------------------------- the game

function Read-Path([string]$prompt) {
    $in = Read-Host $prompt
    if ($null -eq $in) { return '' }
    return $in.Trim().Trim('"').Trim("'").Trim()
}

# Why `exe` cannot be the game, or '' if it can. VIctory Try itself as the
# "game" made VIctory run VIctory Try, which ran itself again, endlessly.
function Not-Game([string]$exe) {
    $name = [IO.Path]::GetFileName($exe)
    if ($name -like 'VIctoryTry*' -or $name -ieq 'victory.exe' -or $name -ieq 'powershell.exe') {
        return "$name is not the game (it is VIctory Try or VIctory itself)"
    }
    try {
        $full = (Resolve-Path -LiteralPath $exe).ProviderPath
        if ($full.StartsWith($TryHome, [StringComparison]::OrdinalIgnoreCase)) { return "keep the game outside $TryHome" }
        $info = (Get-Item -LiteralPath $full).VersionInfo
        if ($info.ProductName -eq 'VIctory Try' -or $info.FileDescription -eq 'VIctory Try') {
            return "$name is VIctory Try itself, not the game"
        }
    } catch {}
    return ''
}

function Ask-Game([bool]$force) {
    $c = $script:Config
    if (-not $force -and $c.game_exe -and (Test-Path -LiteralPath $c.game_exe -PathType Leaf)) {
        $why = Not-Game $c.game_exe
        if (-not $why) {
            Say "game: $($c.game_exe)"
            if ($c.game_dir) { Say "game files: $($c.game_dir)" }
            return $true
        }
        Bad "the saved game path is wrong: $why"
        $c.game_exe = ''
        Save-Config
    }
    Step 'Where is the game?'
    Say 'Paste each path and press Enter (right-click pastes in this window).'
    Say 'The game files folder is the one with the game''s .rpf archives; the exe may be elsewhere.'
    $dir = ''; $exe = ''
    while ($true) {
        $dir = Read-Path '  1) Game files folder (with the .rpf files)'
        if (-not $dir) { continue }
        if (Test-Path -LiteralPath $dir -PathType Leaf) {
            # The exe was given first: keep it, and still ask for the files.
            $why = Not-Game $dir
            if ($why) { Bad $why; continue }
            $exe = (Resolve-Path -LiteralPath $dir).ProviderPath
            $exeDir = Split-Path -Parent $exe
            Say "game exe: $exe"
            $dir = Read-Path "  1) Game files folder (with the .rpf files) [Enter = $exeDir]"
            if (-not $dir) { $dir = $exeDir }
        }
        if (Test-Path -LiteralPath $dir -PathType Container) { break }
        Bad "not found: $dir"
        $exe = ''
    }
    $dir = (Resolve-Path -LiteralPath $dir).ProviderPath.TrimEnd('\')
    if ($dir.StartsWith($TryHome, [StringComparison]::OrdinalIgnoreCase)) {
        Bad "keep the game outside $TryHome"
        return $false
    }
    $rpfs = @(Get-ChildItem -LiteralPath $dir -Filter '*.rpf' -File -ErrorAction SilentlyContinue)
    if ($rpfs.Count -eq 0) {
        Say 'note: no .rpf files in that folder. If it is not the game files folder, press G later to change it.'
    }
    while (-not $exe) {
        $found = @(Get-ChildItem -LiteralPath $dir -Filter '*.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 9)
        if ($found.Count -gt 1) {
            Say 'exe files in that folder:'
            for ($i = 0; $i -lt $found.Count; $i++) { Say ("  {0}) {1}" -f ($i + 1), $found[$i].Name) }
        }
        $hint = ''
        if ($found.Count -eq 1) { $hint = " [Enter = $($found[0].Name)]" }
        elseif ($found.Count -gt 1) { $hint = ' (a number above, or the path)' }
        $in = Read-Path "  2) Game exe file$hint"
        $pick = ''
        $n = 0
        if (-not $in) {
            if ($found.Count -eq 1) { $pick = $found[0].FullName } else { continue }
        } elseif ([int]::TryParse($in, [ref]$n) -and $n -ge 1 -and $n -le $found.Count) {
            $pick = $found[$n - 1].FullName
        }
        if ($pick) {
            $why = Not-Game $pick
            if ($why) { Bad $why; Say 'pick the game''s own exe (the Xbox title, usually next to MicrosoftGame.config)' }
            else { $exe = $pick }
            continue
        }
        if (-not [IO.Path]::IsPathRooted($in)) { $in = Join-Path $dir $in }
        if (Test-Path -LiteralPath $in -PathType Leaf) { $exe = (Resolve-Path -LiteralPath $in).ProviderPath }
        else { Bad "not found: $in" }
        if ($exe) {
            $why = Not-Game $exe
            if ($why) { Bad $why; Say 'pick the game''s own exe (the Xbox title, usually next to MicrosoftGame.config)'; $exe = '' }
        }
    }
    $c.game_dir = $dir
    $c.game_exe = $exe
    Save-Config
    Say "game files: $dir"
    Say "game exe:   $exe"
    return $true
}

# ------------------------------------------------------------- tools

function Has-Msvc {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $false }
    $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    return [bool]$found
}

# Git, to fetch the fork and upstream and apply upstream's new commits: the
# installed one, else a portable MinGit kept in our folder.
function Ensure-Git {
    $sys = Find-Exe 'git'
    if ($sys) { $script:Git = $sys; return $true }
    $mine = Join-Path $TryHome 'mingit\cmd\git.exe'
    if (-not (Test-Path $mine)) {
        Step 'Getting a portable Git (first time only, about 40 MB)'
        $zip = Join-Path $TryHome 'mingit.zip'
        if (Download $MinGitUrl $zip) {
            try { Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $TryHome 'mingit') -Force } catch { Bad "could not unpack Git: $_" }
            Remove-Item -Force $zip -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path $mine) { $script:Git = $mine; return $true }
    Bad 'Git is missing and could not be downloaded. Check the internet connection and start again.'
    return $false
}

# Runs git in the source folder; the exit code lands in $script:GCode.
function G {
    $out = @(& $script:Git -C $Src -c core.autocrlf=false -c advice.detachedHead=false @args 2>&1 | ForEach-Object { "$_" })
    $script:GCode = $LASTEXITCODE
    return $out
}

function Ensure-Tools {
    Refresh-Path
    if (-not (Has-Msvc)) {
        Step 'Installing Microsoft C++ Build Tools (first time only, a few GB)'
        Say 'Windows will ask for permission. A progress window opens; wait for it to finish.'
        $vs = Join-Path $TryHome 'vs_BuildTools.exe'
        if (Download 'https://aka.ms/vs/17/release/vs_BuildTools.exe' $vs) {
            $p = Start-Process -FilePath $vs -ArgumentList '--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended' -Wait -PassThru
            Say "build tools installer finished (code $($p.ExitCode))"
        }
        if (-not (Has-Msvc)) {
            Bad 'the C++ build tools are still missing. Restart Windows and start VIctory Try again.'
            return $false
        }
    }
    if (-not (Ensure-Git)) { return $false }
    if (-not (Find-Exe 'rustup')) {
        Step 'Installing Rust (first time only)'
        $ri = Join-Path $TryHome 'rustup-init.exe'
        if (Download 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe' $ri) {
            & $ri -y --profile minimal --default-toolchain none
        }
        Refresh-Path
        if (-not (Find-Exe 'rustup')) {
            Bad 'Rust did not install. Start VIctory Try again; if it keeps failing, install it from https://rustup.rs'
            return $false
        }
    }
    return $true
}

# ------------------------------------------------------------- build

# Fetches the fork and upstream; returns $true when there is something new to
# build, $false when not, $null when there is no source at all.
function Update-Source {
    Step 'Getting the newest VIctory (this fork, plus upstream''s new fixes)'
    if ((Test-Path $Src) -and -not (Test-Path (Join-Path $Src '.git'))) { Remove-Item -Recurse -Force $Src }
    if (-not (Test-Path $Src)) {
        & $script:Git -c core.autocrlf=false clone -q --single-branch --branch $Branch $ForkUrl $Src
        if ($LASTEXITCODE -ne 0) { Bad 'could not download the source: check the internet connection'; return $null }
    }
    $null = G fetch -q origin "+refs/heads/${Branch}:refs/remotes/origin/fork"
    if ($script:GCode -ne 0) { Say 'could not reach the fork: using what was downloaded before' }
    $null = G fetch -q $UpstreamUrl '+refs/heads/main:refs/remotes/upstream/main'
    if ($script:GCode -ne 0) { Say 'could not reach upstream VIctory: building the fork without its new fixes' }
    $fork = (G rev-parse -q --verify refs/remotes/origin/fork) | Select-Object -First 1
    if (-not $fork) { Bad "the source download is broken: delete the victory folder in $TryHome and start again"; return $null }
    $up = (G rev-parse -q --verify refs/remotes/upstream/main) | Select-Object -First 1
    $key = "fork:$fork+$up"
    if (($key -eq $script:Config.built_key) -and (Test-Path (Victory-Exe))) {
        Say 'no changes since last time'
        return $false
    }
    $script:Key = $key
    return $true
}

# Puts the fork in the source folder, then upstream's commits that came after
# what the fork has (scripts/try/upstream.txt). The added subjects land in
# $script:Added (returned arrays: an empty one counted as one fix).
function Prepare-Source([bool]$withUpstream) {
    $null = G cherry-pick --abort
    $null = G checkout -q -f -B try refs/remotes/origin/fork
    $null = G clean -fdq
    $added = @()
    if (-not $withUpstream) { $script:Added = $added; return }
    $up = (G rev-parse -q --verify refs/remotes/upstream/main) | Select-Object -First 1
    if (-not $up) { $script:Added = $added; return }
    $marker = Join-Path $Src 'scripts\try\upstream.txt'
    $synced = ''; $have = @{}
    if (Test-Path $marker) {
        foreach ($line in Get-Content -LiteralPath $marker) {
            if ($line -match '^\s*synced\s+([0-9a-f]{7,40})') { $synced = $Matches[1] }
            elseif ($line -match '^\s*have\s+([0-9a-f]{7,40})') { $have[$Matches[1]] = $true }
        }
    }
    if (-not $synced) { Say 'the fork does not say which upstream commits it has: not adding any'; $script:Added = $added; return }
    $null = G merge-base --is-ancestor $synced $up
    if ($script:GCode -ne 0) { Say 'upstream history changed: not adding its commits this time'; $script:Added = $added; return }
    $new = @(G rev-list --reverse --first-parent "$synced..$up")
    if ($new.Count -eq 0) { Say 'upstream VIctory has nothing new beyond this fork'; $script:Added = $added; return }
    Say "upstream VIctory has $($new.Count) new commit(s) beyond this fork"
    foreach ($sha in $new) {
        $known = $false
        foreach ($h in $have.Keys) { if ($sha.StartsWith($h)) { $known = $true } }
        $subject = (G log -1 --format=%s $sha) | Select-Object -First 1
        if ($known) { Say "  have already: $subject"; continue }
        $parents = @(((G rev-list --parents -n 1 $sha) | Select-Object -First 1) -split ' ').Count - 1
        $pick = @('-c', 'user.name=VIctory Try', '-c', 'user.email=try@localhost', 'cherry-pick', '-x')
        if ($parents -gt 1) { $pick += @('-m', '1') }
        $null = G @pick $sha
        if ($script:GCode -eq 0) { Say "  added: $subject"; $added += $subject; continue }
        $conflict = @(G diff --name-only --diff-filter=U).Count -gt 0
        $null = G cherry-pick --abort
        $null = G reset -q --hard
        if ($conflict) { Say "  skipped (clashes with this fork's changes): $subject" }
        else { Say "  have already: $subject" }
    }
    $script:Added = $added; return
}

function Victory-Exe { return Join-Path $Target 'release\victory.exe' }

function Build {
    Step 'Building VIctory (the first build takes a while; later ones are quicker)'
    $channel = ''
    $tc = Join-Path $Src 'rust-toolchain.toml'
    if (Test-Path $tc) {
        $m = Select-String -LiteralPath $tc -Pattern '^\s*channel\s*=\s*"([^"]+)"' | Select-Object -First 1
        if ($m) { $channel = $m.Matches[0].Groups[1].Value }
    }
    if ($channel) {
        Say "Rust $channel"
        & rustup toolchain install $channel --profile minimal --target x86_64-pc-windows-msvc
    }
    $env:CARGO_TARGET_DIR = $Target
    $log = Join-Path $LogDir 'build.log'
    Push-Location $Src
    try {
        & cargo build --release -p victory 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    if ($code -ne 0 -or -not (Test-Path (Victory-Exe))) {
        Bad "the build failed (code $code). The full output is in $log"
        $lines = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
        $first = -1
        for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^error') { $first = $i; break } }
        if ($first -lt 0) { $first = [Math]::Max(0, $lines.Count - 25) }
        Say 'the first error (paste this when reporting it):'
        $lines[$first..([Math]::Min($lines.Count - 1, $first + 24))] | ForEach-Object { Write-Host ('    ' + $_) }
        return $false
    }
    Say 'build finished'
    return $true
}

# The fork with upstream's new fixes; if that does not build, the fork alone.
function Build-All {
    $null = Prepare-Source $true
    $added = @($script:Added | Where-Object { $_ })
    $ok = Build
    if (-not $ok -and $added.Count -gt 0) {
        Bad "the build failed with upstream's new fixes added: building this fork without them"
        $added = @()
        $null = Prepare-Source $false
        $ok = Build
    }
    if ($ok) {
        $script:Config.built_key = $script:Key
        $script:Config.added = ($added -join "`n")
        Save-Config
        if ($added.Count -gt 0) { Say "this build includes $($added.Count) new upstream fix(es)" }
    }
    return $ok
}

# ------------------------------------------------------------- run

function Run-Game {
    $c = $script:Config
    Step 'Running the game with VIctory'
    Say "game: $($c.game_exe)"
    # What victory.exe was built from (not just what was downloaded): a
    # failed build keeps the last working one.
    $built = ''
    if ("$($c.built_key)" -match '^fork:([0-9a-f]{7,40})') { $built = $Matches[1] }
    $fetched = (G rev-parse -q --verify refs/remotes/origin/fork) | Select-Object -First 1
    if ($built) {
        $line = (G log -1 --format='%h %s' $built) | Select-Object -First 1
        Say "VIctory: built from $line"
    }
    if ($built -and $fetched -and ($built -ne $fetched)) {
        $new = (G log -1 --format='%h %s' $fetched) | Select-Object -First 1
        Bad "this is an OLDER build: the newest fork ($new) did not build."
        Bad "Press U to try again; if it fails, send $(Join-Path $LogDir 'build.log')"
    }
    if ($c.added) { Say "plus upstream: $(@($c.added -split "`n").Count) new fix(es) on top" }
    if ($c.console -ne 'series-x') { $c.console = 'series-s' }
    Say "console: Xbox $(if ($c.console -eq 'series-s') { 'Series S' } else { 'Series X' })   (C switches)"
    Say "command: victory.exe run `"$($c.game_exe)`" --execute --console $($c.console)   (VICTORY_GAME_DIR=$($c.game_dir))"
    Say 'To stop: close the game window. (Ctrl+C closes VIctory Try too.)'
    $env:VICTORY_GAME_DIR = $c.game_dir
    $env:RUST_BACKTRACE = '1'
    # Long shader loops hang the GPU (device hung about 40 s into the world):
    # leave a loop after 1024 turns unless set already (64 if it still hangs).
    if (-not $env:VICTORY_SHADER_LOOP_LIMIT) { $env:VICTORY_SHADER_LOOP_LIMIT = '1024' }
    # Series X draws the world at 2560x1440 and its 4K upscale stays black:
    # show the 1440p picture instead (set VICTORY_PRESENT_1440P=0 to turn off).
    if ($c.console -eq 'series-x' -and -not $env:VICTORY_PRESENT_1440P) { $env:VICTORY_PRESENT_1440P = '1' }
    Say "shader loop limit: $($env:VICTORY_SHADER_LOOP_LIMIT)$(if ($env:VICTORY_PRESENT_1440P) { "   world view: $($env:VICTORY_PRESENT_1440P)" })"
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $log = Join-Path $LogDir "game-$stamp.log"
    Push-Location $TryHome
    try {
        & (Victory-Exe) run $c.game_exe --execute --console $c.console 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    Write-Host ''
    Say "the game stopped (exit code $code)"
    Say "its output: $log"
    $crashes = Join-Path $env:LOCALAPPDATA 'victory\data\crashes.log'
    if (Test-Path $crashes) { Say "crash details, if it crashed: $crashes" }
}

# ------------------------------------------------------------- main

Write-Host ''
Write-Host "  VIctory Try $TryVersion" -ForegroundColor Green
Say 'Builds the newest VIctory from MfDev0/victory, with upstream VIctory''s new'
Say 'fixes added by themselves, on this PC and runs your game.'
Say 'Nothing is uploaded or reported anywhere.'

# Started from inside a game run (the game path pointed at VIctory Try, or a
# title started it): stop instead of running again inside it.
if ($env:VICTORY_TRY_RUNNING -eq '1' -and $env:VICTORY_TRY_HANDOFF -ne '1') {
    Bad 'VIctory Try was started from inside a game run: stopping this copy.'
    Bad 'If this repeats, press G in the first VIctory Try window and give the game''s own exe.'
    exit 3
}
$handoff = $env:VICTORY_TRY_HANDOFF -eq '1'
$env:VICTORY_TRY_HANDOFF = ''
$env:VICTORY_TRY_RUNNING = '1'

# One VIctory Try at a time: two would build in the same folders.
$script:Lock = New-Object System.Threading.Mutex($false, 'Local\VIctoryTry')
$owned = $false
# An abandoned lock (a copy that was killed) is ours now.
try { $owned = $script:Lock.WaitOne(0) } catch { $owned = $true }
if (-not $owned) {
    Bad 'VIctory Try is already running in another window: use that one, or close it first.'
    exit 4
}

# A newer VIctory Try on the branch than this one: run that instead, so its
# fixes need no new download of VIctoryTry.exe.
function Run-Newer {
    if ($handoff) { return }
    $text = (G show refs/remotes/origin/fork:scripts/try/try.ps1) -join "`r`n"
    if ($script:GCode -ne 0 -or $text -notmatch "\`$TryVersion = '([0-9.]+)'") { return }
    $theirs = $Matches[1]
    try { if ([version]$theirs -le [version]$TryVersion) { return } } catch { return }
    Step "Starting VIctory Try $theirs (newer than this $TryVersion)"
    $newer = Join-Path $TryHome 'try-latest.ps1'
    Set-Content -LiteralPath $newer -Value $text -Encoding ASCII
    $script:Lock.ReleaseMutex()
    $env:VICTORY_TRY_HANDOFF = '1'
    $ps = (Get-Process -Id $PID).Path
    & $ps -NoProfile -ExecutionPolicy Bypass -File $newer
    exit $LASTEXITCODE
}

$script:Config = Load-Config
while (-not (Ask-Game $false)) {}
if (-not (Ensure-Tools)) { exit 1 }
$changed = Update-Source
if ($null -eq $changed) { Bad 'no source to build: check the internet connection'; exit 1 }
Run-Newer
if ($changed -or -not (Test-Path (Victory-Exe))) { if (-not (Build-All)) { exit 1 } }

while ($true) {
    Run-Game
    Write-Host ''
    Say 'Enter = run again    U = update to the newest version and run'
    Say 'G = change the game paths    C = switch Series S / Series X    Q = quit'
    $k = (Read-Host '  Choice').Trim().ToUpper()
    if ($k -eq 'Q') { exit 0 }
    if ($k -eq 'G') { while (-not (Ask-Game $true)) {} }
    if ($k -eq 'C') {
        $script:Config.console = $(if ($script:Config.console -eq 'series-s') { 'series-x' } else { 'series-s' })
        Save-Config
    }
    if ($k -eq 'U') {
        $changed = Update-Source
        if ($changed -or -not (Test-Path (Victory-Exe))) { if (-not (Build-All)) { Bad 'the new build FAILED: the game runs on the last working build (older fixes). Send the error above.' } }
    }
}
