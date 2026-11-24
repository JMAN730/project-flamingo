# VIctory Try: builds this fork's VIctory on your PC and runs your game with
# it. Started by VIctoryTry.exe (see scripts/try/README.md).
#
# Nothing is sent anywhere: no sign-in, no reports, no accounts. It only
# downloads (anonymously) the fork's source as a zip and, the first time, Rust
# and Microsoft's C++ build tools, which building needs.
#
# ASCII only: Windows PowerShell 5.1 reads a script without a BOM as ANSI.

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$TryVersion = '1.0'
$SourceZip = 'https://codeload.github.com/MfDev0/victory/zip/refs/heads/arena/59fabb00-victory'
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
    $c = @{ game_dir = ''; game_exe = ''; source_hash = '' }
    if (Test-Path $ConfigPath) {
        try {
            $j = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
            foreach ($k in @('game_dir', 'game_exe', 'source_hash')) { if ($j.$k) { $c[$k] = [string]$j.$k } }
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

function Ask-Game([bool]$force) {
    $c = $script:Config
    if (-not $force -and $c.game_exe -and (Test-Path -LiteralPath $c.game_exe -PathType Leaf)) {
        Say "game: $($c.game_exe)"
        return $true
    }
    Step 'Where is the game?'
    Say 'Paste each path and press Enter (right-click pastes in this window).'
    $dir = ''; $exe = ''
    while ($true) {
        $dir = Read-Path '  1) Game files folder'
        if (-not $dir) { continue }
        if (Test-Path -LiteralPath $dir -PathType Leaf) { $exe = $dir; $dir = Split-Path -Parent $dir; break }
        if (Test-Path -LiteralPath $dir -PathType Container) { break }
        Bad "not found: $dir"
    }
    $dir = (Resolve-Path -LiteralPath $dir).ProviderPath.TrimEnd('\')
    if ($dir.StartsWith($TryHome, [StringComparison]::OrdinalIgnoreCase)) {
        Bad "keep the game outside $TryHome"
        return $false
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
        if (-not $in) {
            if ($found.Count -eq 1) { $exe = $found[0].FullName }
            continue
        }
        $n = 0
        if ([int]::TryParse($in, [ref]$n) -and $n -ge 1 -and $n -le $found.Count) { $exe = $found[$n - 1].FullName; continue }
        if (-not [IO.Path]::IsPathRooted($in)) { $in = Join-Path $dir $in }
        if (Test-Path -LiteralPath $in -PathType Leaf) { $exe = (Resolve-Path -LiteralPath $in).ProviderPath }
        else { Bad "not found: $in" }
    }
    $c.game_dir = $dir
    $c.game_exe = $exe
    Save-Config
    Say "game folder: $dir"
    Say "game exe:    $exe"
    return $true
}

# ------------------------------------------------------------- tools

function Has-Msvc {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $false }
    $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    return [bool]$found
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

# Downloads the newest source; returns $true when it changed (or is new).
function Update-Source {
    Step 'Getting the newest VIctory source'
    $zip = Join-Path $TryHome 'source.zip'
    if (-not (Download $SourceZip $zip)) {
        if (Test-Path (Join-Path $Src 'Cargo.toml')) { Say 'using the source downloaded last time'; return $false }
        return $null
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash
    if (($hash -eq $script:Config.source_hash) -and (Test-Path (Join-Path $Src 'Cargo.toml'))) {
        Say 'no changes since last time'
        return $false
    }
    $tmp = Join-Path $TryHome 'unpack'
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    try { Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force }
    catch { Bad "could not unpack the source: $_"; return $null }
    $top = Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1
    if (-not $top) { Bad 'the source zip is empty'; return $null }
    if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
    Move-Item -LiteralPath $top.FullName -Destination $Src
    Remove-Item -Recurse -Force $tmp
    $script:Config.source_hash = $hash
    Save-Config
    Say 'new source unpacked'
    return $true
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
        return $false
    }
    Say 'build finished'
    return $true
}

# ------------------------------------------------------------- run

function Run-Game {
    $c = $script:Config
    Step 'Running the game with VIctory'
    Say "game: $($c.game_exe)"
    Say 'To stop: close the game window. (Ctrl+C closes VIctory Try too.)'
    $env:VICTORY_GAME_DIR = $c.game_dir
    $env:RUST_BACKTRACE = '1'
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $log = Join-Path $LogDir "game-$stamp.log"
    Push-Location $TryHome
    try {
        & (Victory-Exe) run $c.game_exe --execute 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
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
Say 'Builds the newest VIctory from MfDev0/victory on this PC and runs your game.'
Say 'Nothing is uploaded or reported anywhere.'

$script:Config = Load-Config
while (-not (Ask-Game $false)) {}
if (-not (Ensure-Tools)) { exit 1 }
$changed = Update-Source
if ($null -eq $changed) { Bad 'no source to build: check the internet connection'; exit 1 }
if ($changed -or -not (Test-Path (Victory-Exe))) { if (-not (Build)) { exit 1 } }

while ($true) {
    Run-Game
    Write-Host ''
    Say 'Enter = run again    U = update to the newest version and run'
    Say 'G = change the game paths    Q = quit'
    $k = (Read-Host '  Choice').Trim().ToUpper()
    if ($k -eq 'Q') { exit 0 }
    if ($k -eq 'G') { while (-not (Ask-Game $true)) {} }
    if ($k -eq 'U') {
        $changed = Update-Source
        if ($changed -or -not (Test-Path (Victory-Exe))) { if (-not (Build)) { Say 'keeping the last working build' } }
    }
}
