# VIctory Pilot: builds, tests and runs VIctory on this PC, sends what
# breaks to GitHub as issues, and picks up fixes from the branch by itself.
# Started by VIctoryPilot.cmd; see scripts/pilot/README.md.
#
# Pilot only runs git, cargo (build and test of this repository), gh (issues
# on the fork) and victory.exe. The branch's scripts/pilot/remote.json can
# change the next run's settings, but only through the allowlists below: no
# command from the branch is run as such.
#
# ASCII only: Windows PowerShell 5.1 reads a script without a BOM as ANSI.

$ErrorActionPreference = 'Continue'
$PilotVersion = '1.0'
$Repo = 'MfDev0/victory'
$Branch = 'arena/59fabb00-victory'
$PilotHome = Join-Path $env:LOCALAPPDATA 'VIctoryPilot'
$RepoDir = Join-Path $PilotHome 'victory'
$LogDir = Join-Path $PilotHome 'logs'
$ConfigPath = Join-Path $PilotHome 'config.json'
$StatePath = Join-Path $PilotHome 'state.json'
New-Item -ItemType Directory -Force -Path $PilotHome, $LogDir | Out-Null
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
try { $Host.UI.RawUI.WindowTitle = 'VIctory Pilot' } catch {}

# ---------------------------------------------------------------- basics

$script:Activity = New-Object System.Collections.Generic.List[string]
function Say([string]$msg) {
    $line = (Get-Date -Format 'HH:mm:ss') + '  ' + $msg
    Write-Host $line
    $script:Activity.Add($line)
    while ($script:Activity.Count -gt 300) { $script:Activity.RemoveAt(0) }
    try { Add-Content -Path (Join-Path $LogDir 'pilot.log') -Value $line -Encoding UTF8 } catch {}
}

function ConvertTo-Hash($obj) {
    if ($null -eq $obj) { return $null }
    if ($obj -is [System.Collections.IDictionary]) { return $obj }
    if ($obj -is [System.Management.Automation.PSCustomObject]) {
        $h = @{}
        foreach ($p in $obj.PSObject.Properties) { $h[$p.Name] = ConvertTo-Hash $p.Value }
        return $h
    }
    if (($obj -is [System.Collections.IEnumerable]) -and ($obj -isnot [string])) {
        $list = New-Object System.Collections.ArrayList
        foreach ($item in $obj) { [void]$list.Add((ConvertTo-Hash $item)) }
        return , $list.ToArray()
    }
    return $obj
}

function Load-Json([string]$path) {
    if (Test-Path $path) {
        try { return ConvertTo-Hash (Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json) }
        catch { Say "could not read $path, starting fresh" }
    }
    return @{}
}

function Save-Json([string]$path, $obj) {
    try { ($obj | ConvertTo-Json -Depth 10) | Set-Content -Path $path -Encoding UTF8 } catch { Say "could not save $path" }
}

function Find-Exe([string]$name) {
    $c = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    return $null
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
    $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
    if ((Test-Path $cargoBin) -and ($env:Path -notlike "*$cargoBin*")) { $env:Path += ";$cargoBin" }
    $ghDir = Join-Path $env:ProgramFiles 'GitHub CLI'
    if ((Test-Path $ghDir) -and ($env:Path -notlike "*$ghDir*")) { $env:Path += ";$ghDir" }
}

function Quote-Arg([string]$a) {
    if ($a -eq '') { return '""' }
    if ($a -notmatch '[\s"]') { return $a }
    $escaped = ($a -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Join-Args([string[]]$argv) { return (($argv | ForEach-Object { Quote-Arg $_ }) -join ' ') }

# Runs a program to the end and returns its exit code and output.
function Run-Sync([string]$exe, [string[]]$argv, [string]$cwd = $PilotHome, [int]$timeout = 600) {
    $path = $exe
    if (-not [IO.Path]::IsPathRooted($exe)) { $path = Find-Exe $exe }
    if (-not $path) { return @{ Code = -1; Out = "$exe is not installed" } }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $path
    $psi.Arguments = Join-Args $argv
    $psi.WorkingDirectory = $cwd
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    try { $p = [System.Diagnostics.Process]::Start($psi) }
    catch { return @{ Code = -1; Out = "$exe could not start: $_" } }
    $errTask = $p.StandardError.ReadToEndAsync()
    $outTask = $p.StandardOutput.ReadToEndAsync()
    if (-not $p.WaitForExit($timeout * 1000)) {
        Kill-Tree $p.Id
        return @{ Code = -2; Out = "$exe timed out after $timeout s" }
    }
    $p.WaitForExit()
    return @{ Code = $p.ExitCode; Out = ($outTask.Result + "`n" + $errTask.Result).Trim() }
}

function Kill-Tree([int]$id) {
    try { & taskkill.exe /PID $id /T /F 2>&1 | Out-Null } catch {}
}

function File-Size([string]$path) {
    try { return (Get-Item -LiteralPath $path -ErrorAction Stop).Length } catch { return 0 }
}

# The last lines of a file another process may still be writing.
function Read-Tail([string]$path, [int]$maxBytes = 262144, [long]$from = -1) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]'ReadWrite, Delete')
        try {
            $len = $fs.Length
            $start = [Math]::Max([long]0, $len - $maxBytes)
            if ($from -ge 0) { $start = [Math]::Max($from, $start) }
            [void]$fs.Seek($start, [System.IO.SeekOrigin]::Begin)
            $buf = New-Object byte[] ([int]($len - $start))
            $n = $fs.Read($buf, 0, $buf.Length)
            $text = [System.Text.Encoding]::UTF8.GetString($buf, 0, $n)
        } finally { $fs.Close() }
        $text = $text -replace "\x1b\[[0-9;?]*[A-Za-z]", ''
        return ($text -split "`r?`n")
    } catch { return }
}

# Hides this PC's user name and game folder in anything sent out.
function Scrub([string]$text) {
    if (-not $text) { return '' }
    $out = $text
    if ($script:Config -and $script:Config.game_exe) {
        $gameDir = Split-Path -Parent $script:Config.game_exe
        if ($gameDir) { $out = $out.Replace($gameDir, '<game folder>') }
    }
    if ($env:USERPROFILE) { $out = $out.Replace($env:USERPROFILE, '%USERPROFILE%') }
    if ($env:USERNAME -and $env:USERNAME.Length -ge 3) { $out = $out.Replace($env:USERNAME, '<user>') }
    return $out
}

function Short([string]$sha) { if ($sha -and $sha.Length -ge 7) { return $sha.Substring(0, 7) }; return '?' }

# ---------------------------------------------------------------- state

$script:Config = Load-Json $ConfigPath
foreach ($pair in @(@('game_exe', ''), @('auto_run', $true), @('run_tests', $true), @('extra_args', ''))) {
    if (-not $script:Config.ContainsKey($pair[0])) { $script:Config[$pair[0]] = $pair[1] }
}
$script:State = Load-Json $StatePath
if (-not $script:State.ContainsKey('issues') -or $null -eq $script:State.issues) { $script:State.issues = @() }
$script:State.issues = @($script:State.issues | Where-Object { $_ -is [System.Collections.IDictionary] })
foreach ($k in @('built', 'tested', 'ran')) { if (-not $script:State.ContainsKey($k)) { $script:State[$k] = '' } }

function Save-State { Save-Json $StatePath $script:State }
function Save-Config { Save-Json $ConfigPath $script:Config }

$script:Phase = 'starting'
$script:Detail = ''
$script:LocalSha = ''
$script:Subject = ''
$script:RemoteSha = ''
$script:Job = $null
$script:LastJob = $null
$script:PendingUpdate = $false
$script:SelfUpdate = $null
$script:Login = ''
$script:CanReport = $false
$script:IssuesOn = $false
$script:LastRemoteCheck = [DateTime]::MinValue
$script:LastHeartbeat = [DateTime]::MinValue
$script:Machine = @{}
$script:Remote = @{}

# ---------------------------------------------------------------- setup

function Has-MsvcTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $false }
    $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    return [bool]$found
}

function Winget-Install([string]$id, [string]$override = '') {
    Say "installing $id (Windows may ask for permission; this can take a while)"
    $wargs = @('install', '--id', $id, '-e', '--source', 'winget', '--accept-source-agreements', '--accept-package-agreements')
    if ($override) { $wargs += @('--override', $override) } else { $wargs += '--silent' }
    & winget @wargs
    Refresh-Path
}

function Ensure-Tools {
    $script:Phase = 'setting up'
    Refresh-Path
    $need = @()
    if (-not (Find-Exe 'git')) { $need += 'Git.Git' }
    if (-not (Find-Exe 'cargo')) { $need += 'Rustlang.Rustup' }
    if (-not (Find-Exe 'gh')) { $need += 'GitHub.cli' }
    $msvc = Has-MsvcTools
    if (($need.Count -eq 0) -and $msvc) { Say 'tools: git, Rust, GitHub CLI and the C++ build tools are installed'; return $true }
    if (-not (Find-Exe 'winget')) {
        Say 'winget is missing. Install "App Installer" from the Microsoft Store (opening it now), then start Pilot again.'
        Start-Process 'ms-windows-store://pdp/?ProductId=9NBLGGH4NNS1'
        return $false
    }
    $listed = @($need)
    if (-not $msvc) { $listed += 'C++ build tools' }
    Say ('first start: installing what is missing: ' + ($listed -join ', '))
    foreach ($id in $need) { Winget-Install $id }
    if (-not $msvc) {
        Winget-Install 'Microsoft.VisualStudio.2022.BuildTools' '--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    }
    Refresh-Path
    $missing = @()
    if (-not (Find-Exe 'git')) { $missing += 'git' }
    if (-not (Find-Exe 'cargo')) { $missing += 'Rust (cargo)' }
    if (-not (Find-Exe 'gh')) { $missing += 'GitHub CLI' }
    if (-not (Has-MsvcTools)) { $missing += 'C++ build tools' }
    if ($missing.Count -gt 0) {
        Say ('still missing: ' + ($missing -join ', ') + '. Close this window and start Pilot again (a restart of Windows may be needed).')
        return $false
    }
    return $true
}

function Ensure-GitHub {
    $r = Run-Sync 'gh' @('auth', 'status', '--hostname', 'github.com')
    if ($r.Code -ne 0) {
        Say 'Sign in to GitHub so Pilot can send reports: press Enter below, then paste the code in the browser page.'
        & gh auth login --hostname github.com --git-protocol https --web
        $r = Run-Sync 'gh' @('auth', 'status', '--hostname', 'github.com')
    }
    if ($r.Code -ne 0) { Say 'not signed in to GitHub: Pilot keeps working but cannot send reports'; return $false }
    $who = Run-Sync 'gh' @('api', 'user', '--jq', '.login')
    if ($who.Code -eq 0) { $script:Login = $who.Out.Trim() }
    Say "GitHub: signed in as $($script:Login), reports go to $Repo"
    return $true
}

# Reports need issues on the fork (forks start with them off) and labels.
function Ensure-Issues {
    $r = Run-Sync 'gh' @('api', "repos/$Repo", '--jq', '.has_issues')
    if (($r.Code -eq 0) -and ($r.Out.Trim() -eq 'false')) {
        Say "issues are off on ${Repo}: switching them on"
        [void](Run-Sync 'gh' @('api', '-X', 'PATCH', "repos/$Repo", '-F', 'has_issues=true'))
        $r = Run-Sync 'gh' @('api', "repos/$Repo", '--jq', '.has_issues')
    }
    if (($r.Code -ne 0) -or ($r.Out.Trim() -ne 'true')) {
        Say "issues are off on $Repo and Pilot cannot switch them on: the owner can, under Settings > General > Features > Issues"
        return $false
    }
    $labels = @(
        @('crash-report', 'b60205', 'Pilot: a crash, hang or player report'),
        @('build-failure', 'd93f0b', 'Pilot: the branch does not build on Windows'),
        @('test-failure', 'fbca04', 'Pilot: Windows tests fail'),
        @('pilot-status', '0e8a16', 'Pilot: the status of a Pilot PC')
    )
    foreach ($l in $labels) { [void](Run-Sync 'gh' @('label', 'create', $l[0], '-R', $Repo, '--color', $l[1], '--description', $l[2], '--force')) }
    return $true
}

function Read-Machine {
    $m = @{ gpu = '?'; driver = '?'; windows = '?'; cpu = '?'; ram_gb = '?' }
    try {
        $gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Basic Display|Remote' })
        if ($gpus.Count -gt 0) {
            $m.gpu = ($gpus | ForEach-Object { $_.Name }) -join ' / '
            $m.driver = ($gpus | ForEach-Object { $_.DriverVersion }) -join ' / '
        }
        $os = Get-CimInstance Win32_OperatingSystem
        $m.windows = "$($os.Caption) build $($os.BuildNumber)"
        $m.ram_gb = [Math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        $m.cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
    } catch {}
    try {
        $smi = Find-Exe 'nvidia-smi'
        if ($smi) {
            $v = (& $smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
            if ($v) { $m.driver = "$($m.driver) (GeForce $($v.Trim()))" }
        }
    } catch {}
    return $m
}

# ---------------------------------------------------------------- the branch

function Remote-Sha {
    $r = Run-Sync 'git' @('ls-remote', "https://github.com/$Repo.git", "refs/heads/$Branch") $PilotHome 60
    if (($r.Code -eq 0) -and ($r.Out -match '([0-9a-f]{40})')) { return $Matches[1] }
    return $null
}

function Update-Repo {
    $script:Phase = 'updating'
    if (-not (Test-Path (Join-Path $RepoDir '.git'))) {
        Say 'downloading the VIctory source (first time only)'
        $r = Run-Sync 'git' @('clone', '--branch', $Branch, "https://github.com/$Repo.git", $RepoDir) $PilotHome 3600
        if ($r.Code -ne 0) { Say ('download failed: ' + $r.Out); return $false }
    }
    $r = Run-Sync 'git' @('fetch', '--quiet', 'origin', $Branch) $RepoDir 900
    if ($r.Code -ne 0) { Say ('update failed: ' + $r.Out); return $false }
    $r = Run-Sync 'git' @('reset', '--hard', '--quiet', 'FETCH_HEAD') $RepoDir 300
    if ($r.Code -ne 0) { Say ('update failed: ' + $r.Out); return $false }
    $old = $script:LocalSha
    $script:LocalSha = (Run-Sync 'git' @('rev-parse', 'HEAD') $RepoDir).Out.Trim()
    $script:Subject = (Run-Sync 'git' @('log', '-1', '--format=%s') $RepoDir).Out.Trim()
    $script:RemoteSha = $script:LocalSha
    $script:PendingUpdate = $false
    if ($old -ne $script:LocalSha) { Say "source at $(Short $script:LocalSha): $($script:Subject)" }
    Load-Remote
    Check-SelfUpdate
    return $true
}

function Check-SelfUpdate {
    $new = Join-Path $RepoDir 'scripts\pilot\pilot.ps1'
    if (-not $PSCommandPath -or -not (Test-Path $new)) { return }
    if ((Get-FileHash $new).Hash -ne (Get-FileHash $PSCommandPath).Hash) {
        $script:SelfUpdate = $new
        Say 'a newer Pilot is on the branch: it takes over once nothing is running'
    }
}

# The branch's run settings, every value checked.
function Load-Remote {
    $r = @{ pause = $false; profile = 'release'; env = @{}; args = @(); test_crates = @('victory-gpu', 'victory-hle', 'victory-loader'); test_filter = ''; hang_minutes = 5; survive_minutes = 10 }
    $path = Join-Path $RepoDir 'scripts\pilot\remote.json'
    $raw = Load-Json $path
    if ($raw.ContainsKey('pause')) { $r.pause = [bool]$raw.pause }
    if ($raw.profile -in @('release', 'diag')) { $r.profile = $raw.profile }
    if ($raw.env -is [System.Collections.IDictionary]) {
        foreach ($k in $raw.env.Keys) {
            $v = [string]$raw.env[$k]
            if ((($k -match '^VICTORY_[A-Z0-9_]{1,40}$') -or ($k -in @('RUST_BACKTRACE'))) -and ($v.Length -le 300) -and ($v -notmatch '[\r\n]')) { $r.env[$k] = $v }
            else { Say "remote.json: env $k ignored" }
        }
    }
    $allowedArgs = @('--console', 'series-s', 'series-x', '--unimplemented', 'trap', 'zero', '--strict')
    foreach ($a in @($raw.args)) {
        if ($null -eq $a) { continue }
        if ($a -in $allowedArgs) { $r.args += $a } else { Say "remote.json: argument $a ignored" }
    }
    $allowedCrates = @('victory', 'victory-gpu', 'victory-hle', 'victory-loader', 'victory-shader', 'victory-formats', 'victory-core', 'victory-game', 'xmem')
    if ($raw.ContainsKey('test_crates')) { $r.test_crates = @(@($raw.test_crates) | Where-Object { $_ -in $allowedCrates }) }
    if (($raw.test_filter -is [string]) -and ($raw.test_filter -match '^[A-Za-z0-9_:]{0,120}$')) { $r.test_filter = $raw.test_filter }
    foreach ($k in @('hang_minutes', 'survive_minutes')) {
        if ($raw.ContainsKey($k)) { $n = 0; if ([int]::TryParse([string]$raw[$k], [ref]$n) -and $n -ge 2 -and $n -le 120) { $r[$k] = $n } }
    }
    $script:Remote = $r
}

function Victory-Exe {
    $dir = 'release'; if ($script:Remote.profile -eq 'diag') { $dir = 'diag' }
    return Join-Path $RepoDir "target\$dir\victory.exe"
}

# ---------------------------------------------------------------- jobs

function Start-LongJob([string]$kind, [string]$exe, [string[]]$argv, [hashtable]$envVars) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $out = Join-Path $LogDir "$kind-$stamp.out.txt"
    $err = Join-Path $LogDir "$kind-$stamp.err.txt"
    $saved = @{}
    $envVars['CARGO_TERM_COLOR'] = 'never'
    foreach ($k in $envVars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $envVars[$k]) }
    $p = $null
    try {
        $p = Start-Process -FilePath $exe -ArgumentList (Join-Args $argv) -WorkingDirectory $RepoDir -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
        $null = $p.Handle
    } catch { Say "$kind could not start: $_" }
    finally { foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
    if (-not $p) { return $false }
    $script:Job = @{
        Kind = $kind; Proc = $p; Out = $out; Err = $err; Started = Get-Date; LastActivity = Get-Date; LastSize = 0
        Sha = $script:LocalSha; Hung = $false; UserStopped = $false; Env = $envVars; Args = $argv; CrashBefore = @{}
    }
    return $true
}

function Start-Build {
    $cargo = Find-Exe 'cargo'
    $script:Phase = 'building'
    $script:Detail = 'compiling VIctory (the first build takes a long time)'
    Say "building $(Short $script:LocalSha) ($($script:Remote.profile))"
    $bargs = @('build', '--locked', '-p', 'victory')
    if ($script:Remote.profile -eq 'diag') { $bargs += @('--profile', 'diag', '--features', 'diagnostics') } else { $bargs += '--release' }
    return Start-LongJob 'build' $cargo $bargs @{}
}

function Start-Tests {
    $crates = @($script:Remote.test_crates)
    if ($crates.Count -eq 0) { return $false }
    $cargo = Find-Exe 'cargo'
    $script:Phase = 'testing'
    $script:Detail = 'running the Windows-only tests: ' + ($crates -join ', ')
    Say "testing $(Short $script:LocalSha): $($crates -join ', ')"
    $targs = @('test', '--locked')
    foreach ($c in $crates) { $targs += @('-p', $c) }
    $targs += '--no-fail-fast'
    if ($script:Remote.test_filter) { $targs += @('--', $script:Remote.test_filter) }
    return Start-LongJob 'test' $cargo $targs @{}
}

function Crash-Logs {
    $h = @{}
    foreach ($root in @((Join-Path $env:LOCALAPPDATA 'victory'))) {
        if ($root -and (Test-Path $root)) {
            Get-ChildItem -Path $root -Recurse -Filter 'crashes.log' -ErrorAction SilentlyContinue -Depth 4 | ForEach-Object { $h[$_.FullName] = $_.Length }
        }
    }
    return $h
}

function Start-Game {
    $exe = Victory-Exe
    if (-not (Test-Path $exe)) { Say 'no build to run yet'; return $false }
    if (-not $script:Config.game_exe -or -not (Test-Path -LiteralPath $script:Config.game_exe)) { Say 'choose the game first (Choose game on the dashboard)'; return $false }
    $script:Phase = 'running the game'
    $script:Detail = 'watching for crashes, hangs and device removal'
    $gargs = @('run', $script:Config.game_exe, '--execute') + @($script:Remote.args)
    $envVars = @{ RUST_BACKTRACE = '1' }
    foreach ($k in $script:Remote.env.Keys) { $envVars[$k] = $script:Remote.env[$k] }
    $before = Crash-Logs
    Say "starting the game on $(Short $script:LocalSha)"
    if (-not (Start-LongJob 'game' $exe $gargs $envVars)) { return $false }
    $script:Job.CrashBefore = $before
    return $true
}

function Stop-Job([string]$why) {
    if ($script:Job) {
        $script:Job.UserStopped = $true
        Say "stopping $($script:Job.Kind): $why"
        Kill-Tree $script:Job.Proc.Id
    }
}

function Job-Lines($j, [int]$maxBytes = 262144) {
    $a = @(Read-Tail $j.Out $maxBytes)
    $b = @(Read-Tail $j.Err $maxBytes)
    return (($a + $b) | Where-Object { $_ -ne '' })
}

function Tick-Job {
    $j = $script:Job
    if (-not $j) { return }
    $size = (File-Size $j.Out) + (File-Size $j.Err)
    if ($size -ne $j.LastSize) { $j.LastSize = $size; $j.LastActivity = Get-Date }
    if (($j.Kind -eq 'game') -and -not $j.Proc.HasExited) {
        $quiet = ((Get-Date) - $j.LastActivity).TotalMinutes
        if ($quiet -ge $script:Remote.hang_minutes) {
            Say "no output for $([int]$quiet) minutes: treating it as a hang"
            $j.Hung = $true
            Kill-Tree $j.Proc.Id
        }
    }
    if ($j.Proc.HasExited) {
        $script:Job = $null
        $script:LastJob = $j
        $code = 0
        try { $code = $j.Proc.ExitCode } catch {}
        $j.Code = $code
        $j.Minutes = [Math]::Round(((Get-Date) - $j.Started).TotalMinutes, 1)
        switch ($j.Kind) {
            'build' { Finish-Build $j }
            'test' { Finish-Tests $j }
            'game' { Finish-Game $j }
        }
    }
}

# ---------------------------------------------------------------- reports

function Issue-Url([int]$n) { return "https://github.com/$Repo/issues/$n" }

function Gh-Body([string]$text) {
    $text = Scrub $text
    if ($text.Length -gt 60000) { $text = $text.Substring(0, 60000) + "`n... (cut)" }
    $file = Join-Path $LogDir 'issue-body.md'
    [IO.File]::WriteAllText($file, $text, (New-Object System.Text.UTF8Encoding $false))
    return $file
}

function New-Issue([string]$kind, [string]$title, [string]$body, [string]$label) {
    if (-not $script:CanReport) { Say "not reported (not signed in): $title"; return 0 }
    $file = Gh-Body $body
    $r = Run-Sync 'gh' @('issue', 'create', '-R', $Repo, '--title', $title, '--body-file', $file, '--label', $label)
    if ($r.Code -ne 0) { $r = Run-Sync 'gh' @('issue', 'create', '-R', $Repo, '--title', $title, '--body-file', $file) }
    if (($r.Code -eq 0) -and ($r.Out -match '/issues/(\d+)')) {
        $n = [int]$Matches[1]
        Say "reported: $title -> $(Issue-Url $n)"
        return $n
    }
    Say ('could not send the report: ' + $r.Out)
    return 0
}

function Comment-Issue([int]$n, [string]$body) {
    if (-not $script:CanReport -or $n -le 0) { return }
    $file = Gh-Body $body
    $r = Run-Sync 'gh' @('issue', 'comment', "$n", '-R', $Repo, '--body-file', $file)
    if ($r.Code -ne 0) { Say ('could not comment: ' + $r.Out) }
}

function Close-Issue($issue, [string]$why) {
    Comment-Issue $issue.number $why
    if ($script:CanReport) { [void](Run-Sync 'gh' @('issue', 'close', "$($issue.number)", '-R', $Repo)) }
    $issue.closed = $true
    Save-State
    Say "closed #$($issue.number): $why"
}

function Open-Issues([string]$kind) { $script:State.issues | Where-Object { $_.kind -eq $kind -and -not $_.closed } }

function Add-Issue([int]$n, [string]$kind, [string]$title, [string]$signature, [double]$atMinutes) {
    if ($n -le 0) { return }
    $script:State.issues = @($script:State.issues) + @(@{ number = $n; kind = $kind; title = $title; signature = $signature; sha = $script:LocalSha; at_minutes = $atMinutes; closed = $false; seen = @($script:LocalSha) })
    Save-State
}

function Machine-Table {
    $m = $script:Machine
    $lines = @(
        '| | |', '|---|---|',
        "| Commit | ``$($script:LocalSha)`` $($script:Subject) |",
        "| GPU | $($m.gpu) |",
        "| Driver | $($m.driver) |",
        "| Windows | $($m.windows) |",
        "| CPU / RAM | $($m.cpu) / $($m.ram_gb) GB |",
        "| Profile | $($script:Remote.profile) |",
        "| Pilot | $PilotVersion |"
    )
    return ($lines -join "`n")
}

function Fence([string[]]$lines, [int]$max) {
    $l = @($lines)
    if ($l.Count -gt $max) { $l = $l[($l.Count - $max)..($l.Count - 1)] }
    return "``````text`n" + (($l -join "`n") -replace '```', "'''") + "`n``````"
}

# ---------------------------------------------------------------- results

function Finish-Build($j) {
    $lines = @(Job-Lines $j)
    if ($j.UserStopped) { $script:Phase = 'idle'; $script:Detail = 'build stopped'; return }
    if ($j.Code -eq 0) {
        Say "build OK in $($j.Minutes) min"
        $script:State.built = $j.Sha
        Save-State
        foreach ($i in @(Open-Issues 'build')) { Close-Issue $i "Builds on ``$($j.Sha)`` (Pilot on $($script:Machine.gpu))." }
        Next-Step
        return
    }
    $errors = New-Object System.Collections.Generic.List[string]
    $take = 0
    foreach ($l in $lines) {
        if ($l -match '^(error|warning: unused)') { $take = 18 }
        if ($take -gt 0) { $errors.Add($l); $take-- }
        if ($errors.Count -gt 400) { break }
    }
    if ($errors.Count -eq 0) { $errors.AddRange([string[]]@($lines | Select-Object -Last 80)) }
    $first = ($errors | Where-Object { $_ -match '^error' } | Select-Object -First 1)
    if (-not $first) { $first = "exit code $($j.Code)" }
    Say "build FAILED: $first"
    $body = "Pilot could not build ``victory`` on Windows.`n`n" + (Machine-Table) + "`n`n### Errors`n" + (Fence $errors.ToArray() 400)
    $open = @(Open-Issues 'build')
    if ($open.Count -gt 0) {
        if ($open[0].seen -notcontains $j.Sha) {
            Comment-Issue $open[0].number ("Still fails on ``$($j.Sha)``.`n`n" + (Fence $errors.ToArray() 300))
            $open[0].seen = @($open[0].seen) + @($j.Sha); Save-State
        }
    } else {
        $n = New-Issue 'build' "[Pilot] build fails on Windows: $first" $body 'build-failure'
        Add-Issue $n 'build' $first $first 0
    }
    $script:Phase = 'waiting for a fix'
    $script:Detail = 'the build failed; reported, waiting for a new commit'
}

function Finish-Tests($j) {
    if ($j.UserStopped) { $script:Phase = 'idle'; return }
    $lines = @(Job-Lines $j 4194304)
    $failed = @($lines | Where-Object { $_ -match '^test (\S+) \.\.\. FAILED' } | ForEach-Object { ($_ -split ' ')[1] } | Select-Object -Unique)
    $results = @($lines | Where-Object { $_ -match '^test result:' })
    $script:State.tested = $j.Sha
    Save-State
    $open = @(Open-Issues 'test')
    if (($j.Code -eq 0) -or (($failed.Count -eq 0) -and ($results.Count -gt 0) -and -not ($results -match 'FAILED'))) {
        Say "tests OK in $($j.Minutes) min ($($results.Count) suites)"
        foreach ($i in $open) { Close-Issue $i "All tests pass on ``$($j.Sha)`` (Pilot on $($script:Machine.gpu))." }
        Next-Step
        return
    }
    if ($failed.Count -eq 0) {
        # The tests did not run: a compile error in a test target.
        Finish-Build $j
        return
    }
    $details = New-Object System.Collections.Generic.List[string]
    $inside = $false
    foreach ($l in $lines) {
        if ($l -match '^---- .* stdout ----') { $inside = $true }
        elseif ($l -match '^(failures:|test result:)') { $inside = $false }
        if ($inside) { $details.Add($l) }
        if ($details.Count -gt 600) { break }
    }
    Say ("tests FAILED: " + ($failed -join ', '))
    $sig = ($failed | Sort-Object) -join ','
    $body = "Windows-only tests fail under Pilot.`n`n" + (Machine-Table) + "`n`n### Failed`n" + (($failed | ForEach-Object { "- ``$_``" }) -join "`n") + "`n`n### Output`n" + (Fence $details.ToArray() 600) + "`n`n### Summary`n" + (Fence $results 40)
    if ($open.Count -gt 0) {
        if (($open[0].signature -ne $sig) -or ($open[0].seen -notcontains $j.Sha)) {
            Comment-Issue $open[0].number ("On ``$($j.Sha)`` these fail:`n" + (($failed | ForEach-Object { "- ``$_``" }) -join "`n") + "`n`n" + (Fence $details.ToArray() 400))
            $open[0].signature = $sig; $open[0].seen = @($open[0].seen) + @($j.Sha); Save-State
        }
    } else {
        $n = New-Issue 'test' "[Pilot] $($failed.Count) Windows test(s) fail: $($failed[0])" $body 'test-failure'
        Add-Issue $n 'test' $failed[0] $sig 0
    }
    Next-Step
}

function Crash-Signature([string[]]$crashLines, [string[]]$outLines, $j) {
    $first = $crashLines | Where-Object { $_ -match '^VIctory: ' } | Select-Object -First 1
    if (-not $first) {
        $first = @($outLines | Where-Object { $_ -match 'panicked at|DEVICE_REMOVED|DEVICE_HUNG|device removed|device hung|DXGI_ERROR|unhandled exception|access violation' }) | Select-Object -Last 1
    }
    if (-not $first) {
        if ($j.Hung) { return "hang: no output for $($script:Remote.hang_minutes) minutes" }
        return ('exit code 0x{0:x8}' -f $j.Code)
    }
    $s = $first -replace '^\S*\s*\d{4}-\d\d-\d\d\S*\s*', ''
    $s = $s -replace '\s\(.*$', ''
    $s = $s -replace '0x[0-9a-fA-F]{9,}', '0x...'
    $s = $s -replace 'thread \S+', 'thread'
    if ($s.Length -gt 140) { $s = $s.Substring(0, 140) }
    return $s.Trim()
}

function Warning-Summary([string[]]$lines) {
    $counts = @{}
    foreach ($l in $lines) {
        if ($l -match '\b(WARN|ERROR)\b') {
            $k = $l -replace '^\S+\s+', '' -replace '0x[0-9a-fA-F]+', '0x_' -replace '\b\d+\b', 'N'
            if ($k.Length -gt 160) { $k = $k.Substring(0, 160) }
            if ($counts.ContainsKey($k)) { $counts[$k]++ } else { $counts[$k] = 1 }
        }
    }
    $top = $counts.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 25 | ForEach-Object { '{0,6}x  {1}' -f $_.Value, $_.Key }
    return $top
}

function Finish-Game($j) {
    $outLines = @(Job-Lines $j 1048576)
    $after = Crash-Logs
    $crashLines = @()
    foreach ($f in $after.Keys) {
        $was = 0; if ($j.CrashBefore.ContainsKey($f)) { $was = $j.CrashBefore[$f] }
        if ($after[$f] -gt $was) { $crashLines += @(Read-Tail $f 262144 $was) }
    }
    $crashLines = @($crashLines | Where-Object { $_ -ne '' })
    $script:State.ran = $j.Sha
    Save-State
    $crashed = $j.Hung -or ($crashLines.Count -gt 0) -or (($j.Code -ne 0) -and -not $j.UserStopped)
    # Fixes that held: open crash reports from older commits this run got past.
    foreach ($i in @(Open-Issues 'crash')) {
        if ($i.sha -eq $j.Sha) { continue }
        $need = [Math]::Max([double]$script:Remote.survive_minutes, 1.5 * [double]$i.at_minutes + 2)
        $sig = ''
        if ($crashed) { $sig = Crash-Signature $crashLines $outLines $j }
        if (($sig -ne $i.signature) -and ($j.Minutes -ge $need)) {
            Close-Issue $i "Not seen on ``$($j.Sha)``: the run lasted $($j.Minutes) min (this crash came at $($i.at_minutes) min). Closing; Pilot reopens a new report if it comes back."
        }
    }
    if (-not $crashed) {
        Say "the game ended normally after $($j.Minutes) min"
        $script:Phase = 'idle'
        $script:Detail = 'last run ended without a crash'
        return
    }
    $sig = Crash-Signature $crashLines $outLines $j
    $kind = 'crash'; if ($j.Hung) { $kind = 'hang' }
    Say "$kind after $($j.Minutes) min: $sig"
    $runArgs = (@($j.Args) | Select-Object -Skip 2) -join ' '
    $runEnv = @()
    foreach ($k in $j.Env.Keys) { if ($k -ne 'CARGO_TERM_COLOR') { $runEnv += "$k=$($j.Env[$k])" } }
    $settings = 'args: `' + $runArgs + '`, env: `' + ($runEnv -join ' ') + '`'
    $existing = @(Open-Issues 'crash') | Where-Object { $_.signature -eq $sig } | Select-Object -First 1
    if ($existing) {
        if ($existing.seen -notcontains $j.Sha) {
            Comment-Issue $existing.number ("Happened again on ``$($j.Sha)`` after $($j.Minutes) min.`n`n" + (Fence $crashLines 120) + "`n`n<details><summary>Output tail</summary>`n`n" + (Fence $outLines 150) + "`n</details>")
            $existing.seen = @($existing.seen) + @($j.Sha); Save-State
        } else { Say "same crash as #$($existing.number) on the same commit: not reported again" }
    } else {
        $body = @(
            "Pilot saw a **$kind** after **$($j.Minutes) min** (exit code ``$('0x{0:x8}' -f $j.Code)``).",
            '', (Machine-Table), '', "Run: $settings", '',
            '### New crashes.log entries', (Fence $crashLines 200), '',
            '### Most frequent warnings', (Fence @(Warning-Summary $outLines) 25), '',
            '<details><summary>Output tail (last 250 lines)</summary>', '', (Fence $outLines 250), '</details>',
            '', '_Sent by VIctory Pilot. Logs only: no game data._'
        ) -join "`n"
        $n = New-Issue 'crash' "[Pilot] $($kind): $sig" $body 'crash-report'
        Add-Issue $n 'crash' $sig $sig $j.Minutes
    }
    $script:Phase = 'waiting for a fix'
    $script:Detail = "$kind reported; the next commit is built, tested and run by itself"
}

# What to do next for the current commit.
function Next-Step {
    if ($script:Job) { return }
    if ($script:Remote.pause) { $script:Phase = 'paused'; $script:Detail = 'paused by remote.json on the branch'; return }
    if ($script:State.built -ne $script:LocalSha) { [void](Start-Build); return }
    if ($script:Config.run_tests -and ($script:State.tested -ne $script:LocalSha) -and (@($script:Remote.test_crates).Count -gt 0)) { if (Start-Tests) { return } }
    if ($script:Config.auto_run -and ($script:State.ran -ne $script:LocalSha) -and $script:Config.game_exe) { if (Start-Game) { return } }
    if (-not $script:Config.game_exe) { $script:Phase = 'idle'; $script:Detail = 'built and tested; choose the game when you have it' }
    elseif ($script:Phase -ne 'waiting for a fix') { $script:Phase = 'idle'; $script:Detail = 'up to date' }
}

function Check-Remote([bool]$force = $false) {
    $script:LastRemoteCheck = Get-Date
    $sha = Remote-Sha
    if (-not $sha) { return }
    $script:RemoteSha = $sha
    if ($sha -eq $script:LocalSha) { return }
    if ($script:Job) {
        if (-not $script:PendingUpdate) { Say "new commit $(Short $sha) on the branch: applied when the current $($script:Job.Kind) ends" }
        $script:PendingUpdate = $true
        if ($script:Job.Kind -ne 'game') { Stop-Job 'a newer commit arrived' }
        return
    }
    Say "new commit $(Short $sha): updating"
    if (Update-Repo) { Next-Step }
}

# A status issue the developer can read at a glance, edited in place.
function Heartbeat {
    $script:LastHeartbeat = Get-Date
    if (-not $script:CanReport) { return }
    $job = 'none'
    if ($script:Job) { $job = "$($script:Job.Kind), $([Math]::Round(((Get-Date) - $script:Job.Started).TotalMinutes, 1)) min" }
    $open = @($script:State.issues | Where-Object { -not $_.closed } | ForEach-Object { "- #$($_.number) ($($_.kind)) $($_.title)" })
    $body = @(
        "Pilot status, updated $(Get-Date -Format 'yyyy-MM-dd HH:mm') (local time).", '',
        (Machine-Table), '',
        "| Phase | $($script:Phase): $($script:Detail) |", '|---|---|',
        "| Job | $job |",
        "| Game chosen | $([bool]$script:Config.game_exe) |",
        "| Built / tested / ran | $(Short $script:State.built) / $(Short $script:State.tested) / $(Short $script:State.ran) |", '',
        '### Open reports', $(if ($open.Count) { $open -join "`n" } else { 'none' }), '',
        '### Recent activity', (Fence ($script:Activity | Select-Object -Last 30) 30)
    ) -join "`n"
    $n = 0
    if ($script:State.ContainsKey('status_issue')) { $n = [int]$script:State.status_issue }
    if ($n -gt 0) {
        $file = Gh-Body $body
        $r = Run-Sync 'gh' @('issue', 'edit', "$n", '-R', $Repo, '--body-file', $file)
        if ($r.Code -eq 0) { return }
    }
    $n = New-Issue 'status' "[Pilot] status of $($script:Login)'s PC" $body 'pilot-status'
    if ($n -gt 0) { $script:State.status_issue = $n; Save-State }
}

# ---------------------------------------------------------------- dashboard

function Choose-Game {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $d = New-Object System.Windows.Forms.OpenFileDialog
        $d.Title = "Choose the game's executable"
        $d.Filter = 'Game executable (*.exe)|*.exe'
        $owner = New-Object System.Windows.Forms.Form
        $owner.TopMost = $true
        if ($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { return $d.FileName }
    } catch { Say "file dialog failed: $_" }
    return $null
}

function Set-Game([string]$path) {
    $path = $path.Trim().Trim('"')
    if (-not $path) { return 'no path given' }
    if (-not (Test-Path -LiteralPath $path)) { return "not found: $path" }
    if ($path.StartsWith($RepoDir, [StringComparison]::OrdinalIgnoreCase)) { return 'the game must not be inside the source folder' }
    $script:Config.game_exe = $path
    Save-Config
    Say "game set: $path"
    $script:State.ran = ''
    Save-State
    Next-Step
    return 'game set'
}

function Do-Action([string]$name, [string]$body) {
    switch ($name) {
        'run' { if ($script:Job) { return "busy: $($script:Job.Kind)" }; $script:State.ran = ''; if (Start-Game) { return 'game started' } else { return 'could not start the game (see activity)' } }
        'stop' { Stop-Job 'stopped from the dashboard'; return 'stopping' }
        'rebuild' { if ($script:Job) { return "busy: $($script:Job.Kind)" }; $script:State.built = ''; $script:State.tested = ''; Save-State; Next-Step; return 'rebuilding' }
        'test' { if ($script:Job) { return "busy: $($script:Job.Kind)" }; if (Start-Tests) { return 'testing' } else { return 'no tests configured' } }
        'check' { Check-Remote $true; return "checked: branch at $(Short $script:RemoteSha)" }
        'browse' { $p = Choose-Game; if ($p) { return (Set-Game $p) } else { return 'nothing chosen' } }
        'setgame' { return (Set-Game $body) }
        'toggle_run' { $script:Config.auto_run = -not $script:Config.auto_run; Save-Config; return "auto-run: $($script:Config.auto_run)" }
        'toggle_tests' { $script:Config.run_tests = -not $script:Config.run_tests; Save-Config; return "tests: $($script:Config.run_tests)" }
        'report' {
            $note = $body.Trim(); if (-not $note) { $note = 'Something looked wrong.' }
            if ($note.Length -gt 2000) { $note = $note.Substring(0, 2000) }
            $lines = @(); if ($script:Job) { $lines = @(Job-Lines $script:Job) } elseif ($script:LastJob) { $lines = @(Job-Lines $script:LastJob) }
            $text = "Report from the player: $note`n`n" + (Machine-Table) + "`n`n### Output tail`n" + (Fence $lines 250)
            $n = New-Issue 'note' "[Pilot] player report: $($note.Split("`n")[0])" $text 'crash-report'
            Add-Issue $n 'note' $note $note 0
            if ($n -gt 0) { return "sent as #$n" } else { return 'could not send' }
        }
        'logs' { Start-Process explorer.exe $LogDir; return 'opened' }
        default { return "unknown action $name" }
    }
}

function Status-Json {
    $tail = @()
    $j = $script:Job; if (-not $j) { $j = $script:LastJob }
    if ($j) { $tail = @(@(Job-Lines $j 65536) | Select-Object -Last 120) }
    $job = $null
    if ($script:Job) {
        $job = @{ kind = $script:Job.Kind; minutes = [Math]::Round(((Get-Date) - $script:Job.Started).TotalMinutes, 1); quiet = [Math]::Round(((Get-Date) - $script:Job.LastActivity).TotalMinutes, 1) }
    }
    $issues = @($script:State.issues | Select-Object -Last 30 | ForEach-Object { @{ number = $_.number; kind = $_.kind; title = $_.title; closed = $_.closed; url = (Issue-Url $_.number) } })
    [array]::Reverse($issues)
    $s = @{
        version = $PilotVersion; phase = $script:Phase; detail = $script:Detail
        local = $script:LocalSha; subject = $script:Subject; remote = $script:RemoteSha; pending = $script:PendingUpdate
        game = $script:Config.game_exe; auto_run = $script:Config.auto_run; run_tests = $script:Config.run_tests
        login = $script:Login; can_report = $script:CanReport; repo = $Repo; branch = $Branch
        machine = $script:Machine; job = $job; tail = $tail; activity = @($script:Activity | Select-Object -Last 80)
        issues = $issues; paused = $script:Remote.pause; profile = $script:Remote.profile
        status_issue = $script:State.status_issue; issues_on = $script:IssuesOn
    }
    return ($s | ConvertTo-Json -Depth 6 -Compress)
}

function Send-Text($ctx, [string]$text, [string]$type, [int]$code = 200) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($text)
    $ctx.Response.StatusCode = $code
    $ctx.Response.ContentType = $type
    $ctx.Response.Headers['Cache-Control'] = 'no-store'
    $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
}

function Serve($ctx) {
    $req = $ctx.Request
    if ($req.Headers['Host'] -notmatch '^(localhost|127\.0\.0\.1)(:\d+)?$') { Send-Text $ctx 'forbidden' 'text/plain' 403; return }
    $path = $req.Url.AbsolutePath
    if (($req.HttpMethod -eq 'GET') -and ($path -eq '/')) { Send-Text $ctx $DashboardHtml 'text/html; charset=utf-8'; return }
    if (($req.HttpMethod -eq 'GET') -and ($path -eq '/api/status')) { Send-Text $ctx (Status-Json) 'application/json'; return }
    if (($req.HttpMethod -eq 'POST') -and $path.StartsWith('/api/')) {
        # A custom header: other web pages cannot send it without asking first.
        if ($req.Headers['X-Pilot'] -ne '1') { Send-Text $ctx 'forbidden' 'text/plain' 403; return }
        $reader = New-Object IO.StreamReader($req.InputStream, [Text.Encoding]::UTF8)
        $body = $reader.ReadToEnd()
        $msg = Do-Action $path.Substring(5) $body
        Send-Text $ctx (@{ message = "$msg" } | ConvertTo-Json -Compress) 'application/json'
        return
    }
    Send-Text $ctx 'not found' 'text/plain' 404
}

function Handle-Http {
    while ($script:CtxTask -and $script:CtxTask.IsCompleted) {
        $ctx = $null
        try { $ctx = $script:CtxTask.Result } catch {}
        $script:CtxTask = $script:Listener.GetContextAsync()
        if (-not $ctx) { continue }
        try { Serve $ctx } catch { Say "dashboard error: $_" }
        finally { try { $ctx.Response.Close() } catch {} }
    }
}

$DashboardHtml = @'
<!doctype html>
<html><head><meta charset="utf-8"><title>VIctory Pilot</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#0e1116; --card:#161b22; --line:#2a313c; --text:#e6edf3; --dim:#8b949e; --ok:#3fb950; --warn:#d29922; --bad:#f85149; --acc:#58a6ff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 "Segoe UI", system-ui, sans-serif; }
header { display:flex; align-items:center; gap:14px; padding:16px 24px; border-bottom:1px solid var(--line); }
header h1 { font-size:18px; margin:0; letter-spacing:.3px; }
.pill { padding:3px 10px; border-radius:999px; font-weight:600; font-size:12px; background:#30363d; }
.pill.ok { background:#12361f; color:var(--ok); } .pill.warn { background:#3a2d0b; color:var(--warn); } .pill.bad { background:#3d1214; color:var(--bad); } .pill.run { background:#0c2d4d; color:var(--acc); }
main { display:grid; grid-template-columns: 380px 1fr; gap:16px; padding:16px 24px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
.card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.8px; color:var(--dim); margin:0 0 10px; }
.row { display:flex; justify-content:space-between; gap:10px; padding:4px 0; border-bottom:1px dashed #222a33; }
.row:last-child { border-bottom:0; } .row span:first-child { color:var(--dim); } .row span:last-child { text-align:right; word-break:break-all; }
button { background:#21262d; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:7px 12px; cursor:pointer; font:inherit; margin:3px 4px 3px 0; }
button:hover { border-color:var(--acc); } button.primary { background:#1f6feb; border-color:#1f6feb; } button.danger { background:#5a1d1d; border-color:#7d2a2a; }
input, textarea { width:100%; background:#0d1117; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:7px; font:inherit; }
pre { background:#0d1117; border:1px solid var(--line); border-radius:8px; padding:10px; margin:0; height:360px; overflow:auto; font:12px/1.4 Consolas, monospace; white-space:pre-wrap; word-break:break-all; }
#activity { height:220px; }
.issue { padding:5px 0; border-bottom:1px dashed #222a33; } .issue a { color:var(--acc); text-decoration:none; }
.closed { color:var(--ok); } .open { color:var(--warn); }
.banner { background:#0c2d4d; border:1px solid #1f6feb; padding:10px 14px; border-radius:8px; margin-bottom:16px; }
#msg { color:var(--acc); min-height:18px; margin-top:6px; }
.dim { color:var(--dim); }
</style></head>
<body>
<header><h1>VIctory Pilot</h1><span id="phase" class="pill">...</span><span id="detail" class="dim"></span></header>
<main>
<div>
  <div class="card"><h2>Connection</h2>
    <div class="row"><span>GitHub</span><span id="gh"></span></div>
    <div class="row"><span>Reports go to</span><span id="repo"></span></div>
    <div class="row"><span>Status issue</span><span id="statusIssue"></span></div>
    <div class="row"><span>Branch</span><span id="branch"></span></div>
    <div class="row"><span>Running commit</span><span id="local"></span></div>
    <div class="row"><span>Latest on branch</span><span id="remote"></span></div>
  </div>
  <div class="card"><h2>This PC</h2>
    <div class="row"><span>GPU</span><span id="gpu"></span></div>
    <div class="row"><span>Driver</span><span id="driver"></span></div>
    <div class="row"><span>Windows</span><span id="win"></span></div>
    <div class="row"><span>CPU / RAM</span><span id="cpu"></span></div>
  </div>
  <div class="card"><h2>Game</h2>
    <div class="row"><span>Executable</span><span id="game"></span></div>
    <button class="primary" onclick="act('browse')">Choose game...</button>
    <button onclick="act('run')">Run game</button>
    <button class="danger" onclick="act('stop')">Stop</button>
    <div style="margin-top:8px"><input id="gamePath" placeholder="or paste the full path to the game's .exe"><button onclick="act('setgame', document.getElementById('gamePath').value)">Set path</button></div>
  </div>
  <div class="card"><h2>Controls</h2>
    <button onclick="act('check')">Check for a fix now</button>
    <button onclick="act('rebuild')">Rebuild</button>
    <button onclick="act('test')">Run tests</button>
    <button onclick="act('logs')">Open logs</button><br>
    <button id="autoRun" onclick="act('toggle_run')"></button>
    <button id="runTests" onclick="act('toggle_tests')"></button>
    <div id="msg"></div>
  </div>
  <div class="card"><h2>Something looked wrong?</h2>
    <textarea id="note" rows="3" placeholder="e.g. the screen went black after the logo"></textarea>
    <button onclick="act('report', document.getElementById('note').value); document.getElementById('note').value=''">Send report</button>
  </div>
</div>
<div>
  <div id="banner" class="banner" style="display:none"></div>
  <div class="card"><h2 id="tailTitle">Output</h2><pre id="tail"></pre></div>
  <div class="card"><h2>Reports</h2><div id="issues" class="dim">none yet</div></div>
  <div class="card"><h2>Pilot activity</h2><pre id="activity"></pre></div>
</div>
</main>
<script>
function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
function short(s) { return s ? s.substring(0, 7) : '-'; }
function set(id, html) { document.getElementById(id).innerHTML = html; }
function act(name, body) {
  set('msg', 'working...');
  fetch('/api/' + name, { method: 'POST', headers: { 'X-Pilot': '1' }, body: body || '' })
    .then(function (r) { return r.json(); }).then(function (j) { set('msg', esc(j.message)); refresh(); })
    .catch(function () { set('msg', 'Pilot is not answering: is its window still open?'); });
}
function pill(phase) {
  if (/running|building|testing|updating|setting/.test(phase)) return 'pill run';
  if (/waiting|paused/.test(phase)) return 'pill warn';
  if (/idle/.test(phase)) return 'pill ok';
  return 'pill';
}
function refresh() {
  fetch('/api/status').then(function (r) { return r.json(); }).then(function (s) {
    var p = document.getElementById('phase'); p.textContent = s.phase; p.className = pill(s.phase);
    var d = s.detail || ''; if (s.job) d += '  (' + s.job.kind + ', ' + s.job.minutes + ' min, quiet ' + s.job.quiet + ' min)';
    set('detail', esc(d));
    set('gh', s.can_report ? '<span class="closed">connected as ' + esc(s.login) + '</span>' : '<span class="open">not signed in</span>');
    set('repo', '<a style="color:#58a6ff" target="_blank" href="https://github.com/' + esc(s.repo) + '/issues">' + esc(s.repo) + '</a>');
    set('statusIssue', s.status_issue ? '<a style="color:#58a6ff" target="_blank" href="https://github.com/' + esc(s.repo) + '/issues/' + s.status_issue + '">#' + s.status_issue + '</a>' : '-');
    set('branch', esc(s.branch));
    set('local', '<code>' + short(s.local) + '</code> ' + esc(s.subject));
    set('remote', '<code>' + short(s.remote) + '</code>' + (s.pending ? ' <span class="open">(update pending)</span>' : ''));
    var m = s.machine || {};
    set('gpu', esc(m.gpu)); set('driver', esc(m.driver)); set('win', esc(m.windows)); set('cpu', esc(m.cpu) + ' / ' + esc(m.ram_gb) + ' GB');
    set('game', s.game ? esc(s.game) : '<span class="open">not chosen yet</span>');
    document.getElementById('autoRun').textContent = 'Auto-run game: ' + (s.auto_run ? 'on' : 'off');
    document.getElementById('runTests').textContent = 'Run tests on each commit: ' + (s.run_tests ? 'on' : 'off');
    var b = document.getElementById('banner');
    if (s.login && !s.issues_on) { b.style.display = ''; b.innerHTML = 'Issues are switched off on <b>' + esc(s.repo) + '</b>, so reports cannot be sent. Its owner can switch them on under <a style="color:#58a6ff" target="_blank" href="https://github.com/' + esc(s.repo) + '/settings">Settings</a> &gt; General &gt; Features &gt; Issues, then restart Pilot.'; }
    else if (!s.game) { b.style.display = ''; b.innerHTML = 'No game yet: Pilot builds and tests each new commit on your PC and reports what fails. When you have the game, click <b>Choose game...</b> and it starts testing it too.'; }
    else if (s.paused) { b.style.display = ''; b.textContent = 'Paused by the developer (remote.json on the branch).'; }
    else b.style.display = 'none';
    set('tailTitle', 'Output' + (s.job ? ' (' + esc(s.job.kind) + ', live)' : ' (last job)'));
    var t = document.getElementById('tail'); var atEnd = t.scrollTop + t.clientHeight >= t.scrollHeight - 20;
    t.textContent = (s.tail || []).join('\n'); if (atEnd) t.scrollTop = t.scrollHeight;
    var a = document.getElementById('activity'); a.textContent = (s.activity || []).join('\n'); a.scrollTop = a.scrollHeight;
    var is = s.issues || [];
    set('issues', is.length ? is.map(function (i) {
      return '<div class="issue"><a target="_blank" href="' + esc(i.url) + '">#' + i.number + '</a> <span class="' + (i.closed ? 'closed">fixed' : 'open">open') + '</span> [' + esc(i.kind) + '] ' + esc(i.title) + '</div>';
    }).join('') : 'none yet');
  }).catch(function () { var p = document.getElementById('phase'); p.textContent = 'Pilot closed'; p.className = 'pill bad'; });
}
refresh(); setInterval(refresh, 2000);
</script>
</body></html>
'@

# ---------------------------------------------------------------- main

Say "VIctory Pilot $PilotVersion"
if (-not (Ensure-Tools)) { Read-Host 'Press Enter to close'; exit 1 }
$script:CanReport = Ensure-GitHub
$script:IssuesOn = $false
if ($script:CanReport) { $script:IssuesOn = Ensure-Issues; $script:CanReport = $script:IssuesOn }
$script:Machine = Read-Machine
Say "this PC: $($script:Machine.gpu), driver $($script:Machine.driver), $($script:Machine.windows)"

$script:Listener = $null
$Port = 0
foreach ($p in 8765..8775) {
    try {
        $l = New-Object System.Net.HttpListener
        $l.Prefixes.Add("http://localhost:$p/")
        $l.Start()
        $script:Listener = $l
        $Port = $p
        break
    } catch {}
}
if ($script:Listener) {
    $script:CtxTask = $script:Listener.GetContextAsync()
    Say "dashboard: http://localhost:$Port/ (keep this window open)"
    Start-Process "http://localhost:$Port/"
} else { Say 'could not open the dashboard port: Pilot keeps working in this window' }

if (Update-Repo) { Next-Step }
Heartbeat

while ($true) {
    Handle-Http
    Tick-Job
    $now = Get-Date
    if (-not $script:Job -and $script:SelfUpdate) {
        Say 'restarting into the newer Pilot'
        try { Copy-Item -LiteralPath $script:SelfUpdate -Destination $PSCommandPath -Force } catch { Say "could not update Pilot: $_" }
        if ($script:Listener) { $script:Listener.Stop() }
        exit 3
    }
    if (-not $script:Job -and $script:PendingUpdate) { if (Update-Repo) { Next-Step } }
    if (($now - $script:LastRemoteCheck).TotalSeconds -ge 60) { Check-Remote }
    if (($now - $script:LastHeartbeat).TotalMinutes -ge 10) { Heartbeat }
    Start-Sleep -Milliseconds 250
}
