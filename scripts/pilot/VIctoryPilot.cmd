@echo off
rem VIctory Pilot: double-click to start. Gets the latest Pilot from the
rem branch and runs it; the first start installs what is missing.
setlocal
set "PILOT_HOME=%LOCALAPPDATA%\VIctoryPilot"
set "PILOT_URL=https://raw.githubusercontent.com/MfDev0/victory/refs/heads/arena/59fabb00-victory/scripts/pilot/pilot.ps1"
if not exist "%PILOT_HOME%" mkdir "%PILOT_HOME%"
title VIctory Pilot
echo Getting the latest VIctory Pilot...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $new = Join-Path $env:PILOT_HOME 'pilot.new.ps1'; Invoke-WebRequest -UseBasicParsing -Uri $env:PILOT_URL -OutFile $new; Move-Item -Force $new (Join-Path $env:PILOT_HOME 'pilot.ps1') } catch { Write-Host ('Download failed, using the last copy: ' + $_) }"
if not exist "%PILOT_HOME%\pilot.ps1" (
  echo Could not download Pilot. Check the internet connection and try again.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "do { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $env:PILOT_HOME 'pilot.ps1'); $code = $LASTEXITCODE } while ($code -eq 3); exit $code"
echo.
echo VIctory Pilot stopped. Start it again with this file.
pause
