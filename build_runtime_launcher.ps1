Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot 'venv\Scripts\python.exe'

# Prevent stale runtime processes from locking dist files during cleanup.
Get-Process -Name 'TextSimplifierRuntime' -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name 'python' -ErrorAction SilentlyContinue | Stop-Process -Force

if (-not (Test-Path $venvPython)) {
    throw "Project venv Python not found at $venvPython"
}

if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist) { Remove-Item dist -Recurse -Force }

& $venvPython -m PyInstaller launcher.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host ''
Write-Host 'Build complete.' -ForegroundColor Green
Write-Host 'The runtime-only launcher is in dist\TextSimplifierRuntime\TextSimplifierRuntime.exe' -ForegroundColor Green