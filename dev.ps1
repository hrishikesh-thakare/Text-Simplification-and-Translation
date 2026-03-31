$ErrorActionPreference = "Stop"

$watchfiles = Join-Path $PSScriptRoot ".venv\Scripts\watchfiles.exe"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $watchfiles)) {
    Write-Host "watchfiles not found in .venv. Installing..." -ForegroundColor Yellow
    & $python -m pip install watchfiles
}

Write-Host "Starting auto-reload dev mode for main.py..." -ForegroundColor Cyan
Write-Host "Any .py file change will restart the app." -ForegroundColor DarkGray

& $watchfiles --filter python "$python main.py" $PSScriptRoot
