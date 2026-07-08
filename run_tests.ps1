$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = "D:\Anaconda3\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Write-Host "Running platform contract tests..." -ForegroundColor Cyan
& $python -m unittest tests.test_platform_contracts -v
