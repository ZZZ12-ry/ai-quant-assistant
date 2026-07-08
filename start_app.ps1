$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = "D:\Anaconda3\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$port = 8015
$lanIp = (Get-NetIPConfiguration |
    Where-Object { $_.IPv4DefaultGateway -and $_.IPv4Address.IPAddress -notlike "169.254*" } |
    Select-Object -First 1 -ExpandProperty IPv4Address).IPAddress

Write-Host "Starting AI quant app..." -ForegroundColor Cyan
Write-Host "Local: http://127.0.0.1:$port" -ForegroundColor Cyan
if ($lanIp) {
    Write-Host "LAN:   http://$lanIp`:$port" -ForegroundColor Green
    Write-Host "Colleagues on the same network can open the LAN URL if Windows Firewall allows port $port." -ForegroundColor Yellow
}

& $python -X utf8 -m uvicorn app.main:app --host 0.0.0.0 --port $port
