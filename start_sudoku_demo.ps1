param(
    [int]$Port = 3010
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Stop-Process -Id $listener.OwningProcess -Force
    Start-Sleep -Milliseconds 300
}

$env:SUDOKU_DEMO_PORT = "$Port"
python .\sudoku_demo_app.py
