# Native-Windows equivalent of `make setup` (see docs/HANDOFF.md section 4, option B).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 [-Cpu] [-Cuda cu126|cu128]
# cu126 = Pascal/Volta GPUs too (e.g. the laptop's MX330, sm_61); cu128 = Turing+ only (sm_75+). DECISIONS D14.
param([switch]$Cpu, [string]$Cuda = 'cu126')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

function Exec([scriptblock]$Block) {
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Block" }
}

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) { Exec { py -3.11 -m venv .venv } }
    else { Exec { python -m venv .venv } }
}
$PY = Join-Path $Root '.venv\Scripts\python.exe'
& $PY -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
if ($LASTEXITCODE -ne 0) { throw 'The .venv is not Python 3.11. Install Python 3.11 (winget install -e --id Python.Python.3.11), delete .venv, rerun.' }

$env:PYTHONUTF8 = '1'
[Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'User')

Exec { & $PY -m pip install -U pip }
$Index = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { "https://download.pytorch.org/whl/$Cuda" }
Exec { & $PY -m pip install torch==2.11.0 torchvision==0.26.0 --index-url $Index }
Exec { & $PY -m pip install -r requirements-lock.txt }
Exec { & $PY -m pip install -e . --no-deps }

if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
New-Item -ItemType Directory -Force 'data\store', 'data\interim', 'data\datasets', 'reports', 'runs' | Out-Null

Exec { & $PY scripts\write_lockfile.py }
Exec { & $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" }

if (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location web
    try { Exec { npm ci } } finally { Pop-Location }
} else {
    Write-Warning 'npm not found - install Node 22 (winget install -e --id OpenJS.NodeJS.LTS), then run: cd web; npm ci'
}
Write-Host 'setup ok - next: .\scripts\windows\tasks.ps1 test'
