# Pull the whole project from the Linux box onto this Windows laptop (native Windows, PowerShell).
# Usage:  powershell -ExecutionPolicy Bypass -File pull.ps1 [-Server 192.168.0.121] [-User coral] [-Dest C:\gunthameter]
param(
    [string]$Server = '192.168.0.121',
    [string]$User = 'coral',
    [string]$Dest = 'C:\gunthameter'
)
$ErrorActionPreference = 'Stop'
function Exec([scriptblock]$Block) {
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Block" }
}
foreach ($tool in 'scp', 'tar') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not found. Settings > System > Optional features > add 'OpenSSH Client' (scp); tar ships with Windows 10 1803+."
    }
}

New-Item -ItemType Directory -Force $Dest | Out-Null
Set-Location $Dest

# 1. copy (~17 GB). Rerun safely: an existing gm_transfer folder is reused.
if (-not (Test-Path '.\gm_transfer\SHA256SUMS')) {
    Write-Host "Copying from $User@$Server (you will be asked for the password) ..."
    Exec { scp -r "${User}@${Server}:gm_transfer" . }
} else {
    Write-Host 'gm_transfer already present - verifying it (delete the folder to force a fresh copy).'
}

# 2. verify checksums
& (Join-Path $Dest 'gm_transfer\verify.ps1') -Dir (Join-Path $Dest 'gm_transfer')

# 3. unpack code + processed data, move raw downloads into data\raw (move, not copy: saves 13 GB)
Exec { tar -xzf .\gm_transfer\gunthameter_code.tar.gz }
Exec { tar -xf .\gm_transfer\gunthameter_data_processed.tar }
New-Item -ItemType Directory -Force 'data\raw' | Out-Null
Get-ChildItem '.\gm_transfer\data_raw' | ForEach-Object { Move-Item $_.FullName 'data\raw\' -Force }

Write-Host ''
Write-Host "Done. Project is in $Dest. Next:"
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1      (add -Cpu if no NVIDIA GPU)'
Write-Host '  .\scripts\windows\tasks.ps1 test'
Write-Host '  then read docs\HANDOFF.md section 2. You can delete .\gm_transfer afterwards.'
