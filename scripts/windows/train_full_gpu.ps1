# Train the pothole segmenter on this laptop's GPU, as busy as it can be kept, in the FOREGROUND (this window).
#
#   cd C:\gunthameter
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_full_gpu.ps1                 # start or continue, no time limit
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_full_gpu.ps1 -Epochs 20      # change the run's total epochs
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_full_gpu.ps1 -SessionHours 8 # pause cleanly after 8 h
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_full_gpu.ps1 -NoCache        # if RAM is short
#
# What it does:
#   1. shows the GPU and the dataset classes;
#   2. starts a new run with profile laptop_mx330_max (YOLO26n-seg, 640 px, fp32, batch = 90 % of free GPU memory via
#      Ultralytics AutoBatch, 4 loader workers, images cached in RAM), or RESUMES it if runs\seg\<Name>\weights\last.pt
#      exists - so you can run the same command again after a pause, a reboot or Ctrl-C;
#   3. trains to the end (30 epochs, or -Epochs N; early stop after 10 epochs without improvement), or until the
#      optional -SessionHours limit (it then stops cleanly after the last epoch that fits);
#   4. when training is complete, prints the evaluate / export / switch-over commands.
# Ctrl-C stops it; the last finished epoch is kept and the next run continues from there.
param(
    [string]$Name = 'y26n_seg_india_laptop',
    [double]$SessionHours = 0,
    [int]$Epochs = 0,
    [switch]$NoCache
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$PY = Join-Path $Root '.venv\Scripts\python.exe'
$env:PYTHONPATH = $Root
$env:PYTHONUTF8 = '1'
if (-not (Test-Path $PY)) { throw "Python venv not found at $PY (run scripts\windows\setup.ps1 first)" }

Write-Host '=== GPU ===' -ForegroundColor Cyan
& nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,driver_version --format=csv
Write-Host 'Tips: keep the charger connected; close the browser and other GPU apps; set Windows power mode to'
Write-Host '"Best performance" and in the NVIDIA Control Panel set Power management mode to "Prefer maximum performance".'

Write-Host "`n=== Dataset classes ===" -ForegroundColor Cyan
& $PY scripts\dataset_classes.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$last = Join-Path $Root "runs\seg\$Name\weights\last.pt"
$session = @()
if ($SessionHours -gt 0) { $session = @('--session-hours', "$SessionHours") }
if ($Epochs -gt 0) { $session += @('--epochs', "$Epochs") }
$cacheArg = @()
if ($NoCache) { $cacheArg = @('--set', 'cache=false') }

if (Test-Path $last) {
    Write-Host "`n=== Resuming run '$Name' from $last ===" -ForegroundColor Cyan
    $trainArgs = @('-m', 'ml.train_seg', '--name', $Name, '--resume') + $session
} else {
    Write-Host "`n=== New run '$Name' (profile laptop_mx330_max) ===" -ForegroundColor Cyan
    $trainArgs = @('-m', 'ml.train_seg', '--profile', 'laptop_mx330_max', '--name', $Name) + $session + $cacheArg
}
Write-Host ("$PY " + ($trainArgs -join ' '))
Write-Host 'Watch the GPU in a second window with:  nvidia-smi -l 5' -ForegroundColor DarkGray
$t0 = Get-Date
& $PY @trainArgs
$code = $LASTEXITCODE
$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host "`ntrain_seg exited with code $code after $mins min" -ForegroundColor Cyan

$best = Join-Path $Root "runs\seg\$Name\weights\best.pt"
$csv = Join-Path $Root "runs\seg\$Name\results.csv"
if (Test-Path $csv) {
    Write-Host "`n=== Last epochs (runs\seg\$Name\results.csv) ===" -ForegroundColor Cyan
    Import-Csv $csv | Select-Object -Last 3 | ForEach-Object {
        $e = $_.PSObject.Properties
        $ep = ($e | Where-Object { $_.Name.Trim() -eq 'epoch' }).Value
        $m50 = ($e | Where-Object { $_.Name.Trim() -eq 'metrics/mAP50(B)' }).Value
        $mm50 = ($e | Where-Object { $_.Name.Trim() -eq 'metrics/mAP50(M)' }).Value
        Write-Host ("  epoch {0}: box mAP50 {1}  mask mAP50 {2}" -f $ep, $m50, $mm50)
    }
}
if ($code -eq 0 -and (Test-Path $best)) {
    Write-Host "`nIf training is not finished yet, run this script again to continue." -ForegroundColor Yellow
    Write-Host 'When all epochs are done, evaluate on the test split and switch the system to the new model:'
    Write-Host "  .\scripts\windows\tasks.ps1 eval --weights runs\seg\$Name\weights\best.pt --stage trained_$Name"
    Write-Host "  .\scripts\windows\tasks.ps1 export --weights runs\seg\$Name\weights\best.pt --formats onnx --register"
    Write-Host '  .\scripts\windows\tasks.ps1 report'
    Write-Host "  then in .env:  SEG_WEIGHTS=runs/seg/$Name/weights/best.pt   (only if it beats 0.0885 box mAP50)"
}
exit $code
