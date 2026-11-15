# Native-Windows stand-in for the Makefile.  Usage:  .\scripts\windows\tasks.ps1 <target> [extra args]
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(ValueFromRemainingArguments = $true)]$Rest
)
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$PY = Join-Path $Root '.venv\Scripts\python.exe'
$env:PYTHONPATH = $Root
$env:PYTHONUTF8 = '1'
if ($null -eq $Rest) { $Rest = @() }

switch ($Target) {
    'setup'        { & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'setup.ps1') @Rest }
    'test'         { & $PY -m pytest -m 'not gpu and not data' @Rest }
    'test-fast'    { & $PY -m pytest -x -m 'not gpu and not data and not slow' @Rest }
    'lint'         { & $PY -m ruff check . }
    'lock'         { & $PY scripts\write_lockfile.py }
    'db'           { & $PY -m server.app.db.init }
    'seed'         { & $PY scripts\seed_demo.py @Rest }
    'api'          { & $PY -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000 --reload }
    'worker'       { & $PY -m server.workers.run @Rest }
    'ops-tick'     { & $PY -m server.workers.run --tick @Rest }
    'tick'         { & $PY -m server.workers.run --tick @Rest }
    'web'          { Push-Location web; try { npm run dev } finally { Pop-Location } }
    'web-build'    { Push-Location web; try { npm run build } finally { Pop-Location } }
    'web-test'     { Push-Location web; try { npm test } finally { Pop-Location } }
    'measure-demo' { & $PY -m measure.cli demo --out reports\measure_demo }
    'dataset'      { & $PY data\build_dataset.py --out data\datasets\india_pothole_seg --sam2 --negatives-frac 0.2 @Rest }
    'derive-dataset' { & $PY data\derive_dataset.py @Rest }   # 4-class training set + manhole re-split (D15)
    'geo'          { & $PY -m geo.build_geo @Rest }
    'train'        { & $PY -m ml.train_seg --profile laptop_mx330 --name y26n_seg_india_laptop --session-hours 8 @Rest }
    'train-max'    { & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'train_full_gpu.ps1') @Rest }
    'classes'      { & $PY scripts\dataset_classes.py @Rest }
    'watch-train'  { & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'watch_training.ps1') @Rest }
    'train-resume' { & $PY -m ml.train_seg --name y26n_seg_india_laptop --resume --session-hours 8 @Rest }
    'eval'         { & $PY -m ml.eval @Rest }
    'export'       { & $PY -m ml.export @Rest }
    'pseudo-label' { & $PY -m ml.pseudo_label @Rest }
    'sim-api'      { & $PY edge\sim\sim_server.py --copy-from data\gunthameter.db @Rest }
    'fleet'        { & $PY edge\sim\fleet.py --api http://127.0.0.1:8002 @Rest }
    'demo'         { & $PY scripts\demo.py @Rest }
    'truth'        { & $PY scripts\staged_pothole_truth.py @Rest }
    'report'       { & $PY scripts\make_report.py @Rest }
    default {
        Write-Host 'targets: setup test test-fast lint lock db seed api worker ops-tick tick web web-build web-test measure-demo dataset derive-dataset geo classes train train-max watch-train train-resume eval export pseudo-label sim-api fleet demo truth report'
        exit 1
    }
}
exit $LASTEXITCODE
