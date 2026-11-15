# Watch a training run without touching it (read-only): one row per finished epoch, an estimate for the epoch in
# progress, time left, GPU use, and whether the training process is still alive.  Ctrl+C stops watching only.
#   .\scripts\windows\watch_training.ps1                 # refresh every 60 s
#   .\scripts\windows\watch_training.ps1 -Every 30
#   .\scripts\windows\watch_training.ps1 -Once           # print once and exit
#   .\scripts\windows\watch_training.ps1 -Run <name>     # another run under runs\seg\
param(
    [string]$Run = 'y26n_seg_india_laptop',
    [int]$Every = 60,
    [switch]$Once
)
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Dir = Join-Path $Root "runs\seg\$Run"
$Csv = Join-Path $Dir 'results.csv'
$TimingFile = Join-Path $Dir 'gm_timing.json'
$RunFile = Join-Path $Dir 'gm_run.json'
$Inv = [Globalization.CultureInfo]::InvariantCulture

function ToNum([string]$s) {
    if ($null -eq $s -or $s.Trim() -eq '') { return [double]::NaN }
    return [double]::Parse($s.Trim(), $Inv)
}

function ReadJson([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    try { return (Get-Content $path -Raw -ErrorAction Stop | ConvertFrom-Json) } catch { return $null }
}

function ShowHealth($timing) {
    # per-epoch health recorded by ml.train_seg (epochs trained before this check existed show nothing)
    if (-not $timing -or -not $timing.epochs) { return }
    $withHealth = @($timing.epochs | Where-Object { $_.health })
    if ($withHealth.Count -eq 0) { Write-Host 'Health checks: none recorded yet (they start with the next epoch).'; return }
    Write-Host 'Health checks per epoch:' -ForegroundColor Cyan
    foreach ($e in ($withHealth | Select-Object -Last 8)) {
        $h = $e.health
        $info = @("{0:N1} min" -f (($e.train_s + $e.val_s) / 60))
        if ($null -ne $h.s_per_iter_median) { $info += "$($h.s_per_iter_median) s/it" }
        if ($null -ne $h.peak_vram_gb) { $info += ("GPU mem {0:N2}/{1} GB" -f $h.peak_vram_gb, $h.gpu_total_gb) }
        if ($null -ne $h.gpu_temp_max_c) { $info += ("GPU max {0:N0} C" -f $h.gpu_temp_max_c) }
        if ($null -ne $h.gpu_util_mean_pct) { $info += ("GPU busy {0:N0} %" -f $h.gpu_util_mean_pct) }
        if ($null -ne $h.other_cpu_mean_pct) { $info += ("other programs CPU {0:N0} %" -f $h.other_cpu_mean_pct) }
        $issues = @($h.issues)
        $color = if ($issues.Count) { 'Yellow' } else { 'Green' }
        $tag = if ($issues.Count) { "$($issues.Count) issue(s)" } else { 'OK' }
        Write-Host ("  epoch {0,2}: {1,-12} {2}" -f $e.epoch, $tag, ($info -join ', ')) -ForegroundColor $color
        foreach ($i in $issues) { Write-Host "             - $i" -ForegroundColor Yellow }
    }
}

function ShowLogWarnings {
    $log = Join-Path $Dir 'train_log.txt'
    if (-not (Test-Path $log)) { return }
    $warn = @(Get-Content $log -Tail 400 -ErrorAction SilentlyContinue |
        Where-Object { $_ -match ' (WARNING|ERROR|CRITICAL) ' -and $_ -notmatch '\[gm\] epoch \d+ ISSUE' })
    if ($warn.Count) {
        Write-Host 'Recent warnings in train_log.txt:' -ForegroundColor Yellow
        $warn | Select-Object -Last 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    }
}

function ShowLiveChecks {
    $b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($b -and $b.BatteryStatus -eq 1) {
        Write-Host ("NOW: laptop is on battery ({0} %) - plug in the charger." -f $b.EstimatedChargeRemaining) -ForegroundColor Yellow
    }
}

function ReadEpochs {
    # results.csv is rewritten by the trainer at the end of every epoch; a read can collide with that write
    if (-not (Test-Path $Csv)) { return @() }
    try { $lines = Get-Content $Csv -ErrorAction Stop } catch { return @() }
    if ($lines.Count -lt 2) { return @() }
    $head = $lines[0].Split(',') | ForEach-Object { $_.Trim() }
    $rows = @()
    $prevT = 0.0
    foreach ($ln in $lines[1..($lines.Count - 1)]) {
        if ($ln.Trim() -eq '') { continue }
        $v = $ln.Split(',')
        $r = @{}
        for ($i = 0; $i -lt $head.Count -and $i -lt $v.Count; $i++) { $r[$head[$i]] = ToNum $v[$i] }
        $t = $r['time']          # cumulative seconds since the run started
        $mins = if ($t -ge $prevT) { ($t - $prevT) / 60 } else { $t / 60 }   # a resume restarts the clock
        $prevT = $t
        $rows += [pscustomobject]@{
            Epoch        = [int]$r['epoch']
            Minutes      = [math]::Round($mins, 1)
            'Box P'      = [math]::Round($r['metrics/precision(B)'], 3)
            'Box R'      = [math]::Round($r['metrics/recall(B)'], 3)
            'Box mAP50'  = [math]::Round($r['metrics/mAP50(B)'], 4)
            'Mask mAP50' = [math]::Round($r['metrics/mAP50(M)'], 4)
            'Mask 50-95' = [math]::Round($r['metrics/mAP50-95(M)'], 4)
            'Val cls'    = [math]::Round($r['val/cls_loss'], 3)
            Best         = ''
        }
    }
    if ($rows.Count -gt 0) {
        $best = ($rows | Sort-Object 'Mask mAP50' -Descending | Select-Object -First 1).Epoch
        foreach ($x in $rows) { if ($x.Epoch -eq $best) { $x.Best = '<-- best' } }
    }
    return $rows
}

function TrainerProcess {
    # the training process (not the .venv launcher stub, not the data-loader workers); $null when not running
    return Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'ml\.train_seg' -and $_.CommandLine -match [regex]::Escape($Run) } |
        Sort-Object CreationDate | Select-Object -First 1
}

function Show {
    $now = Get-Date
    Write-Host ("=== Training run '{0}'   {1:dd-MM-yyyy HH:mm:ss} ===" -f $Run, $now) -ForegroundColor Cyan
    if (-not (Test-Path $Dir)) { Write-Host "No run folder $Dir yet." -ForegroundColor Yellow; return }

    $runInfo = ReadJson $RunFile
    $timing = ReadJson $TimingFile
    $epochsTotal = if ($timing -and $timing.epochs_total) { [int]$timing.epochs_total }
                   elseif ($runInfo) { [int]$runInfo.train_kwargs.epochs } else { 0 }
    if ($runInfo -and $runInfo.epochs_changed) {   # 'train-max -Epochs N' after the last timing update
        $chg = @($runInfo.epochs_changed)[-1]
        if (-not $timing -or -not $timing.updated_at -or [string]$chg.at -gt [string]$timing.updated_at) {
            $epochsTotal = [int]$chg.to
        }
    }
    $proc = TrainerProcess
    $alive = [bool]$proc
    $procStart = if ($proc) { $proc.CreationDate } else { $null }
    $sessionH = 0.0   # only the RUNNING session's own limit counts
    if ($proc -and $proc.CommandLine -match '--session-hours\s+([\d.]+)') { $sessionH = [double]::Parse($Matches[1], $Inv) }

    $rows = @(ReadEpochs)
    $done = $rows.Count
    $state = if ($alive) { 'RUNNING' } else { 'NOT RUNNING' }
    Write-Host ("Training process: {0}    epochs done: {1} / {2}" -f $state, $done, $epochsTotal) `
        -ForegroundColor $(if ($alive) { 'Green' } else { 'Yellow' })

    if ($done -gt 0) {
        $rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
    } else {
        Write-Host 'No finished epoch yet (the first one takes about 25 min on the MX330).'
    }

    # estimate for the epoch in progress: time since the last epoch ended vs the mean epoch time so far
    if ($alive -and $done -lt $epochsTotal) {
        $recent = @($rows | Select-Object -Last 5)
        $avgMin = if ($recent.Count) { ($recent | Measure-Object Minutes -Average).Average } else { 25.0 }
        $since = if ($done -gt 0) { (Get-Item $Csv).LastWriteTime }
                 elseif ($runInfo) { [datetime]::Parse($runInfo.created_at, $Inv).ToLocalTime() } else { $now }
        if ($procStart -and $procStart -gt $since) { $since = $procStart.AddMinutes(2) }   # resumed: ~2 min start-up
        $elapsed = [math]::Max(0, ($now - $since).TotalMinutes)
        $pct = [math]::Min(99, [math]::Round(100 * $elapsed / [math]::Max($avgMin, 1)))
        Write-Host ("Epoch {0} in progress: {1:N0} min elapsed, about {2} % (estimate, {3:N1} min per epoch)" -f `
            ($done + 1), $elapsed, $pct, $avgMin)
        $left = ($epochsTotal - $done) * $avgMin - [math]::Min($elapsed, $avgMin)
        Write-Host ("All {0} epochs: about {1:N1} h left -> around {2:dd-MM HH:mm} (if patience does not stop it earlier)" -f `
            $epochsTotal, ($left / 60), $now.AddMinutes($left))
        if ($sessionH -gt 0 -and $procStart) {
            $end = $procStart.AddHours($sessionH)
            Write-Host ("This session stops itself at {0:dd-MM HH:mm} ({1} h cap); run 'tasks.ps1 train-max' again to continue." -f `
                $end, $sessionH)
        }
    } elseif (-not $alive -and $done -gt 0 -and $done -lt $epochsTotal) {
        Write-Host "Stopped after epoch $done. Continue with:  .\scripts\windows\tasks.ps1 train-max" -ForegroundColor Yellow
    } elseif ($done -ge $epochsTotal -and $epochsTotal -gt 0) {
        Write-Host 'All epochs finished. Next: .\scripts\windows\tasks.ps1 eval' -ForegroundColor Green
    }

    Write-Host ''
    ShowHealth $timing
    ShowLogWarnings
    if ($alive) { ShowLiveChecks }
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        $g = $gpu.Split(',') | ForEach-Object { $_.Trim() }
        Write-Host ("GPU: {0} % busy, {1} / {2} MiB, {3} C, clock {4} of {5} MHz" -f $g[0], $g[1], $g[2], $g[3], $g[4], $g[5])
        $reasons = 0L
        if ($g.Count -ge 7 -and $g[6] -match '^0x([0-9a-fA-F]+)$') { $reasons = [Convert]::ToInt64($Matches[1], 16) }
        if ($alive -and ($reasons -band 0x60)) {   # 0x20 software thermal slowdown, 0x40 hardware thermal slowdown
            Write-Host 'NOW: the GPU clock is being lowered because of heat - raise the back of the laptop, keep vents clear.' -ForegroundColor Yellow
        }
        if ($alive -and [int]$g[3] -ge 85) { Write-Host 'NOW: GPU is hot - keep the vents clear / raise the laptop.' -ForegroundColor Yellow }
    }
    $w = Join-Path $Dir 'weights'
    if (Test-Path $w) {
        $last = Get-ChildItem $w -Filter 'epoch*.pt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($last) { Write-Host ("Last checkpoint: {0} ({1:HH:mm})" -f $last.Name, $last.LastWriteTime) }
    }
}

if ($Once) { Show; exit 0 }
while ($true) {
    Clear-Host
    Show
    Write-Host ("`nRefreshing every {0} s - Ctrl+C stops watching (training keeps running)." -f $Every) -ForegroundColor DarkGray
    Start-Sleep -Seconds $Every
}
