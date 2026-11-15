# Verify the transfer folder against SHA256SUMS (written on the Linux box).
# Usage:  powershell -ExecutionPolicy Bypass -File verify.ps1 [-Dir <gm_transfer folder>]
param([string]$Dir = $PSScriptRoot)
$bad = 0
Get-Content -LiteralPath (Join-Path $Dir 'SHA256SUMS') | ForEach-Object {
    if ($_ -match '^([0-9a-f]{64})\s+\*?(.+)$') {
        $expected = $Matches[1]
        $rel = $Matches[2]
        $path = Join-Path $Dir ($rel -replace '/', '\')
        if (-not (Test-Path -LiteralPath $path)) { Write-Host "MISSING $rel"; $script:bad++; return }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
        if ($actual -ne $expected) { Write-Host "BAD     $rel"; $script:bad++ } else { Write-Host "OK      $rel" }
    }
}
if ($bad -gt 0) { throw "$bad file(s) failed verification - copy them again" }
Write-Host 'All files verified.'
