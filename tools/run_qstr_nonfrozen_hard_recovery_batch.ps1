param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [Parameter(Mandatory = $true)]
    [string]$OutRoot,
    [int]$MaxSequences = 0,
    [int]$MaxFrames = 60,
    [string[]]$ExcludeSequences = @(),
    [switch]$SkipExisting,
    [string]$Device = "0",
    [string]$TrackletClassifierWeights = "runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v2_hardtiny_aug.pt"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path $Manifest)) {
    throw "Missing manifest: $Manifest"
}

$rows = Import-Csv $Manifest
$selected = New-Object System.Collections.Generic.List[object]
foreach ($row in $rows) {
    if (-not $row.video_path) {
        continue
    }
    $seq = Split-Path (Split-Path $row.video_path -Parent) -Leaf
    if ($ExcludeSequences -contains $seq) {
        continue
    }
    if (-not (Test-Path $row.video_path)) {
        Write-Warning "Skipping missing video: $($row.video_path)"
        continue
    }
    $selected.Add([pscustomobject]@{
        Seq = $seq
        Video = $row.video_path
        Scenario = $row.scenario
    })
    if ($MaxSequences -gt 0 -and $selected.Count -ge $MaxSequences) {
        break
    }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$summary = @()
foreach ($item in $selected) {
    $out = Join-Path (Join-Path $OutRoot "hard_recovery") $item.Seq
    $pred = Join-Path $out "predictions.jsonl"
    if ($SkipExisting -and (Test-Path $pred)) {
        Write-Host "=== Skip existing $($item.Seq) ==="
        $summary += [pscustomobject]@{ seq = $item.Seq; scenario = $item.Scenario; status = "skipped"; out = $out }
        continue
    }
    Write-Host "=== Run $($item.Seq) [$($item.Scenario)] ==="
    $args = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "run_qstr_hard_recovery_profile.ps1"),
        "-Video", $item.Video,
        "-Out", $out,
        "-Device", $Device,
        "-MaxFrames", "$MaxFrames",
        "-DisableTrackletPromotion"
    )
    if ($TrackletClassifierWeights -eq "") {
        $args += @("-TrackletClassifierWeights", "")
    } else {
        $args += @("-TrackletClassifierWeights", $TrackletClassifierWeights)
    }
    powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Profile failed for $($item.Seq) with exit code $LASTEXITCODE"
    }
    $summary += [pscustomobject]@{ seq = $item.Seq; scenario = $item.Scenario; status = "done"; out = $out }
}

$summaryPath = Join-Path $OutRoot "batch_summary.csv"
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 $summaryPath
Write-Host "Wrote batch summary: $summaryPath"
