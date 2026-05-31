param(
    [string]$Out = "D:\datasets\my_video\full_infer_compare\dji_stage_b_source_scene_profile_20260530",
    [string]$ProfileName = "source_scene_stageb_select",
    [int]$Tail = 50
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutAbs = if ([System.IO.Path]::IsPathRooted($Out)) {
    [System.IO.Path]::GetFullPath($Out)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $Root $Out))
}
if (-not (Test-Path -LiteralPath $OutAbs)) {
    throw "Missing output directory: $OutAbs"
}

$PidFile = Join-Path $OutAbs "pid.txt"
$Stdout = Join-Path $OutAbs "stdout.log"
$Stderr = Join-Path $OutAbs "stderr.log"
$Status = Join-Path $OutAbs "status.txt"
$Done = Join-Path $OutAbs ".done"
$Failed = Join-Path $OutAbs ".failed"

$PidValue = $null
if (Test-Path -LiteralPath $PidFile) {
    $PidValue = (Get-Content -LiteralPath $PidFile -TotalCount 1).Trim()
}
$Running = $false
if ($PidValue) {
    $proc = Get-Process -Id ([int]$PidValue) -ErrorAction SilentlyContinue
    $Running = $null -ne $proc
}

Write-Host "Out:     $OutAbs"
Write-Host "PID:     $(if ($PidValue) { $PidValue } else { 'missing' })"
Write-Host "Running: $Running"
Write-Host "Done:    $(Test-Path -LiteralPath $Done)"
Write-Host "Failed:  $(Test-Path -LiteralPath $Failed)"
Write-Host "Updated: $((Get-Item -LiteralPath $OutAbs).LastWriteTime)"
if (Test-Path -LiteralPath $Status) {
    Write-Host "Status:  $((Get-Content -LiteralPath $Status -Tail 1))"
}

function Show-QstrProfileProgress {
    param([string]$Label, [string]$ProfileRoot, [string]$InnerProfile)
    if (-not (Test-Path -LiteralPath $ProfileRoot)) {
        Write-Host "$Label pending"
        return
    }
    $summary = Join-Path $ProfileRoot "summary.csv"
    if (Test-Path -LiteralPath $summary) {
        Write-Host ""
        Write-Host "=== $Label summary.csv ==="
        Get-Content -LiteralPath $summary
    }
    $inner = Join-Path $ProfileRoot $InnerProfile
    if (-not (Test-Path -LiteralPath $inner)) {
        Write-Host "$Label/$InnerProfile pending"
        return
    }
    Get-ChildItem -LiteralPath $inner -Directory | Sort-Object Name | ForEach-Object {
        $pred = Join-Path $_.FullName "predictions.jsonl"
        $diag = Join-Path $_.FullName "diagnostics.jsonl"
        $meta = Join-Path $_.FullName "run_meta.json"
        $expected = "?"
        $expectedNumber = $null
        $frameStride = 1
        if (Test-Path -LiteralPath $meta) {
            try {
                $metaJson = Get-Content -LiteralPath $meta -Raw | ConvertFrom-Json
                $expected = $metaJson.evaluated_frames
                $expectedNumber = [int]$expected
                if ($null -ne $metaJson.frame_stride) {
                    $frameStride = [math]::Max(1, [int]$metaJson.frame_stride)
                }
            } catch {
                $expected = "?"
            }
        }
        $predLines = if (Test-Path -LiteralPath $pred) { (Get-Content -LiteralPath $pred -ErrorAction SilentlyContinue | Measure-Object -Line).Lines } else { 0 }
        $diagLines = if (Test-Path -LiteralPath $diag) { (Get-Content -LiteralPath $diag -ErrorAction SilentlyContinue | Measure-Object -Line).Lines } else { 0 }
        $lastFrame = $null
        foreach ($jsonl in @($pred, $diag)) {
            if (-not (Test-Path -LiteralPath $jsonl)) {
                continue
            }
            $lastLine = Get-Content -LiteralPath $jsonl -Tail 1 -ErrorAction SilentlyContinue
            if (-not $lastLine) {
                continue
            }
            try {
                $row = $lastLine | ConvertFrom-Json
                if ($null -ne $row.frame_id) {
                    $frame = [int]$row.frame_id
                    if ($null -eq $lastFrame -or $frame -gt $lastFrame) {
                        $lastFrame = $frame
                    }
                }
            } catch {
            }
        }
        $progressText = "unknown"
        if ($null -ne $lastFrame -and $null -ne $expectedNumber -and $expectedNumber -gt 0) {
            $emittedFrame = [math]::Floor($lastFrame / $frameStride) + 1
            $percent = [math]::Min(100.0, ($emittedFrame / $expectedNumber) * 100.0)
            $progressText = ("last_frame={0} frame_stride={1} approx_percent={2:N1}%" -f $lastFrame, $frameStride, $percent)
        } elseif ($null -ne $lastFrame) {
            $progressText = "last_frame=$lastFrame"
        }
        Write-Host "$Label/$InnerProfile/$($_.Name) expected_frames=$expected predictions=$predLines diagnostics=$diagLines $progressText"
    }
}

Show-QstrProfileProgress -Label "recall_oriented" -ProfileRoot (Join-Path $OutAbs "recall_oriented") -InnerProfile "yolo_only"
Show-QstrProfileProgress -Label "strict_fp_control" -ProfileRoot (Join-Path $OutAbs "strict_fp_control") -InnerProfile "yolo_only"
Show-QstrProfileProgress -Label "selected" -ProfileRoot (Join-Path $OutAbs "selected") -InnerProfile $ProfileName

$selectionSummary = Join-Path (Join-Path $OutAbs "selected") "selection_summary.json"
if (Test-Path -LiteralPath $selectionSummary) {
    Write-Host ""
    Write-Host "=== selection counts ==="
    try {
        $json = Get-Content -LiteralPath $selectionSummary -Raw | ConvertFrom-Json
        $json.selection_counts.PSObject.Properties | ForEach-Object {
            Write-Host "$($_.Name): $($_.Value)"
        }
    } catch {
        Write-Host "Could not parse $selectionSummary"
    }
}

foreach ($log in @($Stdout, $Stderr)) {
    if (Test-Path -LiteralPath $log) {
        Write-Host ""
        Write-Host "=== $([System.IO.Path]::GetFileName($log)) tail ==="
        Get-Content -LiteralPath $log -Tail $Tail
    }
}
