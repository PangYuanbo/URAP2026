param(
  [string]$DataRoot = "U:\URAP_datasets",
  [int]$MinTrainFrames = 50000,
  [int]$MinValFrames = 1,
  [int]$MinTestFrames = 1,
  [int]$UseGtExpectedFrames = 1,
  [int]$VerifyGtFrameFiles = 1,
  [string]$TrainGtCsv = "",
  [string]$ValGtCsv = "",
  [string]$TestGtCsv = ""
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($TrainGtCsv)) {
  $TrainGtCsv = Join-Path $Repo "artifacts\nps_sota_research\tvd_nps_train_route_b_v3\gt.csv"
}
if ([string]::IsNullOrWhiteSpace($ValGtCsv)) {
  $ValGtCsv = Join-Path $Repo "artifacts\nps_sota_research\tvd_nps_val_route_b_v2\gt.csv"
}
if ([string]::IsNullOrWhiteSpace($TestGtCsv)) {
  $TestGtCsv = Join-Path $Repo "artifacts\nps_sota_research\tvd_nps_test_route_b_v2\gt.csv"
}
$Splits = @(
  @{ name = "train"; min = $MinTrainFrames; gt = $TrainGtCsv },
  @{ name = "val"; min = $MinValFrames; gt = $ValGtCsv },
  @{ name = "test"; min = $MinTestFrames; gt = $TestGtCsv }
)

Write-Output "DataRoot: $DataRoot"
if (-not (Test-Path $DataRoot)) {
  Write-Output "Status: NOT READY"
  Write-Output "Reason: data root not found"
  exit 2
}

$AllReady = $true
foreach ($Split in $Splits) {
  $FramesDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\$($Split.name)"
  $GtCsv = [string]$Split.gt
  $Exists = Test-Path $FramesDir
  $Count = 0
  if ($Exists) {
    $Count = (Get-ChildItem -LiteralPath $FramesDir -Filter "*.png" -File | Measure-Object).Count
  }
  $GtExists = Test-Path $GtCsv
  $GtUniqueFrames = 0
  $MissingGtFrames = 0
  $MissingGtFrameExamples = @()
  $UseGtExpected = $UseGtExpectedFrames -ne 0
  $VerifyGtFrames = $VerifyGtFrameFiles -ne 0
  if ($GtExists -and $UseGtExpected) {
    $Seen = @{}
    foreach ($Row in Import-Csv -LiteralPath $GtCsv) {
      $VideoPath = [string]$Row.video_path
      if ([string]::IsNullOrWhiteSpace($VideoPath)) {
        continue
      }
      $FrameName = [System.IO.Path]::GetFileName($VideoPath)
      if ([string]::IsNullOrWhiteSpace($FrameName)) {
        continue
      }
      $Seen[$FrameName] = $true
    }
    $GtUniqueFrames = $Seen.Count
    if ($VerifyGtFrames -and $Exists) {
      foreach ($FrameName in $Seen.Keys) {
        $FramePath = Join-Path $FramesDir $FrameName
        if (-not (Test-Path $FramePath)) {
          $MissingGtFrames += 1
          if ($MissingGtFrameExamples.Count -lt 5) {
            $MissingGtFrameExamples += $FrameName
          }
        }
      }
    }
  }
  $RequiredFrames = [Math]::Max([int]$Split.min, [int]$GtUniqueFrames)
  $Ready = $Exists -and ($Count -ge $RequiredFrames) -and $GtExists -and ($MissingGtFrames -eq 0)
  if (-not $Ready) {
    $AllReady = $false
  }
  $MissingExamplesText = if ($MissingGtFrameExamples.Count -gt 0) { $MissingGtFrameExamples -join "|" } else { "" }
  Write-Output "$($Split.name): ready=$Ready frames=$Count required=$RequiredFrames user_min=$($Split.min) gt_unique_frames=$GtUniqueFrames use_gt_expected_frames=$UseGtExpected verify_gt_frame_files=$VerifyGtFrames missing_gt_frames=$MissingGtFrames missing_gt_examples=$MissingExamplesText frames_dir=$FramesDir gt_exists=$GtExists gt=$GtCsv"
}

if ($AllReady) {
  Write-Output "Status: READY"
  exit 0
}

Write-Output "Status: NOT READY"
exit 1
