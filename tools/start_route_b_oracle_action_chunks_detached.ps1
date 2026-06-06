param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string[]]$ListFiles = @(),
  [string[]]$SourceNames = @(),
  [string]$OutDir = (Join-Path $RepoRoot 'artifacts\route_b_official\oracle_action_chunks'),
  [Nullable[int]]$MaxImagesPerSource = $null,
  [Nullable[int]]$MaxLabeledImagesPerSeq = $null,
  [int[]]$SkipImagesPerSource = @(),
  [int]$OracleMinTrackletRows = 4,
  [Nullable[int]]$ImageWidth = $null,
  [Nullable[int]]$ImageHeight = $null,
  [int]$PastLen = 8,
  [int]$FutureLen = 8,
  [double]$CalibFraction = 0.2,
  [double]$TestFraction = 0.0,
  [int]$Seed = 59,
  [string]$RunId = 'route_b_oracle_action_chunks',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_oracle_action_chunks')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if ($ListFiles.Count -lt 1) { throw 'ListFiles must contain at least one YOLO image-list file' }
if ($SourceNames.Count -ne $ListFiles.Count) { throw 'SourceNames must have the same length as ListFiles' }
if ($SkipImagesPerSource.Count -gt 0 -and $SkipImagesPerSource.Count -ne $ListFiles.Count) { throw 'SkipImagesPerSource must be empty or have the same length as ListFiles' }
if ($null -ne $MaxImagesPerSource -and $MaxImagesPerSource -le 0) { throw 'MaxImagesPerSource must be positive when provided; omit it for a full export' }
if ($null -ne $MaxLabeledImagesPerSeq -and $MaxLabeledImagesPerSeq -le 0) { throw 'MaxLabeledImagesPerSeq must be positive when provided; omit it for no per-sequence cap' }
if (($null -eq $ImageWidth) -ne ($null -eq $ImageHeight)) { throw 'ImageWidth and ImageHeight must be provided together' }
foreach ($path in $ListFiles) {
  if (-not (Test-Path -Path $path -PathType Leaf)) { throw "List file not found: $path" }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
$workerScript = Join-Path $OutputRoot ("runner_{0}_worker.ps1" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    $expectedWorkerName = Split-Path -Leaf $workerScript
    if ($existing -and $existing.CommandLine -like "*$expectedWorkerName*") {
      Write-Host "Route B oracle action chunk export already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 140 }
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)
$summaryJson = Join-Path $OutDir 'oracle_action_chunk_pipeline_summary.json'

function ConvertTo-LiteralArray([string[]]$Values) {
  $items = foreach ($value in $Values) {
    "'" + ($value -replace "'", "''") + "'"
  }
  return "@(" + ($items -join ', ') + ")"
}

function ConvertTo-IntLiteralArray([int[]]$Values) {
  $items = foreach ($value in $Values) { [string]$value }
  return "@(" + ($items -join ', ') + ")"
}

$listLiteral = ConvertTo-LiteralArray -Values $ListFiles
$sourceLiteral = ConvertTo-LiteralArray -Values $SourceNames
$skipValues = if ($SkipImagesPerSource.Count -gt 0) { $SkipImagesPerSource } else { @(foreach ($unused in $ListFiles) { 0 }) }
$skipLiteral = ConvertTo-IntLiteralArray -Values $skipValues
$maxImagesLine = if ($null -ne $MaxImagesPerSource) { "`$maxImages = $MaxImagesPerSource" } else { '$maxImages = $null' }
$maxLabeledLine = if ($null -ne $MaxLabeledImagesPerSeq) { "`$maxLabeledImagesPerSeq = $MaxLabeledImagesPerSeq" } else { '$maxLabeledImagesPerSeq = $null' }
$imageWidthLine = if ($null -ne $ImageWidth) { "`$imageWidth = $ImageWidth" } else { '$imageWidth = $null' }
$imageHeightLine = if ($null -ne $ImageHeight) { "`$imageHeight = $ImageHeight" } else { '$imageHeight = $null' }

@"
`$ErrorActionPreference = 'Stop'
`$repoRoot = '$($RepoRoot -replace "'", "''")'
`$python = '$($PythonExe -replace "'", "''")'
`$outDir = '$($OutDir -replace "'", "''")'
`$listFiles = $listLiteral
`$sourceNames = $sourceLiteral
`$skipImages = $skipLiteral
$maxImagesLine
$maxLabeledLine
$imageWidthLine
$imageHeightLine
`$oracleJsonls = @()
`$actionJsonls = @()
`$summaries = @()
for (`$i = 0; `$i -lt `$listFiles.Count; `$i++) {
  `$source = `$sourceNames[`$i]
  `$sourceDir = Join-Path `$outDir `$source
  New-Item -ItemType Directory -Force -Path `$sourceDir | Out-Null
  `$exportArgs = @('-m', 'qstr_dronedet.cli', 'export-yolo-oracle-tracklets', '--list-files', `$listFiles[`$i], '--out', `$sourceDir, '--dataset-source', `$source, '--min-tracklet-rows', '$OracleMinTrackletRows')
  if (`$null -ne `$imageWidth) { `$exportArgs += @('--image-width', [string]`$imageWidth, '--image-height', [string]`$imageHeight) }
  if ([int]`$skipImages[`$i] -gt 0) { `$exportArgs += @('--skip-images', [string]`$skipImages[`$i]) }
  if (`$null -ne `$maxImages) { `$exportArgs += @('--max-images', [string]`$maxImages) }
  if (`$null -ne `$maxLabeledImagesPerSeq) { `$exportArgs += @('--max-labeled-images-per-seq', [string]`$maxLabeledImagesPerSeq) }
  & `$python @exportArgs
  if (`$LASTEXITCODE -ne 0) { throw "oracle export failed for `$source" }
  `$oracleJsonl = Join-Path `$sourceDir 'oracle_tracklets.jsonl'
  `$actionJsonl = Join-Path `$sourceDir 'action_chunk_samples.jsonl'
  & `$python -m qstr_dronedet.cli build-action-chunk-dataset --tracklet-jsonl `$oracleJsonl --out `$actionJsonl --past-len '$PastLen' --future-len '$FutureLen' --positives-only --normalize-by-row-image-size
  if (`$LASTEXITCODE -ne 0) { throw "action chunk export failed for `$source" }
  `$oracleJsonls += `$oracleJsonl
  `$actionJsonls += `$actionJsonl
  `$summaryPath = Join-Path `$sourceDir 'summary.json'
  `$actionSummary = Join-Path `$sourceDir 'action_chunk_samples.jsonl.summary.json'
  `$summaries += [ordered]@{ source=`$source; oracle_tracklets=`$oracleJsonl; action_chunks=`$actionJsonl; summary=`$summaryPath }
}
`$merged = Join-Path `$outDir 'merged_action_chunks.jsonl'
`$manifest = Join-Path `$outDir 'merged_action_chunks.manifest.json'
& `$python -m qstr_dronedet.cli merge-action-chunk-datasets --inputs `$actionJsonls --out `$merged --source-names `$sourceNames --manifest-out `$manifest
if (`$LASTEXITCODE -ne 0) { throw 'merge action chunks failed' }
`$splitDir = Join-Path `$outDir 'split'
& `$python -m qstr_dronedet.cli split-action-chunk-dataset --jsonl `$merged --out-dir `$splitDir --calib-fraction '$CalibFraction' --test-fraction '$TestFraction' --seed '$Seed'
if (`$LASTEXITCODE -ne 0) { throw 'split action chunks failed' }
`$pipeline = [ordered]@{
  list_files = `$listFiles
  source_names = `$sourceNames
  skip_images_per_source = `$skipImages
  out_dir = `$outDir
  max_images_per_source = `$maxImages
  max_labeled_images_per_seq = `$maxLabeledImagesPerSeq
  image_width = `$imageWidth
  image_height = `$imageHeight
  past_len = $PastLen
  future_len = $FutureLen
  oracle_min_tracklet_rows = $OracleMinTrackletRows
  normalize_by_row_image_size = `$true
  sources = `$summaries
  merged_action_chunks = `$merged
  merged_manifest = `$manifest
  split_dir = `$splitDir
  split_manifest = (Join-Path `$splitDir 'action_chunk_split_manifest.json')
  note = 'Oracle YOLO-label action chunks are for motion-prior pretraining/sanity only, not detector benchmark reporting.'
}
`$pipeline | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -Path '$($summaryJson -replace "'", "''")'
"@ | Set-Content -Encoding utf8 -Path $workerScript

$process = Start-Process `
  -FilePath 'powershell.exe' `
  -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $workerScript) `
  -WorkingDirectory $RepoRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$process.Id | Set-Content -Encoding ascii -Path $pidFile

@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $process.Id),
  ('python={0}' -f $PythonExe),
  ('run_id={0}' -f $RunId),
  ('repo_root={0}' -f $RepoRoot),
  ('out_dir={0}' -f $OutDir),
  ('output_root={0}' -f $OutputRoot),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('worker_script={0}' -f $workerScript),
  ('summary_json={0}' -f $summaryJson),
  ('list_files={0}' -f ($ListFiles -join ';')),
  ('source_names={0}' -f ($SourceNames -join ';')),
  ('skip_images_per_source={0}' -f ($skipValues -join ';')),
  ('max_labeled_images_per_seq={0}' -f $MaxLabeledImagesPerSeq),
  ('max_images_per_source={0}' -f $MaxImagesPerSource),
  ('image_width={0}' -f $ImageWidth),
  ('image_height={0}' -f $ImageHeight)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Route B oracle action chunk export.'
Get-Content $metaFile
