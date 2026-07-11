param(
    [string]$SourceRoot = "U:\URAP_datasets\AOT\part1",
    [string]$Volume = "urap-aot-part1-archives-v1",
    [string]$StateDir = "C:\Users\aaron\Desktop\URAP\artifacts\modal_aot_upload",
    [string]$ScratchDir = "D:\urap_modal_aot_staging",
    [int]$MaxAttempts = 6
)

$ErrorActionPreference = "Stop"
$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$imagesRoot = Join-Path $SourceRoot "Images"
$groundTruth = Join-Path $SourceRoot "ImageSets\groundtruth.json"
if (-not (Test-Path -LiteralPath $imagesRoot)) { throw "Missing images root: $imagesRoot" }
if (-not (Test-Path -LiteralPath $groundTruth)) { throw "Missing ground truth: $groundTruth" }

$completedDir = Join-Path $StateDir "completed"
$progressPath = Join-Path $StateDir "progress.json"
New-Item -ItemType Directory -Force -Path $completedDir, $ScratchDir | Out-Null

function Invoke-ModalUploadWithRetry {
    param([string]$LocalPath, [string]$RemotePath, [string]$ItemName, [int]$Done, [int]$Total)
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        & modal volume put -f $Volume $LocalPath $RemotePath
        if ($LASTEXITCODE -eq 0) { return }
        if ($attempt -lt $MaxAttempts) {
            $delaySeconds = [Math]::Min(120, [Math]::Pow(2, $attempt) * 5)
            [ordered]@{status="retrying";done=$Done;total=$Total;current=$ItemName;attempt=$attempt;max_attempts=$MaxAttempts;delay_seconds=$delaySeconds;updated=(Get-Date).ToString("o")} |
                ConvertTo-Json | Set-Content $progressPath -Encoding utf8
            Start-Sleep -Seconds $delaySeconds
        }
    }
    throw "Upload failed after $MaxAttempts attempts: $ItemName"
}

$flights = @(Get-ChildItem -LiteralPath $imagesRoot -Directory | Sort-Object Name)
$total = $flights.Count + 2

$groundTruthMarker = Join-Path $completedDir "groundtruth.json"
if (-not (Test-Path -LiteralPath $groundTruthMarker)) {
    $done = @(Get-ChildItem $completedDir -File -Filter "*.json").Count
    [ordered]@{status="running";done=$done;total=$total;current="groundtruth";updated=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content $progressPath -Encoding utf8
    Invoke-ModalUploadWithRetry $groundTruth "/AOT_part1/ImageSets/groundtruth.json" "groundtruth" $done $total
    $info = Get-Item -LiteralPath $groundTruth
    [ordered]@{name="groundtruth";files=1;source_bytes=$info.Length;archive_bytes=$info.Length;completed=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content $groundTruthMarker -Encoding utf8
}

$readmePath = Join-Path $StateDir "MOUNT_README.txt"
@(
    "Mount urap-aot-part1-archives-v1 at /aot_archives."
    "Each file in /aot_archives/AOT_part1/flight_archives is a tar containing one Images/<flight_id> directory."
    "Extract only the flights required by a training/evaluation shard to ephemeral storage."
    "Official annotations are at /aot_archives/AOT_part1/ImageSets/groundtruth.json."
    "Use tools/prepare_transvisdrone_aot_part1.py from urap-code-artifacts-v1 to build TransVisDrone layouts."
) | Set-Content $readmePath -Encoding utf8
Invoke-ModalUploadWithRetry $readmePath "/AOT_part1/MOUNT_README.txt" "readme" 1 $total

foreach ($flight in $flights) {
    $marker = Join-Path $completedDir ($flight.Name + ".json")
    if (Test-Path -LiteralPath $marker) { continue }
    $done = @(Get-ChildItem $completedDir -File -Filter "*.json").Count
    $archivePath = Join-Path $ScratchDir ($flight.Name + ".tar")
    [ordered]@{status="packing";done=$done;total=$total;current=$flight.Name;updated=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content $progressPath -Encoding utf8
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }
    & tar.exe -cf $archivePath -C $imagesRoot $flight.Name
    if ($LASTEXITCODE -ne 0) { throw "Archive failed: $($flight.Name) exit=$LASTEXITCODE" }
    $files = @(Get-ChildItem -LiteralPath $flight.FullName -File)
    $sourceBytes = ($files | Measure-Object Length -Sum).Sum
    $archiveBytes = (Get-Item -LiteralPath $archivePath).Length
    [ordered]@{status="uploading";done=$done;total=$total;current=$flight.Name;files=$files.Count;source_bytes=$sourceBytes;archive_bytes=$archiveBytes;updated=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content $progressPath -Encoding utf8
    Invoke-ModalUploadWithRetry $archivePath ("/AOT_part1/flight_archives/" + $flight.Name + ".tar") $flight.Name $done $total
    [ordered]@{name=$flight.Name;files=$files.Count;source_bytes=$sourceBytes;archive_bytes=$archiveBytes;completed=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content $marker -Encoding utf8
    Remove-Item -LiteralPath $archivePath -Force
}

$manifestPath = Join-Path $StateDir "manifest.json"
$items = @(Get-ChildItem -LiteralPath $completedDir -File -Filter "*.json" |
    Where-Object Name -ne "_manifest.json" |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
[ordered]@{
    schema_version = 1
    source_root = $SourceRoot
    volume = $Volume
    flights = $flights.Count
    image_files = ($items | Where-Object name -ne "groundtruth" | Measure-Object files -Sum).Sum
    source_bytes = ($items | Measure-Object source_bytes -Sum).Sum
    archive_bytes = ($items | Measure-Object archive_bytes -Sum).Sum
    generated = (Get-Date).ToString("o")
    items = $items
} | ConvertTo-Json -Depth 5 | Set-Content $manifestPath -Encoding utf8

$done = @(Get-ChildItem $completedDir -File -Filter "*.json").Count
Invoke-ModalUploadWithRetry $manifestPath "/AOT_part1/manifest.json" "manifest" $done $total
[ordered]@{name="manifest";files=1;source_bytes=(Get-Item $manifestPath).Length;archive_bytes=(Get-Item $manifestPath).Length;completed=(Get-Date).ToString("o")} |
    ConvertTo-Json | Set-Content (Join-Path $completedDir "_manifest.json") -Encoding utf8
[ordered]@{status="complete";done=$total;total=$total;updated=(Get-Date).ToString("o")} |
    ConvertTo-Json | Set-Content $progressPath -Encoding utf8
