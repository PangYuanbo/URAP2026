param(
    [string]$SourceRoot = "U:\URAP_datasets\ARD100",
    [string]$Volume = "urap-ard100-raw-v1",
    [string]$StateDir = "C:\Users\aaron\Desktop\URAP\artifacts\modal_ard100_raw_upload",
    [int]$MaxAttempts = 6
)

$ErrorActionPreference = "Stop"
$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$jobs = @()
$jobs += @{ Name="annotations"; Local=(Join-Path $SourceRoot "annotations.zip"); Remote="/ARD100/annotations.zip" }
foreach($video in Get-ChildItem (Join-Path $SourceRoot "train_videos") -File | Sort-Object Name){
    $jobs += @{ Name=("train_" + $video.BaseName); Local=$video.FullName; Remote=("/ARD100/train_videos/" + $video.Name) }
}
foreach($video in Get-ChildItem (Join-Path $SourceRoot "test_videos") -File | Sort-Object Name){
    $jobs += @{ Name=("test_" + $video.BaseName); Local=$video.FullName; Remote=("/ARD100/test_videos/" + $video.Name) }
}
$jobs += @{ Name="extract_frames"; Local=(Join-Path $SourceRoot "YOLOMG_extract_frames.py"); Remote="/ARD100/YOLOMG_extract_frames.py" }
$progressPath = Join-Path $StateDir "progress.json"
$completedDir = Join-Path $StateDir "completed"
New-Item -ItemType Directory -Force -Path $completedDir | Out-Null
for ($index = 0; $index -lt $jobs.Count; $index++) {
    $job = $jobs[$index]
    $marker = Join-Path $completedDir ($job.Name + ".json")
    if (Test-Path $marker) { continue }
    if (-not (Test-Path -LiteralPath $job.Local)) { throw "Missing source: $($job.Local)" }
    $done = @(Get-ChildItem $completedDir -File -Filter "*.json").Count
    [ordered]@{status="running";done=$done;total=$jobs.Count;current=$job.Name;local=$job.Local;remote=$job.Remote;updated=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content $progressPath -Encoding utf8
    $uploaded = $false
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        & modal volume put -f $Volume $job.Local $job.Remote
        if ($LASTEXITCODE -eq 0) {
            $uploaded = $true
            break
        }
        if ($attempt -lt $MaxAttempts) {
            $delaySeconds = [Math]::Min(120, [Math]::Pow(2, $attempt) * 5)
            [ordered]@{status="retrying";done=$done;total=$jobs.Count;current=$job.Name;attempt=$attempt;max_attempts=$MaxAttempts;delay_seconds=$delaySeconds;updated=(Get-Date).ToString("o")} |
                ConvertTo-Json | Set-Content $progressPath -Encoding utf8
            Start-Sleep -Seconds $delaySeconds
        }
    }
    if (-not $uploaded) { throw "Upload failed after $MaxAttempts attempts: $($job.Name)" }
    [ordered]@{name=$job.Name;completed=(Get-Date).ToString("o");local=$job.Local;remote=$job.Remote} |
        ConvertTo-Json | Set-Content $marker -Encoding utf8
}
[ordered]@{status="complete";done=$jobs.Count;total=$jobs.Count;updated=(Get-Date).ToString("o")} |
    ConvertTo-Json | Set-Content $progressPath -Encoding utf8
