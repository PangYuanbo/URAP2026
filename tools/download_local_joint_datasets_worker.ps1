param(
    [string]$DestinationRoot = "D:\URAP_local_datasets",
    [switch]$IncludeAot
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if (-not (Get-Command modal -ErrorAction SilentlyContinue)) {
    throw "Modal CLI not found in PATH"
}

New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
$stateRoot = Join-Path $DestinationRoot ".download_state"
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null

$jobs = @(
    @{ Name = "nps_images_train"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/images/train"; Destination = "NPS_YOLOMG/images"; Target = "NPS_YOLOMG/images/train" },
    @{ Name = "nps_images2_train"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/images2/train"; Destination = "NPS_YOLOMG/images2"; Target = "NPS_YOLOMG/images2/train" },
    @{ Name = "nps_labels_train"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/labels/train"; Destination = "NPS_YOLOMG/labels"; Target = "NPS_YOLOMG/labels/train" },
    @{ Name = "ard_images_train"; Volume = "urap-ard100-yolomg-train-v1"; Remote = "ARD100_YOLOMG/images/train"; Destination = "ARD100_YOLOMG/images"; Target = "ARD100_YOLOMG/images/train" },
    @{ Name = "ard_images2_train"; Volume = "urap-ard100-yolomg-train-v1"; Remote = "ARD100_YOLOMG/images2/train"; Destination = "ARD100_YOLOMG/images2"; Target = "ARD100_YOLOMG/images2/train" },
    @{ Name = "ard_labels_train"; Volume = "urap-ard100-yolomg-train-v1"; Remote = "ARD100_YOLOMG/labels/train"; Destination = "ARD100_YOLOMG/labels"; Target = "ARD100_YOLOMG/labels/train" },
    @{ Name = "nps_images_val"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/images/val"; Destination = "NPS_YOLOMG/images"; Target = "NPS_YOLOMG/images/val" },
    @{ Name = "nps_images2_val"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/images2/val"; Destination = "NPS_YOLOMG/images2"; Target = "NPS_YOLOMG/images2/val" },
    @{ Name = "nps_labels_val"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/labels/val"; Destination = "NPS_YOLOMG/labels"; Target = "NPS_YOLOMG/labels/val" },
    @{ Name = "ard_images_val"; Volume = "urap-ard100-yolomg-eval-v1"; Remote = "ARD100_YOLOMG/images/val"; Destination = "ARD100_YOLOMG/images"; Target = "ARD100_YOLOMG/images/val" },
    @{ Name = "ard_images2_val"; Volume = "urap-ard100-yolomg-eval-v1"; Remote = "ARD100_YOLOMG/images2/val"; Destination = "ARD100_YOLOMG/images2"; Target = "ARD100_YOLOMG/images2/val" },
    @{ Name = "ard_labels_val"; Volume = "urap-ard100-yolomg-eval-v1"; Remote = "ARD100_YOLOMG/labels/val"; Destination = "ARD100_YOLOMG/labels"; Target = "ARD100_YOLOMG/labels/val" },
    @{ Name = "nps_images_test"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/images/test"; Destination = "NPS_YOLOMG/images"; Target = "NPS_YOLOMG/images/test" },
    @{ Name = "nps_images2_test"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/images2/test"; Destination = "NPS_YOLOMG/images2"; Target = "NPS_YOLOMG/images2/test" },
    @{ Name = "nps_labels_test"; Volume = "urap-nps-yolomg-v1"; Remote = "NPS_YOLOMG/labels/test"; Destination = "NPS_YOLOMG/labels"; Target = "NPS_YOLOMG/labels/test" },
    @{ Name = "ard_images_test"; Volume = "urap-ard100-yolomg-eval-v1"; Remote = "ARD100_YOLOMG/images/test"; Destination = "ARD100_YOLOMG/images"; Target = "ARD100_YOLOMG/images/test" },
    @{ Name = "ard_images2_test"; Volume = "urap-ard100-yolomg-eval-v1"; Remote = "ARD100_YOLOMG/images2/test"; Destination = "ARD100_YOLOMG/images2"; Target = "ARD100_YOLOMG/images2/test" },
    @{ Name = "ard_labels_test"; Volume = "urap-ard100-yolomg-eval-v1"; Remote = "ARD100_YOLOMG/labels/test"; Destination = "ARD100_YOLOMG/labels"; Target = "ARD100_YOLOMG/labels/test" }
)

if ($IncludeAot) {
    $jobs += @{ Name = "aot_part1"; Volume = "urap-aot-part1-raw-v1"; Remote = "AOT_part1"; Destination = "."; Target = "AOT_part1" }
}

$planPath = Join-Path $stateRoot "plan.json"
$jobs | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $planPath -Encoding UTF8

for ($index = 0; $index -lt $jobs.Count; $index++) {
    $job = $jobs[$index]
    $donePath = Join-Path $stateRoot "$($job.Name).complete.json"
    if (Test-Path -LiteralPath $donePath) {
        Write-Output "SKIP completed $($index + 1)/$($jobs.Count): $($job.Name)"
        continue
    }

    $destinationPath = Join-Path $DestinationRoot $job.Destination
    $targetPath = Join-Path $DestinationRoot $job.Target
    New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
    @{
        index = $index + 1
        total = $jobs.Count
        name = $job.Name
        volume = $job.Volume
        remote = $job.Remote
        local = $targetPath
        started = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateRoot "current.json") -Encoding UTF8

    Write-Output "START $($index + 1)/$($jobs.Count): $($job.Volume):$($job.Remote) -> $targetPath"
    & modal volume get --force $job.Volume $job.Remote $destinationPath
    if ($LASTEXITCODE -ne 0) {
        throw "Modal download failed for $($job.Name), exit=$LASTEXITCODE"
    }

    $files = @(Get-ChildItem -LiteralPath $targetPath -Recurse -File -ErrorAction Stop)
    if ($files.Count -eq 0) {
        throw "Downloaded unit is empty: $($job.Name) at $targetPath"
    }
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    @{
        index = $index + 1
        total = $jobs.Count
        name = $job.Name
        completed = (Get-Date).ToString("o")
        files = $files.Count
        bytes = $bytes
        local = $targetPath
    } | ConvertTo-Json | Set-Content -LiteralPath $donePath -Encoding UTF8
    Write-Output "DONE $($index + 1)/$($jobs.Count): $($job.Name) files=$($files.Count) bytes=$bytes"
}

Remove-Item -LiteralPath (Join-Path $stateRoot "current.json") -Force -ErrorAction SilentlyContinue
@{
    completed = (Get-Date).ToString("o")
    total = $jobs.Count
    destination = $DestinationRoot
    include_aot = [bool]$IncludeAot
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateRoot "all.complete.json") -Encoding UTF8
Write-Output "ALL DOWNLOAD UNITS COMPLETE: $($jobs.Count)/$($jobs.Count)"
