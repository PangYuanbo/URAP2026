param([string]$RunName = "ard100_short166_pack_download_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$root = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logs = Join-Path $root "logs"
$statePath = Join-Path $root "state.json"
$modal = Join-Path $env:USERPROFILE ".local\bin\modal.exe"
$target = "D:\URAP_local_datasets\ARD100_SAMURAI_SHORT166"
New-Item -ItemType Directory -Force -Path $logs,$target | Out-Null
$env:PYTHONUTF8 = "1"

function Write-State([string]$Stage, [hashtable]$Extra = @{}) {
    $payload = [ordered]@{ stage = $Stage; updated_at = (Get-Date).ToString("o") }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
}

function Get-Manifest([string]$Split) {
    $probe = Join-Path $root "probe_$Split.json"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $modal volume get --force urap-ard100-samurai-short166-v1 "ARD100_SAMURAI_SHORT166/$($Split)_v1/manifest.json" $probe 2>$null | Out-Null; $exit = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
    if ($exit -ne 0 -or -not (Test-Path -LiteralPath $probe)) { return $null }
    try { return Get-Content -LiteralPath $probe -Raw | ConvertFrom-Json } catch { return $null }
}

$expected = [ordered]@{ val = 10; test = 35; train = 55 }
$downloaded = @()
$manifests = [ordered]@{}
while ($downloaded.Count -lt $expected.Count) {
    $ready = @()
    foreach ($split in $expected.Keys) {
        if ($downloaded -contains $split) { continue }
        $manifest = Get-Manifest $split
        if ($manifest) { $manifests[$split] = $manifest }
        if ($manifest -and [int]$manifest.source_video_count -eq [int]$expected[$split]) { $ready += $split }
    }
    if (-not $ready.Count) {
        Write-State "waiting_for_dataset" @{ done = $downloaded.Count; total = 3; ready = @(); downloaded = $downloaded; manifests = $manifests }
        Start-Sleep -Seconds 60
        continue
    }

    foreach ($split in $ready) {
        $packStdout = Join-Path $logs "pack_$split.stdout.log"
        $packStderr = Join-Path $logs "pack_$split.stderr.log"
        $pack = Start-Process -FilePath $modal -ArgumentList @("run", "tools\modal_pack_ard100_short166.py", "--split", $split) -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $packStdout -RedirectStandardError $packStderr -PassThru
        Write-State "packing" @{ current = $split; pid = $pack.Id; done = $downloaded.Count; total = 3; ready = $ready; downloaded = $downloaded; stdout_log = $packStdout; stderr_log = $packStderr }
        $pack.WaitForExit()
        if ($pack.ExitCode -ne 0) { throw "Modal pack failed for $split with exit code $($pack.ExitCode)" }

    $archive = Join-Path $target "$split.tar"
        Write-State "downloading" @{ current = $split; done = $downloaded.Count; total = 3; ready = $ready; downloaded = $downloaded; archive = $archive }
        & $modal volume get --force urap-ard100-samurai-short166-packs-v1 "ARD100_SAMURAI_SHORT166_$($split)_v1.tar" $archive
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $split" }
        tar -xf $archive -C $target
        if ($LASTEXITCODE -ne 0) { throw "Extract failed: $split" }
        Remove-Item -LiteralPath $archive
        $downloaded += $split
        Write-State "split_downloaded" @{ current = $split; done = $downloaded.Count; total = 3; downloaded = $downloaded; target = (Join-Path $target "${split}_v1") }
    }
}

$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
& $python (Join-Path $repoRoot "tools\materialize_ard100_short166_local.py") --root $target
if ($LASTEXITCODE -ne 0) { throw "Local materialization validation failed" }
Write-State "completed" @{ done = 3; total = 3; target = $target; marker = (Join-Path $target "LOCAL_MATERIALIZE_COMPLETE.json") }
