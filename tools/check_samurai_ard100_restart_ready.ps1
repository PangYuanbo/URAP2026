param(
    [string]$OutputPath = "C:\Users\aaron\Desktop\URAP\artifacts\samurai_runs\ard100_restart_preflight.json"
)

$ErrorActionPreference = "Stop"
$datasetRoot = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI"
$modelPath = "C:\Users\aaron\Desktop\URAP\artifacts\samurai_models\sam2.1_hiera_base_plus.pt"
$checkpointDir = "C:\Users\aaron\Desktop\URAP\artifacts\samurai_checkpoints\finetune_base_plus_ard100_fullframe_stage1_restart1"
$checks = [ordered]@{}

function Test-ReadableFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $stream = $null
    try {
        $stream = [System.IO.File]::Open($Path, "Open", "Read", "ReadWrite")
        return $stream.ReadByte() -ge 0
    } catch {
        return $false
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
    }
}

$checks.u_root_exists = Test-Path -LiteralPath "U:\"
$volume = Get-Volume -DriveLetter U -ErrorAction SilentlyContinue
$checks.volume_label = if ($volume) { $volume.FileSystemLabel } else { $null }
$checks.volume_label_ok = [bool]($volume -and $volume.FileSystemLabel -eq "DATASETS")
$checks.volume_health = if ($volume) { [string]$volume.HealthStatus } else { $null }
$checks.volume_operational_status = if ($volume) { [string]$volume.OperationalStatus } else { $null }
$checks.volume_health_ok = [bool]($volume -and $volume.HealthStatus -eq "Healthy" -and $volume.OperationalStatus -eq "OK")
$disk = Get-Disk -ErrorAction SilentlyContinue | Where-Object {
    $_.FriendlyName -eq "Fanxiang S880 2TB" -and $_.BusType -eq "USB"
} | Select-Object -First 1
$checks.device_present = [bool]$disk
$checks.device_name = if ($disk) { $disk.FriendlyName } else { $null }
$checks.device_bus = if ($disk) { [string]$disk.BusType } else { $null }

$expected = [ordered]@{ train_v1 = 55; val_v1 = 10; test_v1 = 35 }
$splitChecks = [ordered]@{}
foreach ($split in $expected.Keys) {
    $splitRoot = "$datasetRoot\$split"
    $splitName = $split -replace "_v1$", ""
    $listPath = "$splitRoot\${splitName}_set.txt"
    $names = @()
    if (Test-Path -LiteralPath $listPath) {
        $names = @(Get-Content -LiteralPath $listPath | Where-Object { $_.Trim() })
    }
    $first = if ($names.Count) { $names[0].Trim() } else { $null }
    $frame = if ($first) {
        Get-ChildItem -LiteralPath "$splitRoot\vos\JPEGImages\$first" -Filter *.jpg -File -ErrorAction SilentlyContinue | Select-Object -First 1
    } else { $null }
    $mask = if ($first -and $frame) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($frame.Name)
        "$splitRoot\vos\Annotations\$first\$stem.png"
    } else { $null }
    $splitChecks[$split] = [ordered]@{
        root_exists = Test-Path -LiteralPath $splitRoot
        expected_sequences = $expected[$split]
        listed_sequences = $names.Count
        sequence_count_ok = $names.Count -eq $expected[$split]
        representative_frame = if ($frame) { $frame.FullName } else { $null }
        representative_frame_readable = [bool]($frame -and (Test-ReadableFile $frame.FullName))
        representative_mask = $mask
        representative_mask_readable = [bool]($mask -and (Test-ReadableFile $mask))
    }
}
$checks.splits = $splitChecks
$checks.base_model = $modelPath
$checks.base_model_readable = Test-ReadableFile $modelPath
$checks.checkpoint_dir = $checkpointDir
$checks.checkpoint_dir_parent_exists = Test-Path -LiteralPath (Split-Path -Parent $checkpointDir)
$cDrive = Get-PSDrive C
$checks.c_free_gb = [math]::Round($cDrive.Free / 1GB, 2)
$checks.c_space_ok = $cDrive.Free -ge 20GB

$ready = $checks.u_root_exists -and $checks.volume_label_ok -and $checks.volume_health_ok -and `
    $checks.device_present -and $checks.base_model_readable -and $checks.c_space_ok
foreach ($splitCheck in $splitChecks.Values) {
    $ready = $ready -and $splitCheck.root_exists -and $splitCheck.sequence_count_ok -and `
        $splitCheck.representative_frame_readable -and $splitCheck.representative_mask_readable
}

$result = [ordered]@{
    status = if ($ready) { "ready" } else { "not_ready" }
    observed_at = (Get-Date).ToString("o")
    checks = $checks
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null
$tmp = "$OutputPath.$PID.tmp"
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tmp -Encoding utf8
Move-Item -LiteralPath $tmp -Destination $OutputPath -Force
$result | ConvertTo-Json -Depth 8
if (-not $ready) { exit 2 }
exit 0
