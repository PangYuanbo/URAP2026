param(
    [string]$RunDir = "C:\Users\aaron\Desktop\URAP\artifacts\joint_training\yolomg_nps_ard100_e20",
    [string]$TrainingPidFile = "C:\Users\aaron\Desktop\URAP\artifacts\joint_training\yolomg_runner\train.pid",
    [int]$IntervalSeconds = 7200
)

$ErrorActionPreference = "Stop"
$snapshotRoot = Join-Path $RunDir "weights\time_snapshots"
New-Item -ItemType Directory -Force -Path $snapshotRoot | Out-Null
$lastCheckpoint = Join-Path $RunDir "weights\last.pt"

while ($true) {
    $trainingPid = if (Test-Path -LiteralPath $TrainingPidFile) { Get-Content -LiteralPath $TrainingPidFile | Select-Object -First 1 } else { $null }
    $trainingProcess = if ($trainingPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $trainingPid" -ErrorAction SilentlyContinue } else { $null }
    if (-not ($trainingProcess -and $trainingProcess.CommandLine -like "*train.py*" -and $trainingProcess.CommandLine -like "*joint_nps_ard100.yaml*")) {
        Write-Output "TRAINING NOT RUNNING; snapshot watcher stopping at $(Get-Date -Format o)"
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
    if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $temporary = Join-Path $snapshotRoot "snapshot_${timestamp}.pt.partial"
        $destination = Join-Path $snapshotRoot "snapshot_${timestamp}.pt"
        Copy-Item -LiteralPath $lastCheckpoint -Destination $temporary -Force
        Move-Item -LiteralPath $temporary -Destination $destination -Force
        Write-Output "SAVED $destination bytes=$((Get-Item -LiteralPath $destination).Length) at $(Get-Date -Format o)"
    } else {
        Write-Output "SKIP no last.pt yet at $(Get-Date -Format o)"
    }
}
