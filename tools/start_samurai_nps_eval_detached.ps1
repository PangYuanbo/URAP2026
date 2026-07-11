param(
    [string]$DatasetRoot = "U:\URAP_datasets\TransVisDrone\NPS\SAMURAI\val_v1",
    [string]$Split = "val",
    [string]$Checkpoint = "U:\URAP_models\samurai\sam2.1_hiera_tiny.pt",
    [string]$ModelConfig = "configs/samurai/sam2.1_hiera_t.yaml",
    [string]$RunRoot = "U:\URAP_runs\samurai\zero_shot_tiny_val_v1",
    [string]$Device = "cuda:0",
    [string]$Dtype = "bfloat16",
    [ValidateSet("video", "image-box")][string]$PropagationMode = "video",
    [string]$FeatureOutput = "",
    [switch]$Resume,
    [switch]$AsyncLoadingFrames,
    [switch]$OffloadStateToCpu,
    [int]$MaxSequences = 0,
    [int]$MaxFrames = 0
    ,[int]$SequenceShardCount = 1
    ,[int]$SequenceShardIndex = 0
    ,[string]$ControlName = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runner = Join-Path $repoRoot "tools\eval_samurai_nps.py"
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
New-Item -ItemType Directory -Force -Path $controlRoot, $RunRoot | Out-Null

$runName = if ($ControlName) { $ControlName } else { Split-Path -Leaf $RunRoot }
$pidPath = Join-Path $controlRoot "$runName.pid"
$metaPath = Join-Path $controlRoot "$runName.meta.json"
$stdoutPath = Join-Path $controlRoot "$runName.stdout.log"
$stderrPath = Join-Path $controlRoot "$runName.stderr.log"

if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*eval_samurai_nps.py*") {
        throw "SAMURAI evaluation is already running with PID $oldPid"
    }
}

$arguments = @(
    $runner,
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--checkpoint", $Checkpoint,
    "--model-config", $ModelConfig,
    "--output-root", $RunRoot,
    "--device", $Device,
    "--dtype", $Dtype,
    "--propagation-mode", $PropagationMode
)
if ($FeatureOutput) { $arguments += @("--feature-output", $FeatureOutput) }
if ($Resume) { $arguments += "--resume" }
if ($AsyncLoadingFrames) { $arguments += "--async-loading-frames" }
if ($OffloadStateToCpu) { $arguments += "--offload-state-to-cpu" }
if ($MaxSequences -gt 0) { $arguments += @("--max-sequences", $MaxSequences) }
if ($MaxFrames -gt 0) { $arguments += @("--max-frames", $MaxFrames) }
if ($SequenceShardCount -gt 1) { $arguments += @("--sequence-shard-count", $SequenceShardCount, "--sequence-shard-index", $SequenceShardIndex) }
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -Path $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "$python $($arguments -join ' ')"
    dataset_root = $DatasetRoot
    checkpoint = $Checkpoint
    model_config = $ModelConfig
    propagation_mode = $PropagationMode
    feature_output = $FeatureOutput
    resume = [bool]$Resume
    async_loading_frames = [bool]$AsyncLoadingFrames
    offload_state_to_cpu = [bool]$OffloadStateToCpu
    sequence_shard_count = $SequenceShardCount
    sequence_shard_index = $SequenceShardIndex
    control_name = $runName
    run_root = $RunRoot
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    progress_file = (Join-Path $RunRoot "progress.json")
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -Path $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4
