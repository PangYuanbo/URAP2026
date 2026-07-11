param([int]$MinimumFreeMiB = 29000)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataset = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166"
$marker = Join-Path $dataset "LOCAL_MATERIALIZE_COMPLETE.json"
$model = Join-Path $repoRoot "artifacts\samurai_models\sam2.1_hiera_base_plus.pt"
$gpu = (& nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits).Trim().Split(',')
$freeMiB = [int]$gpu[0].Trim(); $util = [int]$gpu[1].Trim()
$pythonGpuJobs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train\.py|eval_samurai_nps\.py' })
$checks = [ordered]@{
    dataset_marker = Test-Path -LiteralPath $marker
    train_manifest = Test-Path -LiteralPath (Join-Path $dataset "train_v1\manifest.json")
    val_manifest = Test-Path -LiteralPath (Join-Path $dataset "val_v1\manifest.json")
    test_manifest = Test-Path -LiteralPath (Join-Path $dataset "test_v1\manifest.json")
    model = Test-Path -LiteralPath $model -PathType Leaf
    model_bytes = if (Test-Path -LiteralPath $model -PathType Leaf) { (Get-Item -LiteralPath $model).Length } else { 0 }
    gpu_free_mib = $freeMiB
    gpu_utilization = $util
    gpu_free_ok = $freeMiB -ge $MinimumFreeMiB
    conflicting_python_gpu_jobs = $pythonGpuJobs.Count
    no_conflicting_python_gpu_jobs = $pythonGpuJobs.Count -eq 0
}
$ready = $checks.dataset_marker -and $checks.train_manifest -and $checks.val_manifest -and $checks.test_manifest -and $checks.model -and $checks.gpu_free_ok -and $checks.no_conflicting_python_gpu_jobs
[ordered]@{ ready = $ready; minimum_free_mib = $MinimumFreeMiB; checks = $checks; conflicting_jobs = $pythonGpuJobs | Select-Object ProcessId,CreationDate,CommandLine } | ConvertTo-Json -Depth 6
if (-not $ready) { exit 2 }
