param(
    [int]$MaxWorkers = 4,
    [int]$FirstSequence = 5,
    [int]$LastSequence = 14
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo 'artifacts\venvs\nps_flow\Scripts\python.exe'
$workerScript = Join-Path $repo 'tools\yolomg_pure_difference_1080p.py'
$batchScript = Join-Path $repo 'tools\run_yolomg_pure_1080p_batch.py'
$inputDir = 'C:\Users\aaron\Desktop\Drone_Videos_Chronological'
$outputDir = Join-Path $repo 'artifacts\yolomg_pure_difference_1080p_new_batch'
$runsRoot = Join-Path $repo 'artifacts\runs\yolomg_pure_1080p_new_batch'
$runId = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $runsRoot $runId
$stdoutLog = Join-Path $runDir 'coordinator_stdout.log'
$stderrLog = Join-Path $runDir 'coordinator_stderr.log'
$pidFile = Join-Path $runDir 'coordinator_pid.txt'
$latestFile = Join-Path $repo 'artifacts\runs\yolomg_pure_1080p_new_batch_latest.json'

foreach ($required in @($python, $workerScript, $batchScript, $inputDir)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $runDir, $outputDir | Out-Null
$arguments = @(
    $batchScript,
    '--input-dir', $inputDir,
    '--output-dir', $outputDir,
    '--run-dir', $runDir,
    '--worker-script', $workerScript,
    '--python', $python,
    '--first-sequence', $FirstSequence.ToString(),
    '--last-sequence', $LastSequence.ToString(),
    '--max-workers', $MaxWorkers.ToString()
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
$startTime = Get-Date
$latest = [ordered]@{
    run_id = $runId
    run_dir = $runDir
    coordinator_pid = $process.Id
    started_at = $startTime.ToString('o')
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    status_json = (Join-Path $runDir 'status.json')
    output_dir = $outputDir
}
$latest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $latestFile -Encoding utf8

Write-Output "Started YOLOMG 1080p batch."
Write-Output "PID: $($process.Id)"
Write-Output "Started: $($startTime.ToString('o'))"
Write-Output "Run directory: $runDir"
Write-Output "Status: $(Join-Path $runDir 'status.json')"
Write-Output "Logs: $stdoutLog ; $stderrLog"

