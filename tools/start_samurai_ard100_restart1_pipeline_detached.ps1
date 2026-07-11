param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$preflight = Join-Path $repoRoot "tools\check_samurai_ard100_restart_ready.ps1"
$finetuneScript = Join-Path $repoRoot "tools\sequence_samurai_ard100_finetune.ps1"
$postScript = Join-Path $repoRoot "tools\sequence_samurai_ard100_post_finetune.ps1"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null

& $preflight | Out-Null
$preflightExit = $LASTEXITCODE
if ($null -eq $preflightExit -or $preflightExit -ne 0) {
    $report = Join-Path $controlRoot "ard100_restart_preflight.json"
    throw "ARD100 restart preflight is not ready (exit $preflightExit). See $report"
}

$jobs = @(
    [ordered]@{
        name = "ard100_finetune_restart1_sequencer"
        script = $finetuneScript
        progress = Join-Path $controlRoot "ard100_finetune_restart1_sequencer.progress.json"
    },
    [ordered]@{
        name = "ard100_post_finetune_restart1_pipeline"
        script = $postScript
        progress = Join-Path $controlRoot "ard100_post_finetune_restart1_pipeline.progress.json"
    }
)

# Remove only stale coordinator state after confirming no matching coordinator is alive.
foreach ($job in $jobs) {
    $pidPath = Join-Path $controlRoot "$($job.name).pid"
    if (Test-Path -LiteralPath $pidPath) {
        $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
        $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
        if ($old -and $old.Name -eq "powershell.exe" -and $old.CommandLine -like "*$($job.script)*") {
            throw "$($job.name) is already running with PID $oldPid"
        }
    }
}
foreach ($job in $jobs) {
    Remove-Item -LiteralPath $job.progress -Force -ErrorAction SilentlyContinue
}
$launched = @()
foreach ($job in $jobs) {
    $pidPath = Join-Path $controlRoot "$($job.name).pid"
    $metaPath = Join-Path $controlRoot "$($job.name).meta.json"
    $stdoutPath = Join-Path $controlRoot "$($job.name).stdout.log"
    $stderrPath = Join-Path $controlRoot "$($job.name).stderr.log"
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $job.script)
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WorkingDirectory $repoRoot -WindowStyle Hidden         -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $proc.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
    $meta = [ordered]@{
        pid = $proc.Id
        started_at = (Get-Date).ToString("o")
        command = "powershell.exe $($args -join ' ')"
        progress_file = $job.progress
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        preflight = Join-Path $controlRoot "ard100_restart_preflight.json"
    }
    $meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
    $launched += $meta
}
$launched | ConvertTo-Json -Depth 4
