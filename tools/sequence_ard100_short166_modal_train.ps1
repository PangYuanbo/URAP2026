param([string]$RunName = "ard100_short166_modal_train_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$statePath = Join-Path $controlRoot "state.json"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null

function Write-State([string]$Stage, [hashtable]$Extra = @{}) {
    $payload = [ordered]@{ stage = $Stage; updated_at = (Get-Date).ToString("o") }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding utf8
}

function Wait-ModalRun([string]$Mode) {
    $metadata = & (Join-Path $PSScriptRoot "start_ard100_short166_modal_train_detached.ps1") -Mode $Mode -RunName $RunName | ConvertFrom-Json
    Write-State "${Mode}_running" @{ pid = $metadata.pid; stdout_log = $metadata.stdout_log; stderr_log = $metadata.stderr_log }
    while ($true) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($metadata.pid)" -ErrorAction SilentlyContinue
        if (-not ($process -and $process.CommandLine -like "*modal_train_ard100_short166.py*")) { break }
        $lastFile = Get-ChildItem $metadata.stdout_log, $metadata.stderr_log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-State "${Mode}_running" @{ pid = $metadata.pid; last_output_timestamp = if ($lastFile) { $lastFile.LastWriteTime.ToString("o") } else { $null }; stdout_log = $metadata.stdout_log; stderr_log = $metadata.stderr_log }
        Start-Sleep -Seconds 30
    }
    $stdout = if (Test-Path -LiteralPath $metadata.stdout_log) { Get-Content -LiteralPath $metadata.stdout_log -Raw } else { "" }
    $stderr = Get-Item -LiteralPath $metadata.stderr_log -ErrorAction SilentlyContinue
    if (($stderr -and $stderr.Length -gt 0) -or $stdout -notmatch 'App completed' -or $stdout -notmatch '"status"\s*:\s*"completed"') {
        Write-State "${Mode}_failed" @{ pid = $metadata.pid; stdout_log = $metadata.stdout_log; stderr_log = $metadata.stderr_log }
        throw "ARD100 short166 Modal $Mode failed"
    }
    Write-State "${Mode}_completed" @{ pid = $metadata.pid; stdout_log = $metadata.stdout_log; stderr_log = $metadata.stderr_log }
}

Wait-ModalRun "smoke"
Wait-ModalRun "train"
Write-State "completed" @{ done = 2; total = 2; results_volume = "urap-ard100-samurai-short166-results-v1" }
