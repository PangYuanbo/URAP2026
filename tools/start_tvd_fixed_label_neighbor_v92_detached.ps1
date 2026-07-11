$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$script = Join-Path $repo 'tools\run_tvd_fixed_label_neighbor_v92.py'
$runDir = Join-Path $repo 'artifacts\detached_tvd_fixed_label_neighbor_v92'
$pidFile = Join-Path $runDir 'pid.txt'
$stdout = Join-Path $runDir 'stdout.log'
$stderr = Join-Path $runDir 'stderr.log'
$meta = Join-Path $runDir 'start_meta.json'

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
if (Test-Path -LiteralPath $pidFile) {
    $existingPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Output "ALREADY RUNNING: PID=$existingPid"
        Write-Output "COMMAND=$($existing.CommandLine)"
        exit 0
    }
}
Set-Content -LiteralPath $stdout -Value '' -Encoding utf8
Set-Content -LiteralPath $stderr -Value '' -Encoding utf8
$started = Get-Date
$process = Start-Process -FilePath $python -ArgumentList @($script) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
@{ pid=$process.Id; start_time=$started.ToString('o'); command_line="$python $script"; stdout=$stdout; stderr=$stderr } | ConvertTo-Json | Set-Content -LiteralPath $meta -Encoding utf8
Write-Output "STARTED: PID=$($process.Id)"
Write-Output "STDOUT=$stdout"
Write-Output "STDERR=$stderr"
