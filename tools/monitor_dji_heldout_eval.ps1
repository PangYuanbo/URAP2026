param(
    [Parameter(Mandatory = $true)]
    [string]$LogDir,
    [int]$Tail = 40
)

$ErrorActionPreference = "Stop"

$pidFile = Join-Path $LogDir "pid.txt"
$stdout = Join-Path $LogDir "stdout.log"
$stderr = Join-Path $LogDir "stderr.log"

if (-not (Test-Path $pidFile)) {
    throw "PID file not found: $pidFile"
}

$pidValue = (Get-Content $pidFile | Select-Object -First 1).Trim()
$proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
if ($proc) {
    Write-Host "RUNNING pid=$pidValue started=$($proc.StartTime)"
}
else {
    Write-Host "NOT RUNNING pid=$pidValue"
}

Write-Host "LogDir: $LogDir"
if (Test-Path $stdout) {
    Write-Host "--- stdout tail ---"
    Get-Content $stdout -Tail $Tail
}
if (Test-Path $stderr) {
    Write-Host "--- stderr tail ---"
    Get-Content $stderr -Tail $Tail
}
