$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repo "artifacts\cloudflare_deploy\vatd-trajectory-gallery"
$pidPath = Join-Path $runDir "deploy.pid"
$stdoutPath = Join-Path $runDir "deploy.stdout.log"
$stderrPath = Join-Path $runDir "deploy.stderr.log"

$pidValue = if (Test-Path -LiteralPath $pidPath) { [int](Get-Content -LiteralPath $pidPath -Raw).Trim() } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
$stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { "" }
$stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
$combined = $stdout + "`n" + $stderr
$matches = [regex]::Matches($combined, "(?m)(\d+)\s*/\s*(\d+)")
$progress = if ($matches.Count) { $matches[$matches.Count - 1] } else { $null }
$urlMatch = [regex]::Matches($combined, "https://[^\s]+\.pages\.dev")
$lastFile = Get-Item -LiteralPath @($stdoutPath, $stderrPath) -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1

[pscustomobject]@{
    Status = if ($process) { "RUNNING" } elseif ($combined -match "Deployment complete|✨ Deployment complete|Success") { "COMPLETED" } else { "NOT RUNNING" }
    Done = if ($progress) { [int]$progress.Groups[1].Value } else { $null }
    Total = if ($progress) { [int]$progress.Groups[2].Value } else { $null }
    PID = $pidValue
    StartTime = if ($process) { $process.CreationDate } else { $null }
    LastOutputTimestamp = $lastFile.LastWriteTime
    URL = if ($urlMatch.Count) { $urlMatch[$urlMatch.Count - 1].Value } else { $null }
    Stdout = $stdoutPath
    Stderr = $stderrPath
} | Format-List

if ($stdout) {
    Write-Output "--- stdout tail ---"
    Get-Content -LiteralPath $stdoutPath -Tail 20
}
if ($stderr) {
    Write-Output "--- stderr tail ---"
    Get-Content -LiteralPath $stderrPath -Tail 20
}
