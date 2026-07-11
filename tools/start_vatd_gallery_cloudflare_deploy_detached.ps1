param(
    [string]$ProjectName = "vatd-trajectory-gallery"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$site = (Resolve-Path (Join-Path $repo "artifacts\vatd_track_visualization")).Path
$runDir = Join-Path $repo "artifacts\cloudflare_deploy\vatd-trajectory-gallery"
$pidPath = Join-Path $runDir "deploy.pid"
$stdoutPath = Join-Path $runDir "deploy.stdout.log"
$stderrPath = Join-Path $runDir "deploy.stderr.log"

New-Item -ItemType Directory -Path $runDir -Force | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -match "wrangler.*pages deploy") {
        throw "Deployment is already running with PID $existingPid."
    }
}

$npx = (Get-Command npx.cmd -ErrorAction Stop).Source
$arguments = @(
    "--yes",
    "wrangler@latest",
    "pages",
    "deploy",
    $site,
    "--project-name",
    $ProjectName,
    "--branch",
    "main",
    "--commit-dirty=true"
)

$process = Start-Process -FilePath $npx -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii

[pscustomobject]@{
    Status = "STARTED"
    PID = $process.Id
    StartedAt = $process.StartTime
    Site = $site
    Stdout = $stdoutPath
    Stderr = $stderrPath
    Project = $ProjectName
} | Format-List
