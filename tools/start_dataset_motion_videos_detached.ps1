param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [ValidateSet('run', 'smoke')][string]$Mode = 'run',
    [int]$SmokeFrames = 90,
    [switch]$Resume,
    [string]$Python = 'C:/Users/aaron/AppData/Local/Programs/Python/Python311/python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$configuration = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$runRoot = $configuration.run_dir
$scriptPath = Join-Path $repoRoot 'optical_flow_advanced/dataset_videos.py'
$stateName = if ($Mode -eq 'smoke') { 'smoke_progress.json' } else { 'progress.json' }
$statePath = Join-Path $runRoot $stateName
$metadataPath = Join-Path $runRoot ($Mode + '.json')
$pidPath = Join-Path $runRoot ($Mode + '.pid')
$observedAt = Get-Date -Format o
$existingProcesses = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains('dataset_videos.py') -and $_.CommandLine.Contains($manifestPath)
})
if ($existingProcesses.Count -gt 0) {
    throw ('Matching run already has live processes: ' + (($existingProcesses | ForEach-Object { $_.ProcessId }) -join ', '))
}
if (Test-Path -LiteralPath $statePath) {
    $previous = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    Write-Output "NOT RUNNING observed=$observedAt previous_phase=$($previous.phase) done=$($previous.done)/$($previous.total)"
    if (-not $Resume) { throw 'Existing state requires explicit -Resume; no automatic restart.' }
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'STOP_REQUESTED')) {
    throw 'STOP_REQUESTED exists. Clear that exact file explicitly before resuming.'
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Python unavailable: $Python" }
$launchId = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $runRoot ($Mode + '_' + $launchId + '.stdout.log')
$stderrPath = Join-Path $runRoot ($Mode + '_' + $launchId + '.stderr.log')
$arguments = @('-u', ('"' + $scriptPath + '"'), $Mode, '--manifest', ('"' + $manifestPath + '"'))
if ($Mode -eq 'smoke') { $arguments += @('--max-frames', $SmokeFrames) }
if ($Resume) { $arguments += '--resume' }
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date -Format o)
    previous_stop_observed_at = $(if ($Resume) { $observedAt } else { $null })
    explicit_resume = [bool]$Resume
    mode = $Mode
    manifest = $manifestPath
    command = ($Python + ' ' + ($arguments -join ' '))
    stdout = $stdoutPath
    stderr = $stderrPath
    progress = $statePath
    output_root = $configuration.output_root
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot ($Mode + '_' + $launchId + '.json')) -Encoding UTF8
Write-Output "LAUNCHED pid=$($process.Id) start=$($metadata.started_at) mode=$Mode resume=$Resume"
Write-Output "stdout=$stdoutPath"
Write-Output "stderr=$stderrPath"
Write-Output 'Launch is not proof of progress; use monitor_dataset_motion_videos.ps1.'
