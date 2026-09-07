param([Parameter(Mandatory = $true)][string]$Manifest)

$ErrorActionPreference = 'Stop'
$configuration = Get-Content -LiteralPath (Resolve-Path -LiteralPath $Manifest).Path -Raw | ConvertFrom-Json
$stopPath = Join-Path $configuration.run_dir 'STOP_REQUESTED'
Get-Date -Format o | Set-Content -LiteralPath $stopPath -Encoding ASCII
Write-Output "Stop requested: $stopPath. This is not confirmation of termination; monitor the recorded PIDs."
Write-Output 'Completed sequences remain intact. Partial sequences require an explicit resume from their beginning.'
