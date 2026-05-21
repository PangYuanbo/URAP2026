param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [string]$Scenario = "unlabeled",
    [string]$OutRoot = "data\real\motion_debug",
    [int[]]$KValues = @(1, 2, 4)
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$ClipName = [System.IO.Path]::GetFileNameWithoutExtension($Video)
$Out = Join-Path (Join-Path $OutRoot $Scenario) $ClipName

python -m qstr_dronedet.cli motion-debug `
    --video $Video `
    --out $Out `
    --k-values $KValues

Write-Host "Motion debug output: $Out"
