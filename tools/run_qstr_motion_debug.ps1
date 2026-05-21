param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [string]$Out = "runs\motion_debug",
    [int[]]$KValues = @(1, 2, 4)
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

python -m qstr_dronedet.cli motion-debug `
    --video $Video `
    --out $Out `
    --k-values $KValues
