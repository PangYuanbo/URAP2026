param(
 [int]$BatchSize=16, [int]$ImgSz=1280
)
$corrected = Join-Path $PSScriptRoot 'start_ard100_yolomg_corrected_detached.ps1'
& $corrected -BatchSize $BatchSize -ImgSz $ImgSz
