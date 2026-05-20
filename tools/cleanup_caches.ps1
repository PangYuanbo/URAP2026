param(
  [string[]]$Roots = @(
    'C:/Users/aaron/Desktop/URAP',
    'D:/URAP_datasets'
  )
)

$ErrorActionPreference = 'Stop'

function Get-DirBytes([string]$Path) {
  $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum).Sum
  if ($null -eq $sum) { return 0 }
  return $sum
}

$before = Get-PSDrive -PSProvider FileSystem |
  Select-Object Name, Root, @{ n = 'FreeGB'; e = { [math]::Round(($_.Free / 1GB), 2) } }

$dirs = foreach ($root in $Roots) {
  if (Test-Path $root) {
    Get-ChildItem -LiteralPath $root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -in @('.cache', '__pycache__', '.pytest_cache', '.mypy_cache') }
  }
}

$files = foreach ($root in $Roots) {
  if (Test-Path $root) {
    Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Extension -in @('.pyc', '.pyo', '.cache') }
  }
}

$dirObjs = foreach ($d in $dirs) {
  [pscustomobject]@{
    Path = $d.FullName
    Size = Get-DirBytes $d.FullName
  }
}

$fileObjs = foreach ($f in $files) {
  [pscustomobject]@{
    Path = $f.FullName
    Size = $f.Length
  }
}

$targetBytes = (($dirObjs | Measure-Object Size -Sum).Sum + ($fileObjs | Measure-Object Size -Sum).Sum)

Write-Host ("DIR_COUNT={0}" -f $dirObjs.Count)
Write-Host ("FILE_COUNT={0}" -f $fileObjs.Count)
Write-Host ("TARGET_GB={0}" -f [math]::Round(($targetBytes / 1GB), 3))
Write-Host "TOP_DIRS="
$dirObjs |
  Sort-Object Size -Descending |
  Select-Object -First 20 @{ n = 'SizeGB'; e = { [math]::Round(($_.Size / 1GB), 3) } }, Path |
  Format-Table -AutoSize

foreach ($f in $fileObjs) {
  Remove-Item -LiteralPath $f.Path -Force -ErrorAction SilentlyContinue
}

foreach ($d in ($dirObjs | Sort-Object { $_.Path.Length } -Descending)) {
  Remove-Item -LiteralPath $d.Path -Recurse -Force -ErrorAction SilentlyContinue
}

$after = Get-PSDrive -PSProvider FileSystem |
  Select-Object Name, Root, @{ n = 'FreeGB'; e = { [math]::Round(($_.Free / 1GB), 2) } }

Write-Host "AFTER_FREE="
$after | Format-Table -AutoSize
