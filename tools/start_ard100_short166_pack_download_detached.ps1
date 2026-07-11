param([string]$RunName = "ard100_short166_pack_download_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$root = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stdout = Join-Path $logs "$RunName.stdout.log"; $stderr = Join-Path $logs "$RunName.stderr.log"
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "sequence_ard100_short166_pack_download.ps1"), "-RunName", $RunName)
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $root "$RunName.pid") -Encoding ascii
$meta = [ordered]@{ pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "powershell.exe $($arguments -join ' ')"; target = "D:\URAP_local_datasets\ARD100_SAMURAI_SHORT166"; stdout_log = $stdout; stderr_log = $stderr }
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root "$RunName.meta.json") -Encoding utf8
$meta | ConvertTo-Json -Depth 4
