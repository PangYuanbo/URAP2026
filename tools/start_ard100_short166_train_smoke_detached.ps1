param([string]$RunName = "ard100_short166_train_smoke_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$root = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$pidPath = Join-Path $root "$RunName.pid"
$metaPath = Join-Path $root "$RunName.meta.json"
$stdout = Join-Path $logs "$RunName.stdout.log"
$stderr = Join-Path $logs "$RunName.stderr.log"
$modal = Join-Path $env:USERPROFILE ".local\bin\modal.exe"
$args = @("run", "tools\modal_train_ard100_short166.py", "--smoke-only")
$oldUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = "1"
$process = Start-Process -FilePath $modal -ArgumentList $args -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$env:PYTHONUTF8 = $oldUtf8
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$meta = [ordered]@{ pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "$modal $($args -join ' ')"; expected_sequences = 1; stdout_log = $stdout; stderr_log = $stderr }
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$meta | ConvertTo-Json -Depth 4
