param(
  [string]$RunName = "two_branch_motion_action_yolomg_20260606"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path ".").Path
$RunDir = Join-Path $Repo ("artifacts\yolomg_action\" + $RunName)
$PidFile = Join-Path $RunDir "runner.pid"
$StatusFile = Join-Path $RunDir "status.json"
$Stdout = Join-Path $RunDir "logs\runner.out.txt"
$Stderr = Join-Path $RunDir "logs\runner.err.txt"
$PidValue = if (Test-Path $PidFile) { [int](Get-Content $PidFile -Raw) } else { $null }
$Proc = if ($PidValue) { Get-CimInstance Win32_Process -Filter "ProcessId=$PidValue" -ErrorAction SilentlyContinue } else { $null }
$Status = if (Test-Path $StatusFile) { Get-Content $StatusFile -Raw | ConvertFrom-Json } else { $null }
$OutInfo = if (Test-Path $Stdout) { Get-Item $Stdout } else { $null }

[pscustomobject]@{
  RunName = $RunName
  PID = $PidValue
  Running = [bool]$Proc
  CommandLine = if ($Proc) { $Proc.CommandLine } else { $null }
  State = if ($Status) { $Status.state } else { $null }
  Step = if ($Status) { $Status.step } else { $null }
  Done = if ($Status) { $Status.done } else { $null }
  Total = if ($Status) { $Status.total } else { $null }
  LastOutputTimestamp = if ($OutInfo) { $OutInfo.LastWriteTime } else { $null }
  Stdout = $Stdout
  Stderr = $Stderr
  Status = $StatusFile
}
if (Test-Path (Join-Path $RunDir "summary.json")) {
  "Summary:"
  Get-Content (Join-Path $RunDir "summary.json") -Raw
}
if (Test-Path $Stderr) {
  "Stderr tail:"
  Get-Content $Stderr -Tail 20
}
