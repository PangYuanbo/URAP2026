param(
  [string]$RepoRoot,
  [string]$ProjectRoot,
  [string]$ARDRoot,
  [string]$ARDSplitRoot,
  [int]$Epochs,
  [int]$BatchSize,
  [int]$FrameStride,
  [int]$MaxFramesPerVideo,
  [int]$EmptyStride,
  [double]$LR,
  [int]$MinSize,
  [int]$MaxSize,
  [int]$NumWorkers,
  [string]$AnchorPreset,
  [string]$Checkpoint,
  [string]$ValJson,
  [string]$TestJson,
  [string]$StateFile
)

$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) { throw "Python exe not found: $pythonExe" }
$currentStageIndex = 0
$currentStageName = 'train'

function Set-State(
  [int]$StageIndex,
  [string]$StageName,
  [string]$Status,
  [string]$LastCompleted,
  [string]$ErrorMessage='',
  [string]$CurrentStdout='',
  [string]$CurrentStderr='',
  [int]$CurrentPid=0
) {
  @{
    stage_index = $StageIndex
    stage_total = 3
    stage_name = $StageName
    status = $Status
    last_completed_unit = $LastCompleted
    checkpoint = $Checkpoint
    val_json = $ValJson
    test_json = $TestJson
    updated_at = (Get-Date).ToString('s')
    error = $ErrorMessage
    current_stdout = $CurrentStdout
    current_stderr = $CurrentStderr
    current_pid = $CurrentPid
  } | ConvertTo-Json | Set-Content -Path $StateFile -Encoding UTF8
}

function Start-NativeLogged([string]$Label,[object[]]$CommandArgs) {
  $stageRoot = Join-Path (Split-Path $StateFile -Parent) 'stage_logs'
  New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
  $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
  $stdout = Join-Path $stageRoot ("{0}_{1}.out.txt" -f $Label, $ts)
  $stderr = Join-Path $stageRoot ("{0}_{1}.err.txt" -f $Label, $ts)
  $argList = @($CommandArgs | Where-Object { $null -ne $_ -and [string]$_ -ne '' } | ForEach-Object { [string]$_ })
  if ($argList.Count -eq 0) { throw "No arguments generated for stage $Label" }
  $proc = Start-Process -FilePath $pythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
  return @{
    Proc = $proc
    Stdout = $stdout
    Stderr = $stderr
  }
}

try {
  Push-Location $ProjectRoot

  $currentStageIndex = 0
  $currentStageName = 'train'
  $trainRun = Start-NativeLogged -Label 'train' -CommandArgs @('-u','train_detector.py','--repo-root',$RepoRoot,'--dataset','ard100','--ard-root',$ARDRoot,'--ard-split-root',$ARDSplitRoot,'--ard-split','train','--frame-stride',$FrameStride,'--include-empty','--empty-stride',$EmptyStride,'--epochs',$Epochs,'--batch-size',$BatchSize,'--lr',$LR,'--min-size',$MinSize,'--max-size',$MaxSize,'--num-workers',$NumWorkers,'--anchor-preset',$AnchorPreset,'--out',$Checkpoint,'--amp')
  Set-State -StageIndex 0 -StageName 'train' -Status 'running' -LastCompleted 'starting train' -CurrentStdout $trainRun.Stdout -CurrentStderr $trainRun.Stderr -CurrentPid $trainRun.Proc.Id
  $trainRun.Proc.WaitForExit()
  try { $trainRun.Proc.Refresh() } catch {}
  $trainExitCode = if ($null -eq $trainRun.Proc.ExitCode) { 0 } else { [int]$trainRun.Proc.ExitCode }
  if ($trainExitCode -ne 0) { throw "train_detector.py exited with code $trainExitCode" }

  $currentStageIndex = 1
  $currentStageName = 'val'
  $valRun = Start-NativeLogged -Label 'val' -CommandArgs @('-u','eval_detector.py','--repo-root',$RepoRoot,'--dataset','ard100','--ard-root',$ARDRoot,'--ard-split-root',$ARDSplitRoot,'--ard-split','val','--ckpt',$Checkpoint,'--frame-stride',$FrameStride,'--include-empty','--empty-stride',$EmptyStride,'--batch-size','1','--num-workers',$NumWorkers,'--min-size',$MinSize,'--max-size',$MaxSize,'--iou','0.5','--scores','0.1','0.2','0.3','0.4','0.5','--out-json',$ValJson)
  Set-State -StageIndex 1 -StageName 'val' -Status 'running' -LastCompleted 'train complete' -CurrentStdout $valRun.Stdout -CurrentStderr $valRun.Stderr -CurrentPid $valRun.Proc.Id
  $valRun.Proc.WaitForExit()
  try { $valRun.Proc.Refresh() } catch {}
  $valExitCode = if ($null -eq $valRun.Proc.ExitCode) { 0 } else { [int]$valRun.Proc.ExitCode }
  if ($valExitCode -ne 0) { throw "eval_detector.py (val) exited with code $valExitCode" }

  $currentStageIndex = 2
  $currentStageName = 'test'
  $testRun = Start-NativeLogged -Label 'test' -CommandArgs @('-u','eval_detector.py','--repo-root',$RepoRoot,'--dataset','ard100','--ard-root',$ARDRoot,'--ard-split-root',$ARDSplitRoot,'--ard-split','test','--ckpt',$Checkpoint,'--frame-stride',$FrameStride,'--include-empty','--empty-stride',$EmptyStride,'--batch-size','1','--num-workers',$NumWorkers,'--min-size',$MinSize,'--max-size',$MaxSize,'--iou','0.5','--scores','0.1','0.2','0.3','0.4','0.5','--out-json',$TestJson)
  Set-State -StageIndex 2 -StageName 'test' -Status 'running' -LastCompleted 'val complete' -CurrentStdout $testRun.Stdout -CurrentStderr $testRun.Stderr -CurrentPid $testRun.Proc.Id
  $testRun.Proc.WaitForExit()
  try { $testRun.Proc.Refresh() } catch {}
  $testExitCode = if ($null -eq $testRun.Proc.ExitCode) { 0 } else { [int]$testRun.Proc.ExitCode }
  if ($testExitCode -ne 0) { throw "eval_detector.py (test) exited with code $testExitCode" }

  $currentStageIndex = 3
  $currentStageName = 'done'
  Set-State -StageIndex 3 -StageName 'done' -Status 'completed' -LastCompleted 'test complete' -CurrentStdout $testRun.Stdout -CurrentStderr $testRun.Stderr -CurrentPid $testRun.Proc.Id
}
catch {
  $msg = $_.Exception.Message
  Set-State -StageIndex $currentStageIndex -StageName $currentStageName -Status 'failed' -LastCompleted 'pipeline failed' -ErrorMessage $msg
  throw
}
finally {
  Pop-Location
}
