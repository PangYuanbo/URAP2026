param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$AOTRoot = "D:\URAP_datasets\AOT\part1",
  [string]$OutRoot = "D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest",
  [int]$BatchSize = 2,
  [int]$Img = 1280,
  [int]$NumFrames = 3,
  [double]$ConfThres = 0.1,
  [string]$RunNameSuffix = "wport_baseline",
  [switch]$SkipPrepare,
  [switch]$SkipInfer,
  [switch]$SkipEval,
  [string[]]$ExtraValArgs = @(),
  [string]$RunId = "aot_fulltest_conf0p1_wport_baseline",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\artifacts\route_b_official\aot_fulltest_conf0p1_runner"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*run_aot_full_test.ps1*') {
      Write-Host "AOT fulltest already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 160 }
      exit 0
    }
  }
}

$runner = Join-Path $URAPRoot "tools\run_aot_full_test.ps1"
if (-not (Test-Path -Path $runner -PathType Leaf)) { throw "Runner not found: $runner" }

$runName = "fulltest_conf{0}" -f ($ConfThres.ToString().Replace(".", "p"))
if ($RunNameSuffix) { $runName = "{0}_{1}" -f $runName, $RunNameSuffix }
$predDir = Join-Path $URAPRoot ("papers\TransVisDrone\runs\val\AOT_URAP\{0}\aotpredictions" -f $runName)
$evalDir = Join-Path $URAPRoot ("papers\TransVisDrone\runs\eval\AOT_URAP\{0}" -f $runName)

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", $runner,
  "-URAPRoot", $URAPRoot,
  "-AOTRoot", $AOTRoot,
  "-OutRoot", $OutRoot,
  "-BatchSize", [string]$BatchSize,
  "-Img", [string]$Img,
  "-NumFrames", [string]$NumFrames,
  "-ConfThres", [string]$ConfThres,
  "-RunNameSuffix", $RunNameSuffix
)
if ($SkipPrepare) { $argList += "-SkipPrepare" }
if ($SkipInfer) { $argList += "-SkipInfer" }
if ($SkipEval) { $argList += "-SkipEval" }
if ($ExtraValArgs -and $ExtraValArgs.Count -gt 0) {
  $argList += "-ExtraValArgs"
  $argList += $ExtraValArgs
}

$proc = Start-Process `
  -FilePath "powershell.exe" `
  -ArgumentList $argList `
  -WorkingDirectory $URAPRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$proc.Id | Set-Content -Encoding ascii -Path $pidFile

@(
  ("started={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
  ("pid={0}" -f $proc.Id),
  ("run_id={0}" -f $RunId),
  ("cwd={0}" -f $URAPRoot),
  ("runner={0}" -f $runner),
  ("aot_root={0}" -f $AOTRoot),
  ("out_root={0}" -f $OutRoot),
  ("batch_size={0}" -f $BatchSize),
  ("img={0}" -f $Img),
  ("num_frames={0}" -f $NumFrames),
  ("conf_thres={0}" -f $ConfThres),
  ("run_name_suffix={0}" -f $RunNameSuffix),
  ("run_name={0}" -f $runName),
  ("prediction_dir={0}" -f $predDir),
  ("evaluation_dir={0}" -f $evalDir),
  ("skip_prepare={0}" -f [bool]$SkipPrepare),
  ("skip_infer={0}" -f [bool]$SkipInfer),
  ("skip_eval={0}" -f [bool]$SkipEval),
  ("extra_val_args={0}" -f ($ExtraValArgs -join " ")),
  ("output_root={0}" -f $OutputRoot),
  ("stdout={0}" -f $stdout),
  ("stderr={0}" -f $stderr),
  ("cmd_args={0}" -f ($argList -join " "))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host "Started detached AOT fulltest."
Get-Content $metaFile
