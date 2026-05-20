param(
  [string]$AutoresearchDir = "C:\Users\aaron\Desktop\URAP\autoresearch",
  [string]$MainRepoDir = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking",
  [string]$RunTag,
  [string]$RunDir,
  [string]$MirrorDir,
  [int]$MaxRounds = 6,
  [switch]$LocalOnly,
  [string]$AgendaFile = "",
  [string]$Model = "gpt-5.4",
  [string]$Reasoning = "xhigh",
  [string]$CodexExe = "codex.cmd"
)

$ErrorActionPreference = "Stop"

if (-not $RunTag) { throw "RunTag is required." }
if (-not $RunDir) { throw "RunDir is required." }
if (-not $MirrorDir) { throw "MirrorDir is required." }
if (-not (Test-Path -Path $AutoresearchDir -PathType Container)) { throw "AutoresearchDir not found: $AutoresearchDir" }
if (-not (Test-Path -Path $MainRepoDir -PathType Container)) { throw "MainRepoDir not found: $MainRepoDir" }

$programPath = Join-Path $AutoresearchDir "program.md"
$contractPath = Join-Path $AutoresearchDir "contracts\research_output_contract.md"
$schemaPath = Join-Path $AutoresearchDir "contracts\research_state.schema.json"
$projectAgentsPath = Join-Path $MainRepoDir "AGENTS.md"

foreach ($required in @($programPath, $contractPath, $schemaPath, $projectAgentsPath)) {
  if (-not (Test-Path -Path $required -PathType Leaf)) { throw "Required file not found: $required" }
}
if (-not [string]::IsNullOrWhiteSpace($AgendaFile)) {
  if (-not (Test-Path -Path $AgendaFile -PathType Leaf)) { throw "AgendaFile not found: $AgendaFile" }
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
New-Item -ItemType Directory -Force -Path $MirrorDir | Out-Null

$logsDir = Join-Path $RunDir "logs"
$contractsDir = Join-Path $RunDir "contracts"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $contractsDir | Out-Null

Copy-Item -Force -Path $programPath -Destination (Join-Path $RunDir "program.md")
Copy-Item -Force -Path $contractPath -Destination (Join-Path $contractsDir "research_output_contract.md")
Copy-Item -Force -Path $schemaPath -Destination (Join-Path $contractsDir "research_state.schema.json")
Copy-Item -Force -Path $projectAgentsPath -Destination (Join-Path $RunDir "PROJECT_AGENTS.md")
if (-not [string]::IsNullOrWhiteSpace($AgendaFile)) {
  Copy-Item -Force -Path $AgendaFile -Destination (Join-Path $RunDir "AGENDA.md")
}

$contextFiles = @(
  (Join-Path $MainRepoDir "AGENTS.md"),
  (Join-Path $MainRepoDir "doc\official_datasets_and_metrics.md"),
  (Join-Path $MainRepoDir "doc\progress_report_for_professor_en.md"),
  (Join-Path $MainRepoDir "doc\winner_vs_tvd_aot_nps_analysis.md"),
  (Join-Path $MainRepoDir "doc\repro_transvisdrone.md"),
  (Join-Path $MainRepoDir "doc\transvisdrone_method_explained.md"),
  (Join-Path $MainRepoDir "doc\v1_plan_edge_tiny_uav_urban.md"),
  (Join-Path $MainRepoDir "doc\index.json")
) | Where-Object { Test-Path -Path $_ -PathType Leaf }

$contextManifestPath = Join-Path $RunDir "local_context_manifest.md"
$contextLines = @("# Local Context Manifest", "", "The launcher expects every round to read these files first.", "")
$contextLines += $contextFiles | ForEach-Object { "- $_" }
if (-not [string]::IsNullOrWhiteSpace($AgendaFile)) {
  $contextLines += ""
  $contextLines += "Focused agenda file:"
  $contextLines += "- $(Join-Path $RunDir "AGENDA.md")"
}
$contextLines | Set-Content -Encoding utf8 -Path $contextManifestPath

$statusPath = Join-Path $RunDir "run_status.json"
$consolidatedReport = Join-Path $RunDir "consolidated_report.md"

function Quote-CmdArg {
  param([string]$Value)
  if ($null -eq $Value) { return '""' }
  if ($Value -notmatch '[\s"]') { return $Value }
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Write-RunStatus {
  param(
    [string]$State,
    [int]$DoneRounds,
    [int]$CurrentRound,
    [string]$LastArtifactPath,
    [string]$LastLogPath,
    [string]$Message
  )

  $payload = [ordered]@{
    run_tag = $RunTag
    state = $State
    done_rounds = $DoneRounds
    total_rounds = $MaxRounds
    current_round = $CurrentRound
    last_updated = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    last_artifact_path = $LastArtifactPath
    last_log_path = $LastLogPath
    mirror_dir = $MirrorDir
    message = $Message
  }
  ($payload | ConvertTo-Json -Depth 6) | Set-Content -Encoding utf8 -Path $statusPath
}

function Copy-RoundArtifacts {
  param(
    [string]$SourceRoundDir,
    [string]$DestRoundDir
  )

  New-Item -ItemType Directory -Force -Path $DestRoundDir | Out-Null
  foreach ($name in @("round_summary.md", "paper_matrix.tsv", "recommended_ablation_plan.md", "rejected_ideas.md", "research_state.json")) {
    $src = Join-Path $SourceRoundDir $name
    if (Test-Path -Path $src -PathType Leaf) {
      Copy-Item -Force -Path $src -Destination (Join-Path $DestRoundDir $name)
    }
  }
  if (Test-Path -Path $consolidatedReport -PathType Leaf) {
    Copy-Item -Force -Path $consolidatedReport -Destination (Join-Path $MirrorDir "consolidated_report.md")
  }
}

$searchEnabled = -not $LocalOnly
Write-RunStatus -State "starting" -DoneRounds 0 -CurrentRound 0 -LastArtifactPath "" -LastLogPath "" -Message "Initializing Codex-backed autoresearch run."

for ($round = 0; $round -lt $MaxRounds; $round++) {
  $roundName = "round_{0:D2}" -f $round
  $roundDir = Join-Path $RunDir $roundName
  $mirrorRoundDir = Join-Path $MirrorDir $roundName
  New-Item -ItemType Directory -Force -Path $roundDir | Out-Null

  $promptPath = Join-Path $roundDir "prompt.md"
  $lastMessagePath = Join-Path $roundDir "codex_last_message.txt"
  $stdoutPath = Join-Path $logsDir ("{0}.out.txt" -f $roundName)
  $stderrPath = Join-Path $logsDir ("{0}.err.txt" -f $roundName)
  $prevState = if ($round -gt 0) { Join-Path (Join-Path $RunDir ("round_{0:D2}" -f ($round - 1))) "research_state.json" } else { "" }

  $researchModeLine = if ($searchEnabled) {
    "Use local project materials first and live web search second. Verify unstable or recent claims with live search."
  } else {
    "Local-only mode: do not use live web search. Use only local project materials already present in the workspace."
  }

  $prevStateLine = if ([string]::IsNullOrWhiteSpace($prevState)) {
    "There is no previous state file for this round."
  } else {
    "Read the previous round state first: $prevState"
  }
  $agendaLine = if ([string]::IsNullOrWhiteSpace($AgendaFile)) {
    ""
  } else {
    "- Read the focused agenda addendum: $(Join-Path $RunDir "AGENDA.md")"
  }

  $prompt = @"
You are running one research round for the URAP project.

Mission:
- Research tiny UAV / tiny obstacle detection in complex urban backgrounds.
- Training can be heavy, but inference must remain realistic for onboard UAV deployment.
- Favor TransVisDrone as the base system and prioritize lightweight add-ons that improve AOT and NPS behavior.

Research mode:
- $researchModeLine
- Read the local context manifest first: $contextManifestPath
- Read the local project rules copy: $(Join-Path $RunDir "PROJECT_AGENTS.md")
- Read the charter: $(Join-Path $RunDir "program.md")
- Read the output contract: $(Join-Path $contractsDir "research_output_contract.md")
- Read the JSON schema: $(Join-Path $contractsDir "research_state.schema.json")
$agendaLine
- $prevStateLine

Hard output requirements for this round:
- Write these files into $roundDir
  - round_summary.md
  - paper_matrix.tsv
  - recommended_ablation_plan.md
  - rejected_ideas.md
  - research_state.json
- Update this run-level file as well:
  - $consolidatedReport

Ranking priorities:
1. Edge inference feasibility
2. Urban clutter robustness
3. Tiny-object suitability
4. Compatibility with the current TransVisDrone-centric stack
5. Cross-dataset generalization potential
6. Code availability

Required analysis questions:
1. Which recent methods improve visibility of tiny aerial objects in urban clutter with low inference overhead?
2. Which methods explicitly suppress distractors such as windows, wires, roof edges, reflections, and structured backgrounds?
3. Which temporal/video modules help when both camera and target move?
4. Which selective ROI refinement methods are edge-feasible?
5. Which Winner v022 ideas are transferable into TransVisDrone without collapsing generalization?
6. Which ideas should be rejected because they are too heavy, too benchmark-specific, or too brittle?

Constraints:
- Research only. Do not edit training code, evaluation code, or dataset configs.
- Do not modify any existing docs outside this run directory.
- If you mention a paper or method, include a source link in markdown.
- Clearly mark any inference that is not directly stated by a source.
- Keep recommendations decision-oriented and tied to AOT and NPS experiments.
- The ablation plan must be incremental and one change at a time.

Round id:
- $roundName

When finished, ensure every required file exists and is internally consistent.
"@

  $prompt | Set-Content -Encoding utf8 -Path $promptPath

  Write-RunStatus -State "running" -DoneRounds $round -CurrentRound $round -LastArtifactPath $roundDir -LastLogPath $stdoutPath -Message ("Running {0}" -f $roundName)

  $codexArgs = @()
  if ($searchEnabled) { $codexArgs += "--search" }
  $codexArgs += @(
    "-a", "never",
    "exec",
    "-C", $RunDir,
    "--sandbox", "workspace-write",
    "--add-dir", $MainRepoDir,
    "-m", $Model,
    "-c", ("model_reasoning_effort=`"{0}`"" -f $Reasoning),
    "--output-last-message", $lastMessagePath,
    "--json",
    "-"
  )

  $quotedCodexArgs = ($codexArgs | ForEach-Object { Quote-CmdArg $_ }) -join " "
  $cmdLine = ('"{0}" {1} < "{2}" 1>>"{3}" 2>>"{4}"' -f $CodexExe, $quotedCodexArgs, $promptPath, $stdoutPath, $stderrPath)
  & cmd.exe /d /c $cmdLine
  $exitCode = $LASTEXITCODE

  if ($exitCode -ne 0) {
    Write-RunStatus -State "failed" -DoneRounds $round -CurrentRound $round -LastArtifactPath $roundDir -LastLogPath $stderrPath -Message ("Codex exited with code {0}" -f $exitCode)
    throw "Codex round failed: $roundName (exit code $exitCode)"
  }

  $requiredArtifacts = @(
    (Join-Path $roundDir "round_summary.md"),
    (Join-Path $roundDir "paper_matrix.tsv"),
    (Join-Path $roundDir "recommended_ablation_plan.md"),
    (Join-Path $roundDir "rejected_ideas.md"),
    (Join-Path $roundDir "research_state.json")
  )

  foreach ($artifact in $requiredArtifacts) {
    if (-not (Test-Path -Path $artifact -PathType Leaf)) {
      Write-RunStatus -State "failed" -DoneRounds $round -CurrentRound $round -LastArtifactPath $artifact -LastLogPath $stderrPath -Message ("Missing required artifact: {0}" -f $artifact)
      throw "Missing required artifact after ${roundName}: $artifact"
    }
  }

  Copy-RoundArtifacts -SourceRoundDir $roundDir -DestRoundDir $mirrorRoundDir
  Write-RunStatus -State "running" -DoneRounds ($round + 1) -CurrentRound ($round + 1) -LastArtifactPath (Join-Path $roundDir "research_state.json") -LastLogPath $stdoutPath -Message ("Completed {0}" -f $roundName)
}

Write-RunStatus -State "completed" -DoneRounds $MaxRounds -CurrentRound $MaxRounds -LastArtifactPath $consolidatedReport -LastLogPath (Join-Path $logsDir ("round_{0:D2}.out.txt" -f ($MaxRounds - 1))) -Message "Autoresearch run completed."
