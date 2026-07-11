param([string]$Destination='D:\URAP_vatd_rank_inputs')
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8';$env:TERM='dumb'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$progress=Join-Path $Destination 'download_progress.json'
$items=@(
  @{remote='/aot_indep/saliency_tracklets/proposal_tracklets.jsonl';local='aot_proposal_tracklets.jsonl'},
  @{remote='/aot_indep/gt.csv';local='aot_gt.csv'},
  @{remote='/nps/vatd/tracklets_with_vatd.jsonl';local='nps_tracklets_with_vatd.jsonl'},
  @{remote='/nps/predictionsgt_split_0.pkl';local='nps_predictionsgt_split_0.pkl'}
)
for($index=0;$index -lt $items.Count;$index++){
  $item=$items[$index];$target=Join-Path $Destination $item.local
  @{stage='download';done=$index;total=$items.Count;remote=$item.remote;target=$target;updated=(Get-Date -Format o)}|ConvertTo-Json|Set-Content -LiteralPath $progress
  & 'C:\Users\aaron\.local\bin\modal.exe' volume get vatd-artifacts $item.remote $target --force
  if($LASTEXITCODE -ne 0){throw "modal download failed: $($item.remote)"}
}
@{stage='done';done=$items.Count;total=$items.Count;updated=(Get-Date -Format o)}|ConvertTo-Json|Set-Content -LiteralPath $progress
