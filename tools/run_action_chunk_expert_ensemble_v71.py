from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');RUN=REPO/'artifacts'/'detached_action_chunk_expert_ensemble_v71';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_expert_ensemble_v71')
SOURCES={'v46':(Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46'),'action_chunk_neighbor_score'),'v51':(Path(r'D:\URAP_vatd_rank_results\action_chunk_candidate_context_v51'),'action_chunk_candidate_context_score'),'v52':(Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52'),'action_chunk_multi_expert_score'),'v38':(Path(r'D:\URAP_vatd_rank_results\action_chunk_causal_v38'),'action_chunk_causal_score')};PAIRS=(('v46','v51'),('v46','v52'),('v46','v38'),('v51','v52'),('v52','v38'))
def report(stage,done,total=7,**extra):RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(payload),encoding='utf8');print(json.dumps(payload),flush=True)
def execute(stage,done,command):
 process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report(stage,done,child_pid=process.pid,command=command);code=process.wait()
 if code:raise RuntimeError(f'{stage} failed with {code}')
def score_file(root,name,split):
 if name=='v52':return root/f'{split}_expert_scores.jsonl'
 return root/(f'{split}_oof_scores.jsonl' if split=='val' else 'test_scores.jsonl')
def main():
 OUT.mkdir(parents=True,exist_ok=True);best=None;tool=REPO/'tools'/'sweep_action_chunk_two_row_ensemble.py';weights='0 .02 .04 .06 .08 .1 .14 .2 .3 .4 .55'
 for index,(left,right) in enumerate(PAIRS):
  left_root,left_field=SOURCES[left];right_root,right_field=SOURCES[right];path=OUT/f'val_{left}_{right}.json';execute(f'validate_{left}_{right}',index,[sys.executable,str(tool),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--score-a',str(score_file(left_root,left,'val')),'--field-a',left_field,'--score-b',str(score_file(right_root,right,'val')),'--field-b',right_field,'--alphas',weights,'--betas',weights,'--out-json',str(path)]);candidate={**json.loads(path.read_text(encoding='utf8'))['best'],'left':left,'right':right}
  if best is None or candidate['map50']>best['map50']:best=candidate
 (OUT/'validation_selection.json').write_text(json.dumps(best,indent=2),encoding='utf8');left_root,left_field=SOURCES[best['left']];right_root,right_field=SOURCES[best['right']];execute('fixed_test',5,[sys.executable,str(tool),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--score-a',str(score_file(left_root,best['left'],'test')),'--field-a',left_field,'--score-b',str(score_file(right_root,best['right'],'test')),'--field-b',right_field,'--alphas',str(best['alpha']),'--betas',str(best['beta']),'--out-json',str(OUT/'test_fixed.json')]);test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best'];summary={'protocol':'pure Action Chunk expert ensemble; pair and weights selected on OOF validation; fixed test','validation_selection':best,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');report('done',7,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
