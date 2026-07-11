from __future__ import annotations
import json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');RUN=REPO/'artifacts'/'detached_action_chunk_bidir_stable_v81';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_bidir_stable_v81');REVERSE=Path(r'D:\URAP_vatd_rank_results\action_chunk_reverse_chain_v80');FORWARD=Path(r'D:\URAP_vatd_rank_results\action_chunk_chain_features_v75');V46=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46')
def report(stage,done,total=2,**extra):RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(payload),encoding='utf8');print(json.dumps(payload),flush=True)
def execute(stage,done,command):
 process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report(stage,done,child_pid=process.pid,command=command);code=process.wait()
 if code:raise RuntimeError(f'{stage} failed with {code}')
def main():
 reverse_progress=REPO/'artifacts'/'detached_action_chunk_reverse_chain_v80'/'progress.json';report('waiting_for_v80',0,dependency=str(reverse_progress))
 while True:
  if reverse_progress.exists():
   state=json.loads(reverse_progress.read_text(encoding='utf8'))
   if state.get('stage')=='done' and int(state.get('done',0))==int(state.get('total',-1)):break
  time.sleep(20)
 selection=json.loads((OUT/'validation_selection.json').read_text(encoding='utf-8-sig'));execute('derive_test_bidir_chain',0,[sys.executable,str(REPO/'tools'/'derive_action_chunk_bidir_stable_chain.py'),'--forward-jsonl',str(FORWARD/'test_chain.jsonl'),'--reverse-jsonl',str(REVERSE/'test_reverse_chain.jsonl'),'--out-jsonl',str(OUT/'test_scores.jsonl')]);execute('fixed_test',1,[sys.executable,str(REPO/'tools'/'sweep_action_chunk_two_row_ensemble.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--score-a',str(V46/'test_scores.jsonl'),'--field-a','action_chunk_neighbor_score','--score-b',str(OUT/'test_scores.jsonl'),'--field-b',selection['field'],'--alphas',str(selection['alpha']),'--betas',str(selection['beta']),'--out-json',str(OUT/'test_fixed.json')]);test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best'];summary={'protocol':'pure Action Chunk forward/backward stable-chain intersection; validation-selected field and weights; single fixed test','validation_selection':selection,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');report('done',2,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
