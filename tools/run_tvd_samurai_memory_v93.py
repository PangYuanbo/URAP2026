from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');RUN=R/'artifacts'/'detached_tvd_samurai_memory_v93';P=RUN/'progress.json';O=Path(r'D:\URAP_vatd_rank_results\tvd_samurai_memory_v93');VAL=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');H=Path(r'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies')
FIELDS=['samurai_memory_sym_0p94','samurai_memory_sym_0p98','samurai_memory_sym_0p995','samurai_memory_min_0p94','samurai_memory_min_0p98','samurai_memory_min_0p995','samurai_memory_span_1s','samurai_memory_span_3s','samurai_memory_motion','samurai_memory_motion_sym']
def rep(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':4,'updated':datetime.now().astimezone().isoformat(),**extra},indent=2),encoding='utf8');print(stage,flush=True)
def go(stage,done,command):
 process=subprocess.Popen(command,cwd=R,env={**os.environ,'PYTHONPATH':str(R),'PYTHONUNBUFFERED':'1'});rep(stage,done,child_pid=process.pid,command=command);code=process.wait()
 if code:raise RuntimeError(f'{stage} failed {code}')
def main():
 O.mkdir(parents=True,exist_ok=True);go('score_validation_memory',0,[sys.executable,str(R/'tools'/'score_action_chunk_samurai_memory.py'),'--predictionsgt-pkl',str(VAL),'--homography-cache',str(H/'val.pkl'),'--out-jsonl',str(O/'val_scores.jsonl'),'--out-summary',str(O/'val_score_summary.json')]);go('score_test_memory',1,[sys.executable,str(R/'tools'/'score_action_chunk_samurai_memory.py'),'--predictionsgt-pkl',str(TEST),'--homography-cache',str(H/'test.pkl'),'--out-jsonl',str(O/'test_scores.jsonl'),'--out-summary',str(O/'test_score_summary.json')]);best=None
 for field in FIELDS:
  sweep=O/f'val_{field}.json';go(f'select_{field}',2,[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(VAL),'--tracklet-jsonl',str(O/'val_scores.jsonl'),'--per-row-score','--score-field',field,'--modes','geom-mix','logit-mix','fp-suppress','tp-boost','replace','--alphas','.01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,.85','--out-json',str(sweep)]);candidate=json.loads(sweep.read_text(encoding='utf8'))['best'];candidate['score_field']=field
  if best is None or candidate['map50']>best['map50']:best=candidate
 fixed=O/'test_fixed.json';go('fixed_test',3,[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(TEST),'--tracklet-jsonl',str(O/'test_scores.jsonl'),'--per-row-score','--score-field',best['score_field'],'--modes',best['mode'],'--alphas',str(best['alpha']),'--out-json',str(fixed)]);test=json.loads(fixed.read_text(encoding='utf8'))['best'];summary={'protocol':'SAMURAI-style bidirectional high-confidence memory; camera-motion compensated; validation-selected and fixed test','val_best':best,'test':test,'gain_over_vatd_points':100*(test['map50']-.93844),'target_3_to_5_met':.03<=test['map50']-.93844<=.05};(O/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');rep('done',4,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
