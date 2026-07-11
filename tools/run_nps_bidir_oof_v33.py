from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');PYTHON=Path(sys.executable);RUN=REPO/'artifacts'/'detached_nps_bidir_oof_v33';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\nps_bidir_oof_v33')
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={'stage':stage,'done':done,'total':2,'updated':datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding='utf-8');print(json.dumps(p),flush=True)
def execute(stage,done,cmd):
 p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report(stage,done,child_pid=p.pid,command=cmd);code=p.wait()
 if code:raise subprocess.CalledProcessError(code,cmd)
def main():
 OUT.mkdir(parents=True,exist_ok=True);execute('train_oof',0,[str(PYTHON),str(REPO/'tools'/'train_bidir_action_bank_oof.py'),'--pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--forward',r'D:\URAP_vatd_rank_results\nps_online_action_bank_v14\val_scores.jsonl','--backward',r'D:\URAP_vatd_rank_results\nps_reverse_action_bank_val_v29\val_reverse_scores.jsonl','--out-scores',str(OUT/'val_oof_scores.jsonl'),'--out-summary',str(OUT/'train_summary.json')]);execute('sweep_oof',1,[str(PYTHON),str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--tracklet-jsonl',str(OUT/'val_oof_scores.jsonl'),'--per-row-score','--score-field','bidir_oof_score','--modes','replace','linear-mix','logit-mix','geom-mix','fp-suppress','tp-boost','--alphas','0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40 0.55 0.70 0.85 1.0','--out-json',str(OUT/'val_sweep.json')]);best=json.loads((OUT/'val_sweep.json').read_text(encoding='utf-8'))['best'];report('done',2,best=best);return 0
if __name__=='__main__':raise SystemExit(main())
