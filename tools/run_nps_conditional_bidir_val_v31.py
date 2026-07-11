from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');PYTHON=Path(sys.executable);RUN=REPO/'artifacts'/'detached_nps_conditional_bidir_val_v31';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\nps_conditional_bidir_val_v31')
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={'stage':stage,'done':done,'total':1,'updated':datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding='utf-8');print(json.dumps(p),flush=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True);cmd=[str(PYTHON),str(REPO/'tools'/'sweep_conditional_bidir_rerank.py'),'--pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--scores',r'D:\URAP_vatd_rank_results\nps_bidir_correction_val_v30\val_scores.jsonl','--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--out',str(OUT/'val_sweep.json')];p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report('sweep_conditional_val',0,child_pid=p.pid,command=cmd);code=p.wait()
 if code:raise subprocess.CalledProcessError(code,cmd)
 best=json.loads((OUT/'val_sweep.json').read_text(encoding='utf-8'))['best'];report('done',1,best=best);return 0
if __name__=='__main__':raise SystemExit(main())
