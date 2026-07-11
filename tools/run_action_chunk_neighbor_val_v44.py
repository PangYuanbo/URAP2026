from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');RUN=REPO/'artifacts'/'detached_action_chunk_neighbor_val_v44';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_v44')
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':1,'updated':datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(payload,indent=2),encoding='utf8');print(json.dumps(payload),flush=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,str(REPO/'tools'/'score_action_chunk_neighbor_bank.py'),'--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--homography-cache',r'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\val.pkl','--out-jsonl',str(OUT/'val_neighbor_scores.jsonl'),'--out-summary',str(OUT/'val_neighbor_summary.json'),'--sequence-fps-json',str(REPO/'data_templates'/'nps_sequence_fps.json'),'--bidirectional'];p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report('score_validation_neighbor_bank',0,child_pid=p.pid,command=cmd);code=p.wait()
 if code:raise subprocess.CalledProcessError(code,cmd)
 report('done',1,output=str(OUT));return 0
if __name__=='__main__':raise SystemExit(main())
