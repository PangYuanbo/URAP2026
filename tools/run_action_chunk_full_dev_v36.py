from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');PYTHON=Path(sys.executable);RUN=REPO/'artifacts'/'detached_action_chunk_full_dev_v36';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_full_dev_v36');TOTAL=6
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={'stage':stage,'done':done,'total':TOTAL,'updated':datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding='utf-8');print(json.dumps(p),flush=True)
def execute(stage,done,cmd):
 p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report(stage,done,child_pid=p.pid,command=cmd);code=p.wait()
 if code:raise subprocess.CalledProcessError(code,cmd)
def main():
 OUT.mkdir(parents=True,exist_ok=True);jobs=[('migrate_forward_train',r'D:\URAP_vatd_rank_results\nps_online_action_bank_v14\train_scores.jsonl',OUT/'train_forward.jsonl'),('migrate_forward_val',r'D:\URAP_vatd_rank_results\nps_online_action_bank_v14\val_scores.jsonl',OUT/'val_forward.jsonl'),('migrate_forward_test',r'D:\URAP_vatd_rank_results\nps_online_action_bank_v14\test_scores.jsonl',OUT/'test_forward.jsonl'),('migrate_reverse_val',r'D:\URAP_vatd_rank_results\nps_reverse_action_bank_val_v29\val_reverse_scores.jsonl',OUT/'val_backward.jsonl'),('migrate_reverse_test',r'D:\URAP_vatd_rank_results\nps_reverse_action_bank_test_v34\test_reverse_scores.jsonl',OUT/'test_backward.jsonl')]
 for i,(stage,source,target) in enumerate(jobs):execute(stage,i,[str(PYTHON),str(REPO/'tools'/'migrate_online_bank_to_action_chunk.py'),'--input',str(source),'--output',str(target)])
 execute('score_reverse_train',5,[str(PYTHON),str(REPO/'tools'/'score_predictionsgt_action_chunk_bank.py'),'--predictionsgt-pkl',r'D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl','--frame-root',r'U:\URAP_datasets\TransVisDrone\NPS\AllFrames\train','--homography-cache',r'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\train.pkl','--out-jsonl',str(OUT/'train_backward.jsonl'),'--out-summary',str(OUT/'train_backward_summary.json'),'--sequence-fps-json',str(REPO/'data_templates'/'nps_sequence_fps.json'),'--short-seconds','1.0','--long-seconds','3.0','--beam-size','6','--short-token-count','8','--long-token-count','16','--start-gate','.12','--update-gate','.08','--internal-alpha','2.5','--reverse']);report('done',TOTAL,output=str(OUT));return 0
if __name__=='__main__':raise SystemExit(main())
