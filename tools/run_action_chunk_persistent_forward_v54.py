from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');RUN=REPO/'artifacts'/'detached_action_chunk_persistent_forward_v54';PROGRESS=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_persistent_forward_v54');HOM=Path(r'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies');FPS=REPO/'data_templates'/'nps_sequence_fps.json';TOTAL=3
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);PROGRESS.write_text(json.dumps({'stage':stage,'done':done,'total':TOTAL,'updated':datetime.now().astimezone().isoformat(),**extra}),encoding='utf8')
def execute(stage,done,pkl,frames,cache,out,summary):
 cmd=[sys.executable,str(REPO/'tools'/'score_predictionsgt_action_chunk_bank.py'),'--predictionsgt-pkl',str(pkl),'--frame-root',str(frames),'--homography-cache',str(cache),'--out-jsonl',str(out),'--out-summary',str(summary),'--sequence-fps-json',str(FPS),'--short-seconds','1','--long-seconds','3','--beam-size','6','--short-token-count','8','--long-token-count','16','--start-gate','.12','--update-gate','.08','--internal-alpha','2.5'];process=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'});report(stage,done,child_pid=process.pid,command=cmd);code=process.wait()
 if code:raise RuntimeError(f'{stage} failed with {code}')
 report(stage+'_done',done+1,output=str(out))
def main():
 OUT.mkdir(parents=True,exist_ok=True);execute('score_train_persistent',0,Path(r'D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl'),Path(r'U:\URAP_datasets\TransVisDrone\NPS\AllFrames\train'),HOM/'train.pkl',OUT/'train_forward.jsonl',OUT/'train_summary.json');execute('score_val_persistent',1,Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl'),Path(r'U:\URAP_datasets\TransVisDrone\NPS\AllFrames\val'),HOM/'val.pkl',OUT/'val_forward.jsonl',OUT/'val_summary.json');execute('score_test_persistent',2,Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl'),Path(r'U:\URAP_datasets\TransVisDrone\NPS\AllFrames\test'),HOM/'test.pkl',OUT/'test_forward.jsonl',OUT/'test_summary.json');report('done',3,output=str(OUT));return 0
if __name__=='__main__':raise SystemExit(main())
