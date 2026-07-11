from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP")
PYTHON=Path(sys.executable)
RUN=REPO/"artifacts"/"detached_nps_teacher_action_bank_v16"
PROGRESS=RUN/"progress.json"
OUT=Path(r"D:\URAP_vatd_rank_results\nps_teacher_action_bank_v16")
def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True); p={"stage":stage,"done":done,"total":1,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra}; PROGRESS.write_text(json.dumps(p,indent=2),encoding="utf-8"); print(json.dumps(p),flush=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cmd=[str(PYTHON),str(REPO/"tools"/"score_predictionsgt_online_action_bank_teacher.py"),"--predictionsgt-pkl",r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl","--frame-root",r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\train","--homography-cache",r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\train.pkl","--out-jsonl",str(OUT/"train_scores.jsonl"),"--out-summary",str(OUT/"train_summary.json"),"--sequence-fps-json",str(REPO/"data_templates"/"nps_sequence_fps.json"),"--short-seconds","1.0","--long-seconds","3.0","--beam-size","6","--short-token-count","8","--long-token-count","16","--teacher-force-updates"]
 p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,"PYTHONPATH":str(REPO),"PYTHONUNBUFFERED":"1"}); report("teacher_train_action_bank",0,child_pid=p.pid,command=cmd); code=p.wait();
 if code: raise subprocess.CalledProcessError(code,cmd)
 report("done",1,output=str(OUT)); return 0
if __name__=="__main__": raise SystemExit(main())
