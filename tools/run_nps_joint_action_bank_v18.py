from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP");PYTHON=Path(sys.executable);RUN=REPO/"artifacts"/"detached_nps_joint_action_bank_v18";PROGRESS=RUN/"progress.json";OUT=Path(r"D:\URAP_vatd_rank_results\nps_joint_action_bank_v18")
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":2,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding="utf-8");print(json.dumps(p),flush=True)
def run(stage,done,pkl,frames,hom,out,summary):
 cmd=[str(PYTHON),str(REPO/"tools"/"score_predictionsgt_joint_action_bank.py"),"--predictionsgt-pkl",str(pkl),"--frame-root",str(frames),"--homography-cache",str(hom),"--out-jsonl",str(out),"--out-summary",str(summary),"--sequence-fps-json",str(REPO/"data_templates"/"nps_sequence_fps.json")];p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,"PYTHONPATH":str(REPO),"PYTHONUNBUFFERED":"1"});report(stage,done,child_pid=p.pid,command=cmd);code=p.wait();
 if code:raise subprocess.CalledProcessError(code,cmd)
 report(stage+"_done",done+1,output=str(out))
def main():
 OUT.mkdir(parents=True,exist_ok=True);run("joint_val",0,Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl"),Path(r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\val"),Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\val.pkl"),OUT/"val_scores.jsonl",OUT/"val_summary.json");run("joint_test",1,Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl"),Path(r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\test"),Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\test.pkl"),OUT/"test_scores.jsonl",OUT/"test_summary.json");report("done",2,output=str(OUT));return 0
if __name__=="__main__":raise SystemExit(main())
