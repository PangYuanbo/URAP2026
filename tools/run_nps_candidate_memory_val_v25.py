from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP");PYTHON=Path(sys.executable);RUN=REPO/"artifacts"/"detached_nps_candidate_memory_val_v25";PROGRESS=RUN/"progress.json";OUT=Path(r"D:\URAP_vatd_rank_results\nps_candidate_memory_val_v25")
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":2,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding="utf-8");print(json.dumps(p),flush=True)
def execute(stage,done,cmd):
 p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,"PYTHONPATH":str(REPO)+os.pathsep+str(REPO/"tools"),"PYTHONUNBUFFERED":"1"});report(stage,done,child_pid=p.pid,command=cmd);code=p.wait()
 if code:raise subprocess.CalledProcessError(code,cmd)
def main():
 OUT.mkdir(parents=True,exist_ok=True);execute("score_val_memory",0,[str(PYTHON),str(REPO/"tools"/"score_predictionsgt_candidate_memory_bank.py"),"--predictionsgt-pkl",r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl","--frame-root",r"U:\URAP_datasets\TransVisDrone\NPS\AllFrames\val","--homography-cache",r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\val.pkl","--sequence-fps-json",str(REPO/"data_templates"/"nps_sequence_fps.json"),"--out-jsonl",str(OUT/"val_scores.jsonl"),"--out-summary",str(OUT/"val_score_summary.json"),"--short-seconds","1.0","--long-seconds","3.0","--short-tokens","8","--long-tokens","16","--memory-top-k","12"]);execute("sweep_val",1,[str(PYTHON),str(REPO/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl","--tracklet-jsonl",str(OUT/"val_scores.jsonl"),"--per-row-score","--score-field","memory_support_score","--modes","replace","linear-mix","logit-mix","geom-mix","fp-suppress","tp-boost","--alphas","0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40 0.55 0.70 0.85 1.0","--out-json",str(OUT/"val_sweep.json")]);best=json.loads((OUT/"val_sweep.json").read_text(encoding="utf-8"))["best"];report("done",2,best=best);return 0
if __name__=="__main__":raise SystemExit(main())
