from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP");PYTHON=Path(sys.executable);RUN=REPO/"artifacts"/"detached_nps_persistence_val_sweep_v20";PROGRESS=RUN/"progress.json";OUT=Path(r"D:\URAP_vatd_rank_results\nps_persistence_action_bank_v20");FIELDS=("persistence_score","persistence_motion","persistence")
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":3,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding="utf-8");print(json.dumps(p),flush=True)
def main():
 s=OUT/"sweeps";s.mkdir(parents=True,exist_ok=True);best=None
 for i,f in enumerate(FIELDS):
  out=s/f"{f}.json";cmd=[str(PYTHON),str(REPO/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl","--tracklet-jsonl",str(OUT/"val_scores.jsonl"),"--per-row-score","--score-field",f,"--modes","linear-mix","logit-mix","geom-mix","fp-suppress","tp-boost","--alphas","0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40","--out-json",str(out)];p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,"PYTHONPATH":str(REPO)+os.pathsep+str(REPO/"tools")});report(f"sweep_{f}",i,child_pid=p.pid,command=cmd);code=p.wait();
  if code:raise subprocess.CalledProcessError(code,cmd)
  row=json.loads(out.read_text())["best"]|{"field":f};best=row if best is None or row["map50"]>best["map50"] else best;report(f"sweep_{f}_done",i+1,best=best)
 (OUT/"validation_selection.json").write_text(json.dumps(best,indent=2),encoding="utf-8");report("done",3,best=best);return 0
if __name__=="__main__":raise SystemExit(main())
