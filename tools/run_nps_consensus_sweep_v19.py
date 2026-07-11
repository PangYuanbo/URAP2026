from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP");PYTHON=Path(sys.executable);RUN=REPO/"artifacts"/"detached_nps_consensus_sweep_v19";PROGRESS=RUN/"progress.json";OUT=Path(r"D:\URAP_vatd_rank_results\nps_candidate_consensus_v19");FIELDS=("consensus_score","consensus_noisy_or","consensus_density","consensus_local_max","consensus_support","consensus_strong_support")
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":len(FIELDS)+1,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(p,indent=2),encoding="utf-8");print(json.dumps(p),flush=True)
def execute(stage, done, command):
 p=subprocess.Popen(command,cwd=REPO,env={**os.environ,"PYTHONPATH":str(REPO)+os.pathsep+str(REPO/"tools"),"PYTHONUNBUFFERED":"1"})
 report(stage,done,child_pid=p.pid,command=command)
 code=p.wait()
 if code:
  raise subprocess.CalledProcessError(code,command)
def main():
 sweeps=OUT/"sweeps";sweeps.mkdir(parents=True,exist_ok=True)
 for i,field in enumerate(FIELDS):
  result=sweeps/f"{field}.json"
  if not result.is_file():execute(f"sweep_{field}",i,[str(PYTHON),str(REPO/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl","--tracklet-jsonl",str(OUT/"val_scores.jsonl"),"--per-row-score","--score-field",field,"--modes","replace","linear-mix","logit-mix","geom-mix","fp-suppress","tp-boost","--alphas","0.001 0.002 0.005 0.01 0.02 0.04 0.06 0.08 0.10 0.14 0.20 0.30 0.40 0.55 0.70 0.85 1.0","--out-json",str(result)])
  report(f"sweep_{field}_done",i+1,output=str(result))
 choices=[json.loads((sweeps/f"{f}.json").read_text())["best"]|{"field":f} for f in FIELDS];best=max(choices,key=lambda r:r["map50"]);fixed=OUT/"test_fixed_consensus.json"
 execute("fixed_test",len(FIELDS),[str(PYTHON),str(REPO/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl","--tracklet-jsonl",str(OUT/"test_scores.jsonl"),"--per-row-score","--score-field",best["field"],"--modes",best["mode"],"--alphas",str(best["alpha"]),"--out-json",str(fixed)])
 test=json.loads(fixed.read_text())["best"];summary={"protocol":"validation select, fixed test","validation_best":best,"test_fixed":test,"target_map50":.97,"target_met":test["map50"]>=.97};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");report("done",len(FIELDS)+1,summary=summary);print(json.dumps(summary,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
