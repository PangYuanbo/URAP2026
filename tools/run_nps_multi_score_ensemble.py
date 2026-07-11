from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP"); TVD=Path(r"D:\urap_modal_stage\TransVisDrone"); INPUT=Path(r"D:\URAP_vatd_rank_inputs"); RESULTS=Path(r"D:\URAP_vatd_rank_results"); OUTPUT=RESULTS/"nps_multi_score_ensemble_v1"; RUNNER=REPO/"artifacts/detached_nps_multi_score_ensemble"; PROGRESS=RUNNER/"progress.json"; PYTHON=Path(sys.executable)
def progress(stage,done,total=2,**extra): RUNNER.mkdir(parents=True,exist_ok=True); PROGRESS.write_text(json.dumps({"stage":stage,"done":done,"total":total,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra},indent=2),encoding="utf-8")
def run(command,stage,done):
 print(json.dumps({"kind":"pipeline_command","stage":stage,"command":command}),flush=True); p=subprocess.Popen(command,cwd=REPO,env={**os.environ,"PYTHONUNBUFFERED":"1","PYTHONPATH":str(REPO/"tools")}); progress(stage,done,child_pid=p.pid,command=command); code=p.wait();
 if code: raise subprocess.CalledProcessError(code,command)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True); score_map=OUTPUT/"multi_scores.pkl"
 run([str(PYTHON),str(REPO/"tools/build_vatd_multi_score_map.py"),"--val-jsonl",str(RESULTS/"nps_official_val_to_test_cuda_rank_v2/nps_test_tracklets_val_rank_scored.jsonl"),"--aot-jsonl",str(RESULTS/"aot_to_nps_cuda_rank_v1/nps_tracklets_aot_rank_scored.jsonl"),"--xgb-jsonl",str(RESULTS/"nps_official_val_xgb_rank_v1/nps_test_tracklets_xgb_scored.jsonl"),"--out-pkl",str(score_map)],"build_scores",0)
 run([str(PYTHON),str(REPO/"tools/sweep_tvd_predictionsgt_multi_score_fusion_fast.py"),"--tvd-root",str(TVD),"--predictionsgt-pkl",str(INPUT/"nps_predictionsgt_split_0.pkl"),"--score-map-pkl",str(score_map),"--modes","meta-logit-row-geom","meta-logit-row-suppress","meta-logit-row-boost","--alphas","0.0","--betas","0.02","0.04","0.06","0.08","0.10","0.12","0.14","0.16","0.18","0.20","0.24","0.28","0.32","0.40","--out-json",str(OUTPUT/"fusion_sweep.json")],"evaluate",1)
 summary=json.loads((OUTPUT/"fusion_sweep.json").read_text(encoding="utf-8")); progress("done",2,best=summary.get("best"),output=str(OUTPUT)); print(json.dumps({"kind":"pipeline_done","best":summary.get("best")}),flush=True); return 0
if __name__=="__main__": raise SystemExit(main())
