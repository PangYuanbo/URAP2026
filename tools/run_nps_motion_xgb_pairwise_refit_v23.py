from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP"); PYTHON=Path(sys.executable)
RUN=REPO/"artifacts"/"detached_nps_motion_xgb_pairwise_refit_v23"; PROGRESS=RUN/"progress.json"
OUT=Path(r"D:\URAP_vatd_rank_results\nps_motion_xgb_pairwise_refit_v23")
def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True); payload={"stage":stage,"done":done,"total":2,"updated":datetime.now(timezone.utc).astimezone().isoformat(),**extra}; PROGRESS.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload),flush=True)
def execute(stage,done,command):
 p=subprocess.Popen(command,cwd=REPO,env={**os.environ,"PYTHONPATH":str(REPO)+os.pathsep+str(REPO/"tools"),"PYTHONUNBUFFERED":"1"}); report(stage,done,child_pid=p.pid,command=command); code=p.wait()
 if code: raise subprocess.CalledProcessError(code,command)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 execute("refit_dev",0,[str(PYTHON),str(REPO/"tools"/"refit_action_bank_motion_xgb_ranker.py"),"--train-pkl",r"D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl","--train-aux",r"D:\URAP_vatd_rank_results\nps_online_action_bank_v14\train_scores.jsonl","--val-pkl",r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl","--val-aux",r"D:\URAP_vatd_rank_results\nps_online_action_bank_v14\val_scores.jsonl","--test-pkl",r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl","--test-aux",r"D:\URAP_vatd_rank_results\nps_online_action_bank_v14\test_scores.jsonl","--out-test",str(OUT/"test_scores.jsonl"),"--out-model",str(OUT/"model.ubj"),"--out-summary",str(OUT/"train_summary.json"),"--rounds","264","--score-field","xgb_pairwise_refit_score"])
 execute("fixed_test",1,[str(PYTHON),str(REPO/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl","--tracklet-jsonl",str(OUT/"test_scores.jsonl"),"--per-row-score","--score-field","xgb_pairwise_refit_score","--modes","linear-mix","--alphas","0.2","--out-json",str(OUT/"test_fixed.json")])
 test=json.loads((OUT/"test_fixed.json").read_text(encoding="utf-8"))["best"]; summary={"protocol":"select rounds/fusion on Clips37-40; refit Clips1-40; fixed test Clips41-50","test_fixed":test,"target_map50":.97,"target_met":test["map50"]>=.97}; (OUT/"official_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); report("done",2,summary=summary); return 0
if __name__=="__main__": raise SystemExit(main())
