from __future__ import annotations
import json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
REPO=Path(r"C:\Users\aaron\Desktop\URAP");RUN=REPO/"artifacts"/"detached_ard100_yolomg_prepare_v2";PROGRESS=RUN/"progress.json";OUT=Path(r"D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2");DETECT=OUT/"yolomg_ard100_test_candidates";RESULTS=DETECT/"results.txt";LABELS=DETECT/"labels";PKL=OUT/"ard100_yolomg_predictionsgt.pkl"
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);payload={"stage":stage,"done":done,"total":3,"updated":datetime.now().astimezone().isoformat(),**extra};PROGRESS.write_text(json.dumps(payload,indent=2),encoding="utf8");print(json.dumps(payload),flush=True)
def execute(stage,done,cmd):
 p=subprocess.Popen(cmd,cwd=REPO,env={**os.environ,"PYTHONUNBUFFERED":"1","PYTHONPATH":str(REPO)})
 report(stage,done,child_pid=p.pid,command=cmd)
 code=p.wait()
 if code:raise RuntimeError(f"{stage} failed {code}")
def main():
 while not RESULTS.is_file():report("waiting_for_yolomg_detector",0);time.sleep(30)
 execute("convert_predictions",1,[sys.executable,str(REPO/"tools"/"convert_ard100_yolomg_predictionsgt.py"),"--image-list",r"D:\URAP_datasets\ARD100_YOLOMG\test.txt","--prediction-label-root",str(LABELS),"--out-pkl",str(PKL),"--out-summary",str(OUT/"conversion_summary.json")])
 execute("evaluate_baseline",2,[sys.executable,str(REPO/"tools"/"eval_tvd_predictionsgt_pkl.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",str(PKL),"--out-json",str(OUT/"detector_baseline.json")]);report("done",3,baseline=json.loads((OUT/"detector_baseline.json").read_text()));return 0
if __name__=="__main__":raise SystemExit(main())
