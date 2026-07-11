from __future__ import annotations
import json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
REPO=Path(r'C:\Users\aaron\Desktop\URAP');RUN=REPO/'artifacts'/'detached_ard100_yolomg_val_prepare_v2';P=RUN/'progress.json';OUT=Path(r'D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2');DET=OUT/'yolomg_ard100_val_candidates';PKL=OUT/'ard100_yolomg_val_predictionsgt.pkl'
def report(stage,done,**x):
 RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':3,'updated':datetime.now().astimezone().isoformat(),**x},indent=2));print(stage,flush=True)
def run(stage,done,args):
 q=subprocess.Popen(args,cwd=REPO,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONPATH':str(REPO)});report(stage,done,child_pid=q.pid);c=q.wait()
 if c:raise RuntimeError(f'{stage} failed {c}')
def main():
 while not (DET/'results.txt').is_file():report('waiting_for_detector',0);time.sleep(20)
 run('convert',1,[sys.executable,str(REPO/'tools'/'convert_ard100_yolomg_predictionsgt.py'),'--image-list',r'D:\URAP_datasets\ARD100_YOLOMG\val.txt','--prediction-label-root',str(DET/'labels'),'--out-pkl',str(PKL),'--out-summary',str(OUT/'val_conversion_summary.json')])
 run('evaluate',2,[sys.executable,str(REPO/'tools'/'eval_tvd_predictionsgt_pkl.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(PKL),'--out-json',str(OUT/'val_detector_baseline.json')]);report('done',3,baseline=json.loads((OUT/'val_detector_baseline.json').read_text()));return 0
if __name__=='__main__':raise SystemExit(main())
