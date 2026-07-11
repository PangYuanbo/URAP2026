from __future__ import annotations
import ctypes,json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');RUN=R/'artifacts'/'detached_tvd_finetune_posteval_v103';P=RUN/'progress.json';TRAIN_RUN=R/'artifacts'/'detached_tvd_detector_finetune_v102';TRAIN_OUT=Path(r'D:\URAP_vatd_rank_results\tvd_detector_finetune_v102\finetune_lowaug_e8');REPO=Path(r'U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone');OUT=Path(r'D:\URAP_vatd_rank_results\tvd_detector_finetune_v102_eval')
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':2,'updated':datetime.now().astimezone().isoformat(),**extra},indent=2),encoding='utf8');print(json.dumps({'stage':stage,**extra}),flush=True)
def alive(pid):
 handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
 if not handle:return False
 ctypes.windll.kernel32.CloseHandle(handle);return True
def main():
 train_pid=int((TRAIN_RUN/'pid.txt').read_text().strip());report('waiting_for_training',0,training_pid=train_pid)
 while alive(train_pid):
  results=TRAIN_OUT/'results.csv';rows=max(0,len(results.read_text(encoding='utf8').splitlines())-1) if results.exists() else 0;report('waiting_for_training',0,training_pid=train_pid,epochs_done=rows,epochs_total=8,last_results_write=results.stat().st_mtime if results.exists() else None);time.sleep(60)
 results=TRAIN_OUT/'results.csv';rows=max(0,len(results.read_text(encoding='utf8').splitlines())-1) if results.exists() else 0
 if rows<8:report('training_not_complete',0,training_pid=train_pid,epochs_done=rows,epochs_total=8);return 2
 weights=TRAIN_OUT/'weights'/'best.pt'
 if not weights.exists():report('missing_best_weight',0,epochs_done=rows);return 3
 OUT.mkdir(parents=True,exist_ok=True);command=[str(REPO/'.venv'/'Scripts'/'python.exe'),str(REPO/'val.py'),'--task','test','--data',str(REPO/'data'/'NPS_URAP_D.yaml'),'--weights',str(weights),'--img','1280','--batch-size','2','--half','--num-frames','5','--conf-thres','0.001','--iou-thres','0.6','--project',str(OUT),'--name','test_best','--exist-ok','--save-json-gt'];env={**os.environ,'WANDB_MODE':'disabled'};process=subprocess.Popen(command,cwd=REPO,env=env);report('evaluate_test',1,child_pid=process.pid,weights=str(weights),command=command);code=process.wait()
 if code:report('test_failed',1,exit_code=code);return code
 result_path=OUT/'test_best'/'results.txt';prediction_path=OUT/'test_best'/'predictionsgt'/'predictionsgt_split_0.pkl';report('done',2,result_path=str(result_path),prediction_path=str(prediction_path),result=result_path.read_text(encoding='utf8') if result_path.exists() else None);return 0
if __name__=='__main__':raise SystemExit(main())
