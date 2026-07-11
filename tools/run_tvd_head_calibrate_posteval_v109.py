from __future__ import annotations
import ctypes,json,os,subprocess,time
from datetime import datetime
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');RUN=R/'artifacts'/'detached_tvd_head_calibrate_posteval_v109';P=RUN/'progress.json';TRAIN_RUN=R/'artifacts'/'detached_tvd_head_calibrate_v108';TRAIN_OUT=Path(r'D:\URAP_vatd_rank_results\tvd_head_calibrate_v108\val_head_e1');REPO=Path(r'U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone');OUT=Path(r'D:\URAP_vatd_rank_results\tvd_head_calibrate_v108_eval')
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':2,'updated':datetime.now().astimezone().isoformat(),**extra},indent=2),encoding='utf8');print(json.dumps({'stage':stage,**extra}),flush=True)
def alive(pid):
 handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
 if not handle:return False
 ctypes.windll.kernel32.CloseHandle(handle);return True
def epochs_done():
 results=TRAIN_OUT/'results.csv';return max(0,len(results.read_text(encoding='utf8').splitlines())-1) if results.exists() else 0
def main():
 train_pid=int((TRAIN_RUN/'pid.txt').read_text().strip());report('waiting_for_training',0,training_pid=train_pid,epochs_done=epochs_done(),epochs_total=1)
 while alive(train_pid):report('waiting_for_training',0,training_pid=train_pid,epochs_done=epochs_done(),epochs_total=1);time.sleep(60)
 done=epochs_done();weights=TRAIN_OUT/'weights'/'last.pt'
 if done<1 or not weights.exists():report('training_not_complete',0,training_pid=train_pid,epochs_done=done,epochs_total=1,weights_exists=weights.exists());return 2
 OUT.mkdir(parents=True,exist_ok=True);command=[str(REPO/'.venv'/'Scripts'/'python.exe'),str(REPO/'val.py'),'--task','test','--data',str(REPO/'data'/'NPS_URAP_D.yaml'),'--weights',str(weights),'--img','1280','--batch-size','2','--half','--num-frames','5','--conf-thres','0.001','--iou-thres','0.6','--project',str(OUT),'--name','test_last','--exist-ok','--save-json-gt'];process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'WANDB_MODE':'disabled'});report('evaluate_test',1,child_pid=process.pid,weights=str(weights));code=process.wait()
 if code:report('test_failed',1,exit_code=code);return code
 result=OUT/'test_last'/'results.txt';prediction=OUT/'test_last'/'predictionsgt'/'predictionsgt_split_0.pkl';report('done',2,result_path=str(result),prediction_path=str(prediction),result=result.read_text(encoding='utf8') if result.exists() else None);return 0
if __name__=='__main__':raise SystemExit(main())
