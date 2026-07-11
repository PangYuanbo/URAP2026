from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');T=Path(r'D:\urap_modal_stage\TransVisDrone');V=Path(r'D:\URAP_nps_val_tvd');I=Path(r'D:\URAP_vatd_rank_inputs');O=Path(r'D:\URAP_vatd_rank_results\nps_val_lambdarank_v1');A=R/'artifacts/detached_nps_val_lambdarank';P=A/'progress.json';PY=Path(sys.executable)
def prog(stage,done,total=2,**x):A.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':total,'updated':datetime.now(timezone.utc).astimezone().isoformat(),**x},indent=2),encoding='utf-8')
def run(c,stage,done):
 print(json.dumps({'kind':'pipeline_command','stage':stage,'command':c}),flush=True);p=subprocess.Popen(c,cwd=R,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONPATH':str(R/'tools')});prog(stage,done,child_pid=p.pid,command=c);code=p.wait()
 if code:raise subprocess.CalledProcessError(code,c)
def main():
 O.mkdir(parents=True,exist_ok=True);scored=O/'nps_test_tracklets_lambdarank_scored.jsonl'
 run([str(PY),str(R/'tools/train_detection_row_lambdarank.py'),'--train-tracklets',str(V/'route_b_official/tracklets/proposal_tracklets.jsonl'),'--train-gt-csv',str(V/'route_b_official/gt.csv'),'--test-tracklets',str(I/'nps_tracklets_with_vatd.jsonl'),'--out-test-tracklets',str(scored),'--out-model',str(O/'lambdarank_model.json'),'--out-summary',str(O/'train_summary.json'),'--score-field','lambda_rank_score','--min-score','0.001','--rounds','1200','--max-depth','7','--eta','0.025'],'train',0)
 run([str(PY),str(R/'tools/sweep_tvd_predictionsgt_two_score_fusion_fast.py'),'--tvd-root',str(T),'--predictionsgt-pkl',str(I/'nps_predictionsgt_split_0.pkl'),'--meta-tracklet-jsonl',str(scored),'--meta-score-field','vatd_score','--row-tracklet-jsonl',str(scored),'--row-score-field','lambda_rank_score','--modes','logit-3mix','meta-logit-row-geom','meta-logit-row-suppress','meta-logit-row-boost','--alphas','0.00','0.01','0.02','0.04','0.06','0.08','0.10','0.14','0.20','--betas','0.005','0.01','0.02','0.04','0.06','0.08','0.10','0.12','0.16','0.20','0.24','0.32','0.40','--out-json',str(O/'fusion_sweep_fast.json'),'--write-best-pkl',str(O/'best_predictionsgt.pkl')],'evaluate',1)
 s=json.loads((O/'fusion_sweep_fast.json').read_text());prog('done',2,best=s.get('best'),output=str(O));print(json.dumps({'kind':'pipeline_done','best':s.get('best')}),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
