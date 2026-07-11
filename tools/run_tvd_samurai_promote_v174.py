from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_v162_samurai_fusion_v173 import load
import tools.run_tvd_track_supported_budget_v162 as v162
RUN=ROOT/'artifacts'/'detached_tvd_samurai_promote_v174';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_samurai_promote_v174');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844
def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':3,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)
def promote(base,sam,strength,mode):
 b=np.clip(base,1e-7,1-1e-7);s=np.clip(sam,1e-7,1-1e-7);out=b.copy();mask=s>b
 if mode=='geom':out[mask]=np.clip(np.exp((1-strength)*np.log(b[mask])+strength*np.log(s[mask])),0,1)
 elif mode=='logit':
  lb=np.log(b[mask]/(1-b[mask]));ls=np.log(s[mask]/(1-s[mask]));out[mask]=1/(1+np.exp(-((1-strength)*lb+strength*ls)))
 elif mode=='delta':
  lb=np.log(b[mask]/(1-b[mask]));ls=np.log(s[mask]/(1-s[mask]));out[mask]=1/(1+np.exp(-(lb+strength*(ls-lb))))
 elif mode=='ratio':out[mask]=np.clip(b[mask]*(s[mask]/b[mask])**strength,0,1)
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);report('select_validation',0);c,p,t,loc,labels,base,sams=load('val',v162.VAL);rows=[]
 for field,sam in sams.items():
  for mode in ('geom','logit','delta','ratio'):
   for strength in (.4,.5,.55,.6,.7,.85,1.,1.15,1.3,1.5,1.8,2.):rows.append({'field':field,'mode':mode,'strength':strength,**metrics(c,promote(base,sam,strength,mode),p,t,TVD)})
 best=max(rows,key=lambda row:float(row['map50']));baseline=metrics(c,base,p,t,TVD);(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'baseline_v162':baseline,'top':sorted(rows,key=lambda row:-float(row['map50']))[:80],'labels':labels},indent=2));report('fixed_test',2,validation_selection=best);qc,qp,qt,qloc,qlabels,qbase,qsams=load('test',v162.TEST);score=promote(qbase,qsams[best['field']],float(best['strength']),best['mode']);test={**metrics(qc,score,qp,qt,TVD),'labels':qlabels,'detections':len(qloc)};gain=100*(test['map50']-VATD);summary={'protocol':'extended validation-selected positive-only SAMURAI camera-motion memory promotion on V162; fixed test','validation_selection':best,'validation_v162':baseline,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5,'target_at_least_3_met':gain>=3};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2));report('done',3,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
