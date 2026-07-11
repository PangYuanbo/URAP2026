from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_grouped_track_rank_v170 import selected_v162
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_samurai_complement_v145 import FIELDS,SAM
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores
import tools.run_tvd_track_supported_budget_v162 as v162
RUN=ROOT/'artifacts'/'detached_tvd_v162_samurai_fusion_v173';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_v162_samurai_fusion_v173');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844
def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':3,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)
def load(split,source):
 result=selected_v162(split,source);c,p,t,loc,labels,*_,base=result[:9];path=SAM/('val_scores.jsonl' if split=='val' else 'test_scores.jsonl');scores={}
 for field in FIELDS:
  mapping,_=load_row_scores(path,field,1);scores[field]=np.asarray([float(mapping.get((seq,fid,idx),raw)) for seq,fid,idx,_image_id,raw in loc],np.float64)
 return c,p,t,loc,labels,base,scores
def combine(base,sam,alpha,mode):
 b=np.clip(base,1e-7,1-1e-7);s=np.clip(sam,1e-7,1-1e-7);out=b.copy()
 if mode=='logit':out=1/(1+np.exp(-((1-alpha)*np.log(b/(1-b))+alpha*np.log(s/(1-s)))))
 elif mode=='geom':out=np.exp((1-alpha)*np.log(b)+alpha*np.log(s))
 elif mode=='demote':
  mask=s<b;out[mask]=np.exp((1-alpha)*np.log(b[mask])+alpha*np.log(s[mask]))
 elif mode=='promote':
  mask=s>b;out[mask]=np.exp((1-alpha)*np.log(b[mask])+alpha*np.log(s[mask]))
 elif mode=='delta':
  raw_delta=np.log(s/(1-s))-np.log(np.clip(base,1e-7,1-1e-7)/np.clip(1-base,1e-7,1-1e-7));out=1/(1+np.exp(-(np.log(b/(1-b))+alpha*raw_delta)))
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);report('select_validation',0);c,p,t,loc,labels,base,sams=load('val',v162.VAL);rows=[]
 for field,sam in sams.items():
  for mode in ('logit','geom','demote','promote','delta'):
   for alpha in (.002,.005,.01,.02,.03,.05,.08,.1,.14,.2,.3,.4,.55):rows.append({'field':field,'mode':mode,'alpha':alpha,**metrics(c,combine(base,sam,alpha,mode),p,t,TVD)})
 best=max(rows,key=lambda row:float(row['map50']));baseline=metrics(c,base,p,t,TVD);(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'baseline_v162':baseline,'top':sorted(rows,key=lambda row:-float(row['map50']))[:80],'labels':labels},indent=2));report('fixed_test',2,validation_selection=best);qc,qp,qt,qloc,qlabels,qbase,qsams=load('test',v162.TEST);score=combine(qbase,qsams[best['field']],float(best['alpha']),best['mode']);test={**metrics(qc,score,qp,qt,TVD),'labels':qlabels,'detections':len(qloc)};gain=100*(test['map50']-VATD);summary={'protocol':'validation-selected camera-motion-compensated bidirectional SAMURAI memory complement on V162; fixed test','validation_selection':best,'validation_v162':baseline,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5,'target_at_least_3_met':gain>=3};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2));report('done',3,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
