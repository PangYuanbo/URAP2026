from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_frame_budget_v146 import VAL,TEST
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_track_competition_v157 import load as load_v157,track_scores,fuse as track_fuse
RUN=ROOT/'artifacts'/'detached_tvd_sequence_calibration_v158';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_sequence_calibration_v158');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844

def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':3,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)

def base(split,source):
 c,p,t,loc,labels,score,tracks,mapped,source_rows=load_v157(split,source);cfg=json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_track_competition_v157\official_summary.json').read_text(encoding='utf-8'))['validation_selection'];track,valid,_=track_scores(score,tracks,cfg['aggregation'],float(cfg['length_scale']),float(cfg['density_power']));score=track_fuse(score,track,valid,float(cfg['alpha']),cfg['mode']);return c,p,t,loc,labels,score

def percentile(values):
 order=np.argsort(values,kind='stable');ranks=np.empty(len(values),np.float64);ranks[order]=(np.arange(len(values),dtype=np.float64)+.5)/len(values);return ranks

def calibrated(base,loc,kind,alpha,temperature,offset):
 auxiliary=np.empty(len(base),np.float64);groups={}
 for i,x in enumerate(loc):groups.setdefault(x[0],[]).append(i)
 for ids0 in groups.values():
  ids=np.asarray(ids0,np.int64);values=np.clip(base[ids],1e-7,1-1e-7)
  if kind=='cdf':auxiliary[ids]=percentile(values)
  elif kind=='logit_z':
   logits=np.log(values/(1-values));median=np.median(logits);scale=max(np.quantile(logits,.75)-np.quantile(logits,.25),.15);z=(logits-median)/scale;auxiliary[ids]=1/(1+np.exp(-(temperature*z+offset)))
  elif kind=='relative_q95':
   q=max(float(np.quantile(values,.95)),1e-5);auxiliary[ids]=np.clip(values/q,1e-7,1-1e-7)
  else:
   q=max(float(np.quantile(values,.99)),1e-5);auxiliary[ids]=np.clip(values/q,1e-7,1-1e-7)
 b=np.clip(base,1e-7,1-1e-7);a=np.clip(auxiliary,1e-7,1-1e-7)
 return np.exp((1-alpha)*np.log(b)+alpha*np.log(a))

def main():
 OUT.mkdir(parents=True,exist_ok=True);report('select_validation',0);c,p,t,loc,labels,score=base('val',VAL);rows=[]
 for kind in ('cdf','logit_z','relative_q95','relative_q99'):
  temperatures=(.25,.5,1.,1.5,2.) if kind=='logit_z' else (1.,)
  offsets=(-4.,-3.,-2.,-1.,0.) if kind=='logit_z' else (0.,)
  for temperature in temperatures:
   for offset in offsets:
    for alpha in (.005,.01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7):
     candidate=calibrated(score,loc,kind,alpha,temperature,offset);rows.append({'kind':kind,'alpha':alpha,'temperature':temperature,'offset':offset,**metrics(c,candidate,p,t,TVD)})
 best=max(rows,key=lambda x:float(x['map50']));(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'top':sorted(rows,key=lambda x:-float(x['map50']))[:50],'labels':labels},indent=2),encoding='utf-8');report('fixed_test',2,validation_selection=best);qc,qp,qt,qloc,qlabels,qscore=base('test',TEST);candidate=calibrated(qscore,qloc,best['kind'],float(best['alpha']),float(best['temperature']),float(best['offset']));test={**metrics(qc,candidate,qp,qt,TVD),'labels':qlabels,'detections':len(qloc)};gain=100*(test['map50']-VATD);summary={'protocol':'validation-selected label-free per-sequence score calibration after V157; fixed test','validation_selection':best,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',3,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
