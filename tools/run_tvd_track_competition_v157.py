from __future__ import annotations
import json, math, sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_frame_budget_v146 import VAL,TEST
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_temporal_nms_v154 import base as structural_base,suppress
from tools.run_tvd_track_memory_v144 import TRACKS,load_track_indices
RUN=ROOT/'artifacts'/'detached_tvd_track_competition_v157';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_track_competition_v157');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844

def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':3,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)

def load(split,source):
 correct,pred,target,locations,labels,score,boxes=structural_base(split,source)
 selected=json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_temporal_nms_v154\official_summary.json').read_text(encoding='utf-8'))['validation_selection']
 score,_=suppress(score,locations,boxes,float(selected['iou_threshold']),float(selected['suppression_factor']),float(selected['minimum_score']))
 lookup={(seq,fid,idx):row for row,(seq,fid,idx,_iid,_raw) in enumerate(locations)}
 tracks,mapped,source_rows=load_track_indices(TRACKS[split],lookup)
 return correct,pred,target,locations,labels,score,tracks,mapped,source_rows

def aggregate(values,kind):
 if kind=='mean':return float(values.mean())
 if kind=='median':return float(np.median(values))
 if kind=='q75':return float(np.quantile(values,.75))
 if kind=='top3':return float(np.mean(np.sort(values)[-3:]))
 return float(values.max())

def track_scores(base,tracks,kind,length_scale,density_power):
 output=base.copy();support=np.zeros(len(base),np.int32);track_value=np.zeros(len(base),np.float64)
 for track in tracks:
  ids=np.asarray([x[1] for x in track],np.int64);frames=np.asarray([x[0] for x in track],np.int64);length=len(ids);span=max(1,int(frames.max()-frames.min()+1));density=length/span
  reliability=1.0-math.exp(-length/length_scale);quality=aggregate(base[ids],kind)*(reliability**.5)*(density**density_power)
  better=quality>track_value[ids];chosen=ids[better];track_value[chosen]=quality;support[chosen]=length
 valid=support>0;output[valid]=track_value[valid];return output,valid,support

def fuse(base,track,valid,alpha,mode):
 out=base.copy();b=np.clip(base[valid],1e-6,1-1e-6);t=np.clip(track[valid],1e-6,1-1e-6)
 if mode=='logit':out[valid]=1/(1+np.exp(-((1-alpha)*np.log(b/(1-b))+alpha*np.log(t/(1-t)))))
 elif mode=='geom':out[valid]=np.exp((1-alpha)*np.log(b)+alpha*np.log(t))
 elif mode=='demote':
  mask=valid.copy();mask[valid]=t<b;out[mask]=np.exp((1-alpha)*np.log(np.clip(base[mask],1e-6,1))+alpha*np.log(np.clip(track[mask],1e-6,1)))
 else:out[valid]=(1-alpha)*b+alpha*t
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=True);report('select_validation',0);c,p,t,loc,labels,base,tracks,mapped,source_rows=load('val',VAL);rows=[]
 for kind in ('mean','median','q75','top3','max'):
  for length_scale in (3.,8.,16.):
   for density_power in (0.,.5):
    track,valid,support=track_scores(base,tracks,kind,length_scale,density_power)
    for mode in ('logit','geom','demote'):
     for alpha in (.02,.05,.1,.2,.35,.5,.7):
      rows.append({'aggregation':kind,'length_scale':length_scale,'density_power':density_power,'mode':mode,'alpha':alpha,'mapped_rows':int(valid.sum()),**metrics(c,fuse(base,track,valid,alpha,mode),p,t,TVD)})
 best=max(rows,key=lambda x:float(x['map50']));(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'top':sorted(rows,key=lambda x:-float(x['map50']))[:50],'labels':labels,'tracks':len(tracks),'mapped':mapped,'source_rows':source_rows},indent=2),encoding='utf-8');report('fixed_test',2,validation_selection=best)
 qc,qp,qt,qloc,qlabels,qbase,qtracks,qmapped,qsource=load('test',TEST);qtrack,qvalid,qsupport=track_scores(qbase,qtracks,best['aggregation'],float(best['length_scale']),float(best['density_power']));qscore=fuse(qbase,qtrack,qvalid,float(best['alpha']),best['mode']);test={**metrics(qc,qscore,qp,qt,TVD),'labels':qlabels,'detections':len(qloc),'tracks':len(qtracks),'mapped':qmapped,'memory_rows':int(qvalid.sum())};gain=100*(test['map50']-VATD);summary={'protocol':'validation-selected global track competition memory after V154; fixed test','validation_selection':best,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',3,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
