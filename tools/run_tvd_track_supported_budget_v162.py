from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_frame_budget_v146 import VAL,TEST
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_sequence_calibration_v158 import base as v157_base,calibrated
from tools.run_tvd_track_competition_v157 import load_track_indices,TRACKS
RUN=ROOT/'artifacts'/'detached_tvd_track_supported_budget_v162';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_track_supported_budget_v162');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844

def report(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':3,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)
def base(split,source):
 c,p,t,loc,labels,score=v157_base(split,source);cfg=json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_sequence_calibration_v158\official_summary.json').read_text(encoding='utf-8'))['validation_selection'];score=calibrated(score,loc,cfg['kind'],float(cfg['alpha']),float(cfg['temperature']),float(cfg['offset']));lookup={(seq,fid,idx):i for i,(seq,fid,idx,_iid,_raw) in enumerate(loc)};tracks,mapped,source_rows=load_track_indices(TRACKS[split],lookup);support=np.zeros(len(loc),np.float64);length=np.ones(len(loc),np.int32)
 for track in tracks:
  ids=np.asarray([x[1] for x in track],np.int64);values=score[ids];reliability=1-np.exp(-len(ids)/8.);value=float(values.max())*reliability;better=value>support[ids];chosen=ids[better];support[chosen]=value;length[chosen]=len(ids)
 support=np.maximum(support,score*.25);return c,p,t,loc,labels,score,support,length,len(tracks),mapped,source_rows

def apply(base,support,loc,k,suppress_factor,score_gate,promote_alpha,min_length):
 out=base.copy();frames={};changed=promoted=0
 for i,x in enumerate(loc):frames.setdefault(x[3],[]).append(i)
 for ids0 in frames.values():
  ids=np.asarray(ids0,np.int64);eligible=ids[length_global[ids]>=min_length];order=eligible[np.argsort(support[eligible])[::-1]] if len(eligible) else eligible;keep=set(order[:k].tolist());demote=np.asarray([i for i in ids if i not in keep and base[i]<score_gate],np.int64);out[demote]*=suppress_factor;changed+=len(demote)
  promote=np.asarray(list(keep),np.int64)
  if len(promote) and promote_alpha>0:
   b=np.clip(out[promote],1e-7,1);s=np.clip(support[promote],1e-7,1);out[promote]=np.exp((1-promote_alpha)*np.log(b)+promote_alpha*np.log(s));promoted+=len(promote)
 return out,changed,promoted

def main():
 global length_global
 OUT.mkdir(parents=True,exist_ok=True);report('select_validation',0);c,p,t,loc,labels,score,support,length_global,tracks,mapped,source_rows=base('val',VAL);rows=[]
 for k in (1,2,3,4,5,8):
  for factor in (.1,.3,.5,.7,.9):
   for gate in (.03,.05,.1,.2,.4):
    for alpha in (0.,.02,.05,.1,.2):
     for minimum in (1,3,5,8):
      candidate,changed,promoted=apply(score,support,loc,k,factor,gate,alpha,minimum);rows.append({'top_k':k,'suppression_factor':factor,'score_gate':gate,'promotion_alpha':alpha,'minimum_track_rows':minimum,'changed_rows':changed,'promoted_rows':promoted,**metrics(c,candidate,p,t,TVD)})
 best=max(rows,key=lambda x:float(x['map50']));(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'top':sorted(rows,key=lambda x:-float(x['map50']))[:50],'labels':labels,'tracks':tracks,'mapped':mapped},indent=2),encoding='utf-8');report('fixed_test',2,validation_selection=best);qc,qp,qt,qloc,qlabels,qscore,qsupport,length_global,qtracks,qmapped,qsource=base('test',TEST);candidate,changed,promoted=apply(qscore,qsupport,qloc,int(best['top_k']),float(best['suppression_factor']),float(best['score_gate']),float(best['promotion_alpha']),int(best['minimum_track_rows']));test={**metrics(qc,candidate,qp,qt,TVD),'labels':qlabels,'detections':len(qloc),'tracks':qtracks,'mapped':qmapped,'changed_rows':changed,'promoted_rows':promoted};gain=100*(test['map50']-VATD);summary={'protocol':'validation-selected label-free track-supported per-frame candidate budget after V158; fixed test','validation_selection':best,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',3,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
