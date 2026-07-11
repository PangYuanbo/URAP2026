from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.run_tvd_oof_stack_v130 import flat_stats,metrics
from tools.run_tvd_track_supported_budget_v162 import base as v162_base,apply as budget_apply
from tools.run_tvd_track_supported_budget_v162 import OUT as V162_OUT
from tools.run_tvd_temporal_nms_v154 import iou
BEST=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');LAST=Path(r'U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone\runs\val\NPS_URAP\official_best_last_tta_val\predictionsgt\predictionsgt_split_0.pkl');OUT=Path(r'D:\URAP_vatd_rank_results\tvd_ensemble_action_val_fusion_v164');TVD=Path(r'D:\urap_modal_stage\TransVisDrone')

def match(last,best_data,loc,threshold,kind):
 out=np.asarray([x[4] for x in loc],np.float64);matched=np.zeros(len(loc),bool);lookup={}
 for idx,(seq,fid,row,image_id,raw) in enumerate(loc):lookup.setdefault(image_id,[]).append(idx)
 for image_id,ids0 in lookup.items():
  ids=np.asarray(ids0,np.int64);last_rows=(last.get(image_id) or {}).get('detections') or []
  if not last_rows:continue
  boxes=np.asarray([best_data[image_id]['detections'][loc[i][2]]['bbox'] for i in ids],np.float64);last_boxes=np.asarray([r['bbox'] for r in last_rows],np.float64);last_scores=np.asarray([float(r.get('score',0)) for r in last_rows]);
  for pos,i in enumerate(ids):
   overlaps=iou(boxes[pos],last_boxes);j=int(np.argmax(overlaps))
   if overlaps[j]>=threshold:
    matched[i]=True
    if kind=='max':out[i]=max(out[i],last_scores[j])
    elif kind=='mean':out[i]=.5*(out[i]+last_scores[j])
    elif kind=='geom':out[i]=np.sqrt(max(out[i],1e-9)*max(last_scores[j],1e-9))
    else:out[i]=last_scores[j]
 return out,matched

def main():
 OUT.mkdir(parents=True,exist_ok=True);best_data=load_predictionsgt(BEST);last=load_predictionsgt(LAST);c,p,t,loc,labels,base,support,length,tracks,mapped,source=v162_base('val',BEST);import tools.run_tvd_track_supported_budget_v162 as module;module.length_global=length;cfg=json.loads((V162_OUT/'official_summary.json').read_text())['validation_selection'];base,_,_=budget_apply(base,support,loc,int(cfg['top_k']),float(cfg['suppression_factor']),float(cfg['score_gate']),float(cfg['promotion_alpha']),int(cfg['minimum_track_rows']));rows=[]
 for threshold in (.3,.4,.5,.6,.7,.8):
  for kind in ('max','mean','geom','last'):
   alt,matched=match(last,best_data,loc,threshold,kind)
   for alpha in (.005,.01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,1.):
    candidate=np.exp((1-alpha)*np.log(np.clip(base,1e-9,1))+alpha*np.log(np.clip(alt,1e-9,1)));rows.append({'iou_threshold':threshold,'kind':kind,'alpha':alpha,'matched_rows':int(matched.sum()),**metrics(c,candidate,p,t,TVD)})
 best=max(rows,key=lambda x:float(x['map50']));(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'top':sorted(rows,key=lambda x:-float(x['map50']))[:50],'v162_validation':cfg,'labels':labels},indent=2),encoding='utf-8');print(json.dumps(best,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())


