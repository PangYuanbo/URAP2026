from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
D=12
def token_mean(row,name,field):
 t=row.get(name) or []; vals=[float(t[i+field]) for i in range(0,len(t)-D+1,D) if float(t[i])>.5];return float(np.mean(vals)) if vals else .5
def load(path):
 out={}
 with Path(path).open(encoding='utf-8-sig') as source:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);meta=item.get('meta') or {}
   for r in item.get('rows') or []:out[(str(r.get('seq') or meta.get('seq')),int(r.get('frame_id')),int(r.get('prediction_index')))]=r
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--forward',type=Path,required=True);p.add_argument('--backward',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();back=load(a.backward);a.output.parent.mkdir(parents=True,exist_ok=True);rows=0
 with a.forward.open(encoding='utf-8-sig') as source,a.output.open('w',encoding='utf-8') as target:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);meta=item.get('meta') or {};out=[]
   for f in item.get('rows') or []:
    key=(str(f.get('seq') or meta.get('seq')),int(f.get('frame_id')),int(f.get('prediction_index')));b=back.get(key,{})
    fi=float(f.get('online_action_bank_predicted_iou',0));bi=float(b.get('online_action_bank_predicted_iou',0));fs=float(f.get('online_action_bank_score',.5));bs=float(b.get('online_action_bank_score',.5))
    comps=[token_mean(f,'online_action_bank_short_tokens',11),token_mean(f,'online_action_bank_long_tokens',11),token_mean(b,'online_action_bank_short_tokens',11),token_mean(b,'online_action_bank_long_tokens',11)]
    compat=float(np.mean(comps));out.append({'seq':key[0],'frame_id':key[1],'prediction_index':key[2],'bidir_iou_mean':.5*(fi+bi),'bidir_iou_min':min(fi,bi),'bidir_score_mean':.5*(fs+bs),'bidir_compatibility':compat,'bidir_compatibility_inverse':1-compat});rows+=1
   target.write(json.dumps({'meta':meta,'rows':out},separators=(',',':'))+'\n')
 print(json.dumps({'rows':rows,'fields':['bidir_iou_mean','bidir_iou_min','bidir_score_mean','bidir_compatibility','bidir_compatibility_inverse']},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
