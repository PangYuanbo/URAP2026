from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt,process_batch,row_to_det,row_to_label
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores

def parse(x):return [float(v) for v in x.replace(',',' ').split()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tvd-root',type=Path,required=True);p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--stable-scores',type=Path,required=True);p.add_argument('--stable-field',required=True);p.add_argument('--empty-scores',type=Path,required=True);p.add_argument('--empty-field',required=True);p.add_argument('--base-alphas',required=True);p.add_argument('--betas',required=True);p.add_argument('--gammas',required=True);p.add_argument('--out-json',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.tvd_root.resolve()));
 if not hasattr(np,'trapz') and hasattr(np,'trapezoid'):np.trapz=np.trapezoid
 from utils.metrics import ap_per_class
 data=load_predictionsgt(a.predictionsgt_pkl);stable,_=load_row_scores(a.stable_scores,a.stable_field,1);empty,_=load_row_scores(a.empty_scores,a.empty_field,1);iouv=torch.linspace(.5,.95,10);correct=[];raw=[];sv=[];ev=[];pc=[];tc=[];labels=dets=0
 for iid in sorted(data):
  item=data[iid];dr=[];idxs=[]
  for i,r in enumerate(item.get('detections',[])):
   d=row_to_det(r)
   if d is not None:dr.append(d);idxs.append(i)
  lr=[v for r in item.get('labels',[]) if (v:=row_to_label(r)) is not None];dets+=len(dr);labels+=len(lr);dt=torch.tensor(dr,dtype=torch.float32) if dr else torch.zeros((0,6));lt=torch.tensor(lr,dtype=torch.float32) if lr else torch.zeros((0,5));correct.append(process_batch(dt,lt,iouv).numpy());tc.extend(lt[:,0].tolist() if lt.numel() else [])
  for i,d in zip(idxs,dr):key=image_key(str(iid),i);raw.append(float(d[4]));sv.append(float(stable.get(key,d[4])));ev.append(float(empty.get(key,d[4])));pc.append(float(d[5]))
 correct=np.concatenate(correct);raw=np.asarray(raw);sv=np.asarray(sv);ev=np.asarray(ev);pc=np.asarray(pc,np.float32);tc=np.asarray(tc,np.float32);rows=[];best=None
 for alpha in parse(a.base_alphas):
  base=np.exp((1-alpha)*np.log(np.maximum(raw,1e-9))+alpha*np.log(np.maximum(sv,1e-9)))
  for beta in parse(a.betas):
   for gamma in parse(a.gammas):
    confidence=np.clip(base*(1-beta*(1-ev)*np.power(1-sv,gamma)),0,1);precision,recall,ap,f1,_=ap_per_class(correct,confidence,pc,tc,plot=False,save_dir=a.out_json.parent,names={0:'drone'});row={'base_alpha':alpha,'beta':beta,'gamma':gamma,'images':len(data),'labels':labels,'detections':dets,'precision':float(precision.mean()),'recall':float(recall.mean()),'map50':float(ap[:,0].mean()),'map5095':float(ap.mean(1).mean()),'f1':float(f1.mean())};rows.append(row)
    if best is None or row['map50']>best['map50']:best=row
 summary={'best':best,'top':sorted(rows,key=lambda x:(-x['map50'],-x['recall']))[:30],'rows':rows};a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps({'kind':'conditional_suppress_done','best':best}));return 0
if __name__=='__main__':raise SystemExit(main())
