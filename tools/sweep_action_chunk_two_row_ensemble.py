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

def parse(text):return [float(x) for x in text.replace(',',' ').split()]
def logit(x):x=np.clip(x,1e-6,1-1e-6);return np.log(x/(1-x))
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--tvd-root',type=Path,required=True);p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--score-a',type=Path,required=True);p.add_argument('--field-a',required=True);p.add_argument('--score-b',type=Path,required=True);p.add_argument('--field-b',required=True);p.add_argument('--alphas',required=True);p.add_argument('--betas',required=True);p.add_argument('--out-json',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.tvd_root.resolve()));
 if not hasattr(np,'trapz') and hasattr(np,'trapezoid'):np.trapz=np.trapezoid
 from utils.metrics import ap_per_class
 data=load_predictionsgt(a.predictionsgt_pkl);sa,_=load_row_scores(a.score_a,a.field_a,1);sb,_=load_row_scores(a.score_b,a.field_b,1);iouv=torch.linspace(.5,.95,10);correct=[];raw=[];av=[];bv=[];pc=[];tc=[];labels=dets=0
 for iid in sorted(data):
  item=data[iid];dr=[];indices=[]
  for idx,r in enumerate(item.get('detections',[])):
   value=row_to_det(r)
   if value is not None:dr.append(value);indices.append(idx)
  lr=[value for r in item.get('labels',[]) if (value:=row_to_label(r)) is not None];dets+=len(dr);labels+=len(lr);dt=torch.tensor(dr,dtype=torch.float32) if dr else torch.zeros((0,6));lt=torch.tensor(lr,dtype=torch.float32) if lr else torch.zeros((0,5));correct.append(process_batch(dt,lt,iouv).numpy());tc.extend(lt[:,0].tolist() if lt.numel() else [])
  for idx,row in zip(indices,dr):
   key=image_key(str(iid),idx);raw.append(float(row[4]));av.append(float(sa.get(key,row[4])));bv.append(float(sb.get(key,row[4])));pc.append(float(row[5]))
 correct=np.concatenate(correct);raw=np.asarray(raw);av=np.asarray(av);bv=np.asarray(bv);pc=np.asarray(pc,np.float32);tc=np.asarray(tc,np.float32);rows=[];best=None
 for alpha in parse(a.alphas):
  for beta in parse(a.betas):
   if alpha+beta>1:continue
   confidence=sigmoid((1-alpha-beta)*logit(raw)+alpha*logit(av)+beta*logit(bv));precision,recall,ap,f1,_=ap_per_class(correct,confidence,pc,tc,plot=False,save_dir=a.out_json.parent,names={0:'drone'});row={'alpha':alpha,'beta':beta,'raw_weight':1-alpha-beta,'images':len(data),'labels':labels,'detections':dets,'precision':float(precision.mean()),'recall':float(recall.mean()),'map50':float(ap[:,0].mean()),'map5095':float(ap.mean(1).mean()),'f1':float(f1.mean())};rows.append(row)
   if best is None or row['map50']>best['map50']:best=row
 summary={'field_a':a.field_a,'field_b':a.field_b,'best':best,'top':sorted(rows,key=lambda x:(-x['map50'],-x['recall']))[:30],'rows':rows};a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps({'kind':'action_chunk_two_row_ensemble_done','best':best}),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
