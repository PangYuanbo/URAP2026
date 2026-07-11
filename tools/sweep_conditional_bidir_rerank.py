from __future__ import annotations
import argparse,copy,json,math,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO/'tools') not in sys.path:sys.path.insert(0,str(REPO/'tools'))
from eval_tvd_predictionsgt_pkl import load_predictionsgt
from sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
def load_scores(path):
 out={}
 with Path(path).open(encoding='utf-8-sig') as source:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);meta=item.get('meta') or {}
   for r in item.get('rows') or []:out[(str(r.get('seq') or meta.get('seq')),int(r['frame_id']),int(r['prediction_index']))]=float(r['bidir_compatibility_inverse'])
 return out
def logit(v):v=min(1-1e-6,max(1e-6,float(v)));return math.log(v/(1-v))
def sigmoid(v):return 1/(1+math.exp(-max(-30,min(30,v))))
def apply(data,scores,gate,floor,relative,beta):
 out={};changed=0
 for iid,item in data.items():
  ds=[dict(r) for r in item.get('detections',[])];raw=[float(r.get('score',0)) for r in ds]
  if raw:
   top=max(raw)
   if top<=gate:
    eligible=[j for j in range(len(ds)) if raw[j]>=floor and raw[j]>=top-relative and scores.get(image_key(str(iid),j)) is not None]
    if len(eligible)>1:
     original_order=sorted(eligible,key=lambda j:raw[j],reverse=True)
     motion_order=sorted(eligible,key=lambda j:logit(raw[j])+beta*scores[image_key(str(iid),j)],reverse=True)
     sorted_scores=sorted((raw[j] for j in eligible),reverse=True)
     for rank,j in enumerate(motion_order):r=ds[j];r['score']=sorted_scores[rank]
     changed+=sum(a!=b for a,b in zip(original_order,motion_order))
  out[iid]={'labels':item.get('labels',[]),'detections':ds}
 return out,changed
def main():
 p=argparse.ArgumentParser();p.add_argument('--pkl',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--tvd-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();data=load_predictionsgt(a.pkl);scores=load_scores(a.scores);rows=[]
 for gate in (.65,.70,.75):
  for floor in (.20,.30):
   for relative in (.10,.20,.40):
    for beta in (1.5,2.0,2.5,3.0):
     adjusted,changed=apply(data,scores,gate,floor,relative,beta);metrics=evaluate_data(adjusted,a.tvd_root,a.out.parent);row={'gate':gate,'floor':floor,'relative':relative,'beta':beta,'changed':changed,**metrics};rows.append(row);print(json.dumps({'kind':'conditional_bidir_trial',**row}),flush=True)
 best=max(rows,key=lambda r:r['map50']);result={'best':best,'top':sorted(rows,key=lambda r:r['map50'],reverse=True)[:20]};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
