from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from typing import Any
import numpy as np
import xgboost as xgb
from train_detection_row_score_head import feature_row,iou_max,load_gt,write_scored_test,load_test_features,_float

def load_rank_rows(paths:list[Path],gt:dict[tuple[str,int],np.ndarray],min_score:float):
 records=[]
 for path in paths:
  with path.open('r',encoding='utf-8-sig') as f:
   for line in f:
    if not line.strip():continue
    item=json.loads(line);meta=dict(item.get('meta') or {});rows=[dict(r) for r in item.get('rows') or []]
    for idx,row in enumerate(rows):
     seq=str(row.get('seq') or meta.get('seq') or '');frame=int(float(row.get('frame_id',0) or 0));bbox=row.get('bbox');raw=max(_float(row.get('score')),_float(row.get('objectness')),_float(row.get('final_drone_score')))
     if not seq or not isinstance(bbox,list) or len(bbox)!=4 or raw<min_score:continue
     iou=iou_max([_float(v) for v in bbox],gt.get((seq,frame),np.zeros((0,4),dtype=np.float32)))
     relevance=3.0 if iou>=0.7 else 2.0 if iou>=0.5 else 1.0 if iou>=0.3 else 0.0
     records.append(((seq,frame),feature_row(row,meta,idx,len(rows)),relevance,raw))
 records.sort(key=lambda r:r[0]);x=np.asarray([r[1] for r in records],dtype=np.float32);y=np.asarray([r[2] for r in records],dtype=np.float32);qid=[];last=None;group=-1
 for key,*_ in records:
  if key!=last:group+=1;last=key
  qid.append(group)
 return x,y,np.asarray(qid,dtype=np.uint32),records

def main():
 p=argparse.ArgumentParser();p.add_argument('--train-tracklets',nargs='+',type=Path,required=True);p.add_argument('--train-gt-csv',nargs='+',type=Path,required=True);p.add_argument('--test-tracklets',type=Path,required=True);p.add_argument('--out-test-tracklets',type=Path,required=True);p.add_argument('--out-model',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);p.add_argument('--score-field',default='lambda_rank_score');p.add_argument('--min-score',type=float,default=0.001);p.add_argument('--rounds',type=int,default=900);p.add_argument('--max-depth',type=int,default=7);p.add_argument('--eta',type=float,default=.03);a=p.parse_args()
 gt=load_gt(a.train_gt_csv);x,y,qid,records=load_rank_rows(a.train_tracklets,gt,a.min_score);xt=load_test_features(a.test_tracklets)
 d=xgb.QuantileDMatrix(x,label=y,qid=qid);params={'objective':'rank:ndcg','eval_metric':'ndcg@10','tree_method':'hist','device':'cuda','lambdarank_pair_method':'topk','lambdarank_num_pair_per_sample':16,'max_depth':a.max_depth,'eta':a.eta,'min_child_weight':8,'subsample':.9,'colsample_bytree':.9,'lambda':8.0,'alpha':.1,'max_bin':256,'seed':2026};b=xgb.train(params,d,num_boost_round=a.rounds,evals=[(d,'train')],verbose_eval=25);dt=xgb.QuantileDMatrix(xt,ref=d);scores=b.predict(dt);write=write_scored_test(a.test_tracklets,scores,a.out_test_tracklets,a.score_field);a.out_model.parent.mkdir(parents=True,exist_ok=True);b.save_model(a.out_model);summary={'device':'cuda','train_rows':len(y),'groups':int(qid.max()+1),'relevance_counts':{str(i):int((y==i).sum()) for i in range(4)},'params':params,'test_mean':float(scores.mean()),**write};a.out_summary.parent.mkdir(parents=True,exist_ok=True);a.out_summary.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
