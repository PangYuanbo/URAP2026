from __future__ import annotations
import argparse,gc,json,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import dataset_arrays,load_neighbor

def hard_rows(x,y,groups,margin=.35,maxneg=28):
 keep=[]
 for st,sp in groups:
  positives=np.flatnonzero(y[st:sp]>=.5);negatives=np.flatnonzero(y[st:sp]<.5)
  if not len(negatives):keep.extend((st+positives).tolist());continue
  raw=x[st:sp,0]
  if not len(positives):keep.extend((st+negatives[np.argsort(raw[negatives])[::-1][:6]]).tolist());continue
  hard=negatives[raw[negatives]>=raw[positives].max()-margin];hard=hard[np.argsort(raw[hard])[::-1][:maxneg]] if len(hard) else negatives[np.argsort(raw[negatives])[::-1][:5]];keep.extend((st+positives).tolist());keep.extend((st+hard).tolist())
 return np.asarray(sorted(set(keep)),np.int64)
def fit(x,y,groups):
 keep=hard_rows(x,y,groups);target=np.clip(y[keep],0,1).astype(np.float32);weights=np.where(target>=.5,2.+2.*target,1.+2.*x[keep,0]).astype(np.float32);model=xgb.XGBRegressor(n_estimators=1200,max_depth=8,learning_rate=.025,min_child_weight=4,subsample=.9,colsample_bytree=.9,reg_lambda=10,reg_alpha=.1,gamma=.015,objective='reg:squarederror',eval_metric='rmse',tree_method='hist',device='cuda',max_bin=256,n_jobs=8,random_state=2026);model.fit(x[keep],target,sample_weight=weights,verbose=False);return model,len(keep),int((target>=.5).sum())
def main():
 p=argparse.ArgumentParser(description='Continuous IoU regression for neighbor-enhanced Action Chunk Bank.')
 for name in ('train-pkl','train-forward','train-backward','train-neighbor','val-pkl','val-forward','val-backward','val-neighbor','test-pkl','test-forward','test-backward','test-neighbor','out-val-scores','out-test-scores','out-model-dir','out-summary'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--score-field',default='action_chunk_neighbor_iou_score');a=p.parse_args();ta,tb=load_aux(a.train_forward),load_aux(a.train_backward);tn,names=load_neighbor(a.train_neighbor);tx,ty,tg,_,_=dataset_arrays(load_predictionsgt(a.train_pkl),ta,tb,tn,True);del ta,tb,tn;gc.collect();va,vb=load_aux(a.val_forward),load_aux(a.val_backward);vn,vnames=load_neighbor(a.val_neighbor);assert names==vnames;vx,vy,vg,vloc,vseq=dataset_arrays(load_predictionsgt(a.val_pkl),va,vb,vn,True);del va,vb,vn;gc.collect();qa,qb=load_aux(a.test_forward),load_aux(a.test_backward);qn,qnames=load_neighbor(a.test_neighbor);assert names==qnames;qx,_,_,qloc,_=dataset_arrays(load_predictionsgt(a.test_pkl),qa,qb,qn,False);del qa,qb,qn;gc.collect();oof=np.zeros(len(vx),np.float32);tests=[];models=[];a.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);model,count,pos=fit(dx,dy,groups);mask=vseq==held;oof[mask]=np.clip(model.predict(vx[mask]),0,1);tests.append(np.clip(model.predict(qx),0,1));path=a.out_model_dir/f'action_chunk_neighbor_iou_without_{held}.ubj';model.save_model(path);record={'excluded_validation_video':held,'hard_rows':count,'positive_rows':pos,'model':str(path)};models.append(record);print(json.dumps({'kind':'action_chunk_neighbor_iou_model',**record}),flush=True);del dx,dy,model;gc.collect()
 write_score_jsonl(a.out_val_scores,oof,vloc,a.score_field);write_score_jsonl(a.out_test_scores,np.mean(np.stack(tests),axis=0).astype(np.float32),qloc,a.score_field);summary={'model':'pure Action Chunk Bank continuous IoU regression','features':tx.shape[1],'models':models};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
