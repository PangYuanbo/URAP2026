from __future__ import annotations
import argparse,gc,json,math,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import greedy_match_qualities,write_score_jsonl
from tools.train_action_chunk_bidir_full import finite,load_aux,percentile
from tools.train_action_chunk_neighbor_full import HEIGHT,WIDTH,load_neighbor
from tools.sweep_tvd_predictionsgt_action_rescore import image_key

def dataset_arrays(pred,immediate,persistent,backward,neighbor,labels):
 chunks=[];ys=[];groups=[];loc=[];seqs=[];cursor=0
 for iid,item in pred.items():
  seq,fid,_=image_key(str(iid),0);ds=list(item.get('detections') or []);raw=np.asarray([finite(r.get('score')) for r in ds],np.float32);rank=percentile(raw);gap=(raw.max()-raw) if len(raw) else raw;iv=immediate.get_many(seq,fid,len(ds));pv=persistent.get_many(seq,fid,len(ds));bv=backward.get_many(seq,fid,len(ds));nv=neighbor.get_many(seq,fid,len(ds));rows=[];boxes=[]
  gt=np.asarray([r.get('bbox') for r in item.get('labels',[]) if isinstance(r.get('bbox'),list) and len(r['bbox'])==4],np.float32);gt=gt if gt.size else np.zeros((0,4),np.float32)
  for i,r in enumerate(ds):
   box=r.get('bbox')
   if not isinstance(box,list) or len(box)!=4:continue
   x1,y1,x2,y2=[finite(z) for z in box];w=max(1e-3,x2-x1);h=max(1e-3,y2-y1);cx=.5*(x1+x2);cy=.5*(y1+y2);clipped=np.clip(raw[i],1e-6,1-1e-6);border=max(0.,min(cx,WIDTH-cx,cy,HEIGHT-cy))/min(WIDTH,HEIGHT);prefix=np.asarray([raw[i],math.log(clipped/(1-clipped)),rank[i],gap[i],cx/WIDTH,cy/HEIGHT,w/WIDTH,h/HEIGHT,w*h/(WIDTH*HEIGHT),math.log(w/h),border,math.log1p(len(ds))/6.],np.float32);rows.append(np.concatenate((prefix,pv[i],bv[i],np.abs(pv[i]-bv[i]),np.minimum(pv[i],bv[i]),.5*(pv[i]+bv[i]),iv[i],pv[i]-iv[i],np.abs(pv[i]-iv[i]),nv[i])));boxes.append([x1,y1,x2,y2]);loc.append((str(iid),i));seqs.append(seq)
  if rows:chunks.append(np.stack(rows).astype(np.float32));ys.append(greedy_match_qualities(boxes,gt) if labels else np.zeros(len(rows),np.float32));groups.append((cursor,cursor+len(rows)));cursor+=len(rows)
 return np.concatenate(chunks),np.concatenate(ys),groups,loc,np.asarray(seqs)

def hard_rows(x,y,groups,margin=.3,maxneg=24):
 keep=[]
 for st,sp in groups:
  p=np.flatnonzero(y[st:sp]>=.5);n=np.flatnonzero(y[st:sp]<.5)
  if not len(n):keep.extend((st+p).tolist());continue
  raw=x[st:sp,0]
  if not len(p):keep.extend((st+n[np.argsort(raw[n])[::-1][:6]]).tolist());continue
  hard=n[raw[n]>=raw[p].max()-margin];hard=hard[np.argsort(raw[hard])[::-1][:maxneg]] if len(hard) else n[np.argsort(raw[n])[::-1][:4]];keep.extend((st+p).tolist());keep.extend((st+hard).tolist())
 return np.asarray(sorted(set(keep)),np.int64)

def fit(x,y,groups):
 keep=hard_rows(x,y,groups);binary=(y[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos;model=xgb.XGBClassifier(n_estimators=1200,max_depth=8,learning_rate=.026,min_child_weight=4,subsample=.9,colsample_bytree=.88,reg_lambda=10,reg_alpha=.12,gamma=.02,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);model.fit(x[keep],binary,verbose=False);return model,len(keep),pos

def main():
 p=argparse.ArgumentParser(description='Dual-memory pure Action Chunk model: immediate plus persistent 1s/3s banks.')
 for split in ('train','val','test'):
  for suffix in ('pkl','immediate','persistent','backward','neighbor'):p.add_argument(f'--{split}-{suffix}',type=Path,required=True)
 for name in ('out-val-scores','out-test-scores','out-model-dir','out-summary'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--score-field',default='action_chunk_dual_memory_score');a=p.parse_args()
 def load(split,labels):
  immediate=load_aux(getattr(a,f'{split}_immediate'));persistent=load_aux(getattr(a,f'{split}_persistent'));backward=load_aux(getattr(a,f'{split}_backward'));neighbor,names=load_neighbor(getattr(a,f'{split}_neighbor'));pred=load_predictionsgt(getattr(a,f'{split}_pkl'));result=dataset_arrays(pred,immediate,persistent,backward,neighbor,labels);del pred,immediate,persistent,backward,neighbor;gc.collect();return result,names
 (tx,ty,tg,_,_),names=load('train',True);(vx,vy,vg,vloc,vseq),vnames=load('val',True);(qx,_,_,qloc,_),qnames=load('test',False);assert names==vnames==qnames;oof=np.zeros(len(vx),np.float32);tests=[];models=[];a.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);model,count,pos=fit(dx,dy,groups);mask=vseq==held;oof[mask]=model.predict_proba(vx[mask])[:,1];tests.append(model.predict_proba(qx)[:,1]);model_path=a.out_model_dir/f'action_chunk_dual_memory_without_{held}.ubj';model.save_model(model_path);record={'excluded_validation_video':held,'hard_rows':count,'positive_rows':pos,'model':str(model_path)};models.append(record);print(json.dumps({'kind':'action_chunk_dual_memory_model',**record}),flush=True);del dx,dy,model;gc.collect()
 write_score_jsonl(a.out_val_scores,oof,vloc,a.score_field);write_score_jsonl(a.out_test_scores,np.mean(np.stack(tests),axis=0).astype(np.float32),qloc,a.score_field);summary={'model':'pure Action Chunk dual immediate/persistent memory','features':tx.shape[1],'models':models};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
