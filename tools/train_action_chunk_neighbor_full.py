from __future__ import annotations
import argparse,gc,json,math,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_bank_motion_token_listwise import greedy_match_qualities,write_score_jsonl
from tools.train_action_chunk_bidir_full import CompactAux,candidate_key,finite,load_aux,percentile
WIDTH=1280.;HEIGHT=960.
def load_neighbor(path):
 keys=[];values=[];chunks_k=[];chunks_v=[];fields=None
 with Path(path).open(encoding='utf8') as source:
  for line in source:
   item=json.loads(line);meta=item.get('meta') or {}
   for row in item.get('rows') or []:
    if fields is None:fields=sorted(k for k in row if k.startswith('action_chunk_neighbor_'))
    seq=str(row.get('seq') or meta.get('seq') or '');fid=row.get('frame_id');idx=row.get('prediction_index')
    if not seq or fid is None or idx is None:continue
    keys.append(candidate_key(seq,int(fid),int(idx)));values.append([finite(row.get(name)) for name in fields])
    if len(keys)>=100000:chunks_k.append(np.asarray(keys,np.uint64));chunks_v.append(np.asarray(values,np.float16));keys=[];values=[]
 if keys:chunks_k.append(np.asarray(keys,np.uint64));chunks_v.append(np.asarray(values,np.float16))
 return CompactAux(np.concatenate(chunks_k),np.concatenate(chunks_v)),fields
def dataset_arrays(pred,forward,backward,neighbor,labels,size_map=None):
 chunks=[];ys=[];groups=[];loc=[];seqs=[];cursor=0
 for iid,item in pred.items():
  seq,fid,_=image_key(str(iid),0);width_image,height_image=(size_map or {}).get(seq,(WIDTH,HEIGHT));width_image=float(width_image);height_image=float(height_image);ds=list(item.get('detections') or []);raw=np.asarray([finite(r.get('score')) for r in ds],np.float32);rank=percentile(raw);gap=(raw.max()-raw) if len(raw) else raw;fv=forward.get_many(seq,fid,len(ds));bv=backward.get_many(seq,fid,len(ds));nv=neighbor.get_many(seq,fid,len(ds));rows=[];boxes=[]
  gt=np.asarray([r.get('bbox') for r in item.get('labels',[]) if isinstance(r.get('bbox'),list) and len(r['bbox'])==4],np.float32);gt=gt if gt.size else np.zeros((0,4),np.float32)
  for i,r in enumerate(ds):
   box=r.get('bbox')
   if not isinstance(box,list) or len(box)!=4:continue
   x1,y1,x2,y2=[finite(z) for z in box];w=max(1e-3,x2-x1);h=max(1e-3,y2-y1);cx=.5*(x1+x2);cy=.5*(y1+y2);clipped=np.clip(raw[i],1e-6,1-1e-6);border=max(0.,min(cx,width_image-cx,cy,height_image-cy))/min(width_image,height_image);prefix=np.asarray([raw[i],math.log(clipped/(1-clipped)),rank[i],gap[i],cx/width_image,cy/height_image,w/width_image,h/height_image,w*h/(width_image*height_image),math.log(w/h),border,math.log1p(len(ds))/6.],np.float32);rows.append(np.concatenate((prefix,fv[i],bv[i],np.abs(fv[i]-bv[i]),np.minimum(fv[i],bv[i]),.5*(fv[i]+bv[i]),nv[i])));boxes.append([x1,y1,x2,y2]);loc.append((str(iid),i));seqs.append(seq)
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
 keep=hard_rows(x,y,groups);binary=(y[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos;model=xgb.XGBClassifier(n_estimators=1100,max_depth=8,learning_rate=.028,min_child_weight=4,subsample=.9,colsample_bytree=.9,reg_lambda=9,reg_alpha=.1,gamma=.02,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);model.fit(x[keep],binary,verbose=False);return model,len(keep),pos
def main():
 p=argparse.ArgumentParser(description='Pure Action Chunk Bank with true-time neighbor residual features.')
 for name in ('train-pkl','train-forward','train-backward','train-neighbor','val-pkl','val-forward','val-backward','val-neighbor','test-pkl','test-forward','test-backward','test-neighbor','out-val-scores','out-test-scores','out-model-dir','out-summary'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--score-field',default='action_chunk_neighbor_score');p.add_argument('--sequence-size-json',type=Path);a=p.parse_args();size_map=json.loads(a.sequence_size_json.read_text(encoding='utf8')) if a.sequence_size_json else None;ta,tb=load_aux(a.train_forward),load_aux(a.train_backward);tn,names=load_neighbor(a.train_neighbor);tx,ty,tg,_,_=dataset_arrays(load_predictionsgt(a.train_pkl),ta,tb,tn,True,size_map);del ta,tb,tn;gc.collect();va,vb=load_aux(a.val_forward),load_aux(a.val_backward);vn,vnames=load_neighbor(a.val_neighbor);assert names==vnames;vx,vy,vg,vloc,vseq=dataset_arrays(load_predictionsgt(a.val_pkl),va,vb,vn,True,size_map);del va,vb,vn;gc.collect();qa,qb=load_aux(a.test_forward),load_aux(a.test_backward);qn,qnames=load_neighbor(a.test_neighbor);assert names==qnames;qx,_,qg,qloc,_=dataset_arrays(load_predictionsgt(a.test_pkl),qa,qb,qn,False,size_map);del qa,qb,qn;gc.collect();oof=np.zeros(len(vx),np.float32);tests=[];models=[];a.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);model,count,pos=fit(dx,dy,groups);mask=vseq==held;oof[mask]=model.predict_proba(vx[mask])[:,1];tests.append(model.predict_proba(qx)[:,1]);path=a.out_model_dir/f'action_chunk_neighbor_without_{held}.ubj';model.save_model(path);record={'excluded_validation_video':held,'hard_rows':count,'positive_rows':pos,'model':str(path)};models.append(record);print(json.dumps({'kind':'action_chunk_neighbor_model',**record}),flush=True);del dx,dy,model;gc.collect()
 write_score_jsonl(a.out_val_scores,oof,vloc,a.score_field);write_score_jsonl(a.out_test_scores,np.mean(np.stack(tests),axis=0).astype(np.float32),qloc,a.score_field);summary={'model':'pure Action Chunk Bank with true-time neighbor residuals','neighbor_features':names,'sequence_size_json':str(a.sequence_size_json) if a.sequence_size_json else None,'features':tx.shape[1],'train_rows':len(tx),'validation_rows':len(vx),'test_rows':len(qx),'models':models};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
