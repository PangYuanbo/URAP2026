from __future__ import annotations
import argparse,gc,json,math,re,sys,zlib
from pathlib import Path
from typing import Any
import numpy as np
import xgboost as xgb
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import greedy_match_qualities,write_score_jsonl
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
D=12
TOKEN_SUMMARY_D=20
SCALARS=('score','predicted_iou','center_similarity','direction_similarity','scale_similarity','track_quality','track_age_seconds','acceleration_similarity','motion_stability','hypotheses')
def finite(v:Any)->float:
 try:z=float(v)
 except(TypeError,ValueError):return 0.
 return z if math.isfinite(z) else 0.
def token_summary(values):
 a=np.asarray(values or [],dtype=np.float32)
 if not len(a) or len(a)%D:return [0.]*TOKEN_SUMMARY_D
 a=a.reshape(-1,D);valid=a[:,0]>.5
 if not valid.any():return [0.]*TOKEN_SUMMARY_D
 selected=a[valid];means=[float(selected[:,j].mean()) for j in (1,2,3,4,5,6,9,11)];motion=np.hypot(selected[:,1],selected[:,2]);velocity_error=np.hypot(selected[:,3],selected[:,4]);acceleration=np.hypot(selected[:,5],selected[:,6]);scale_error=np.hypot(selected[:,7],selected[:,8]);compatibility=selected[:,11]
 extras=[float(motion.mean()),float(velocity_error.mean()),float(acceleration.mean()),float(scale_error.mean()),float(selected[:,10].mean()),float(motion.std()),float(velocity_error.std()),float(selected[:,9].std()),float(compatibility.std()),float(compatibility.min()),float(compatibility.max()),float(valid.mean())]
 return means+extras
AUX_D=len(SCALARS)+2*TOKEN_SUMMARY_D
class CompactAux:
 def __init__(self,keys,values):
  order=np.argsort(keys);self.keys=keys[order];self.values=values[order]
 def get_many(self,seq,fid,count):
  if count<=0:return np.zeros((0,self.values.shape[1]),np.float32)
  query=np.asarray([candidate_key(seq,fid,i) for i in range(count)],dtype=np.uint64);pos=np.searchsorted(self.keys,query);valid=pos<len(self.keys);matched=np.zeros(count,dtype=bool);matched[valid]=self.keys[pos[valid]]==query[valid];out=np.zeros((count,self.values.shape[1]),np.float32);out[matched]=self.values[pos[matched]];return out
def candidate_key(seq,fid,idx):
 match=re.search(r'(\d+)$',str(seq));sid=int(match.group(1)) if match else zlib.crc32(str(seq).encode())&0xffff
 if not 0<=int(fid)<1<<24 or not 0<=int(idx)<1<<24:raise ValueError(f'candidate key out of range: {seq=} {fid=} {idx=}')
 return np.uint64((sid<<48)|(int(fid)<<24)|int(idx))
def load_aux(path):
 key_chunks=[];value_chunks=[];keys=[];values=[]
 with Path(path).open(encoding='utf-8-sig') as source:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);meta=item.get('meta') or {}
   for r in item.get('rows') or []:
    seq=str(r.get('seq') or meta.get('seq') or '');fid=r.get('frame_id');idx=r.get('prediction_index')
    if not seq or fid is None or idx is None:continue
    short_summary=r.get('action_chunk_bank_short_token_summary');long_summary=r.get('action_chunk_bank_long_token_summary');vals=[finite(r.get('action_chunk_bank_'+n)) for n in SCALARS];vals+=([finite(v) for v in short_summary] if isinstance(short_summary,list) and len(short_summary)==TOKEN_SUMMARY_D else token_summary(r.get('action_chunk_bank_short_tokens')));vals+=([finite(v) for v in long_summary] if isinstance(long_summary,list) and len(long_summary)==TOKEN_SUMMARY_D else token_summary(r.get('action_chunk_bank_long_tokens')));keys.append(candidate_key(seq,int(fid),int(idx)));values.append(vals)
    if len(keys)>=100000:key_chunks.append(np.asarray(keys,dtype=np.uint64));value_chunks.append(np.asarray(values,dtype=np.float16));keys=[];values=[]
 if keys:key_chunks.append(np.asarray(keys,dtype=np.uint64));value_chunks.append(np.asarray(values,dtype=np.float16))
 return CompactAux(np.concatenate(key_chunks),np.concatenate(value_chunks))
def percentile(v):
 if len(v)<=1:return np.ones_like(v,dtype=np.float32)
 rank=np.argsort(np.argsort(v,kind='stable'),kind='stable');return rank.astype(np.float32)/float(len(v)-1)
def arrays(pred,forward,backward,labels):
 chunks=[];ys=[];groups=[];loc=[];seqs=[];cursor=0
 for iid,item in pred.items():
  seq,fid,_=image_key(str(iid),0);ds=list(item.get('detections') or []);raw=np.asarray([finite(r.get('score')) for r in ds],dtype=np.float32);rank=percentile(raw);gap=(raw.max()-raw) if len(raw) else raw;rows=[];boxes=[]
  gt=np.asarray([r.get('bbox') for r in item.get('labels',[]) if isinstance(r.get('bbox'),list) and len(r['bbox'])==4],dtype=np.float32);gt=gt if gt.size else np.zeros((0,4),np.float32)
  fvals=forward.get_many(seq,fid,len(ds));bvals=backward.get_many(seq,fid,len(ds))
  for i,r in enumerate(ds):
   box=r.get('bbox');
   if not isinstance(box,list) or len(box)!=4:continue
   clipped=np.clip(raw[i],1e-6,1-1e-6);prefix=np.asarray([raw[i],math.log(clipped/(1-clipped)),rank[i],gap[i]],np.float32);fv=fvals[i];bv=bvals[i];row=np.concatenate((prefix,fv,bv,np.abs(fv-bv),np.minimum(fv,bv),.5*(fv+bv)));rows.append(row);boxes.append([finite(z) for z in box]);loc.append((str(iid),i));seqs.append(seq)
  if rows:
   a=np.stack(rows).astype(np.float32);chunks.append(a);quality=greedy_match_qualities(boxes,gt) if labels else np.zeros(len(rows),np.float32);ys.append(quality);groups.append((cursor,cursor+len(rows)));cursor+=len(rows)
 return np.concatenate(chunks),np.concatenate(ys),groups,loc,np.asarray(seqs)
def hard_rows(x,y,groups,margin=.2,maxneg=12):
 keep=[]
 for st,sp in groups:
  p=np.flatnonzero(y[st:sp]>=.5);n=np.flatnonzero(y[st:sp]<.5)
  if not len(p) or not len(n):continue
  raw=x[st:sp,0];hn=n[raw[n]>=raw[p].max()-margin];hn=hn[np.argsort(raw[hn])[::-1][:maxneg]] if len(hn) else n[np.argsort(raw[n])[::-1][:2]];keep.extend((st+p).tolist());keep.extend((st+hn).tolist())
 return np.asarray(sorted(set(keep)),np.int64)
def model_fit(x,y,groups):
 keep=hard_rows(x,y,groups);b=(y[keep]>=.5).astype(np.int32);pos=max(1,int(b.sum()));neg=len(b)-pos;m=xgb.XGBClassifier(n_estimators=700,max_depth=7,learning_rate=.04,min_child_weight=5,subsample=.85,colsample_bytree=.8,reg_lambda=6,reg_alpha=.05,gamma=.03,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);m.fit(x[keep],b,verbose=False);return m,len(keep),pos
def main():
 p=argparse.ArgumentParser()
 for n in ('train-pkl','train-forward','train-backward','val-pkl','val-forward','val-backward','test-pkl','test-forward','test-backward','out-scores','out-model-dir','out-summary'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--score-field',default='action_chunk_bidir_score');a=p.parse_args();ta,tb=load_aux(a.train_forward),load_aux(a.train_backward);tx,ty,tg,_,_=arrays(load_predictionsgt(a.train_pkl),ta,tb,True);del ta,tb;gc.collect();va,vb=load_aux(a.val_forward),load_aux(a.val_backward);vx,vy,vg,_,vseq=arrays(load_predictionsgt(a.val_pkl),va,vb,True);del va,vb;gc.collect();qa,qb=load_aux(a.test_forward),load_aux(a.test_backward);qx,_,qg,qloc,_=arrays(load_predictionsgt(a.test_pkl),qa,qb,False);del qa,qb;gc.collect();preds=[];models=[];a.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);m,n,pos=model_fit(dx,dy,groups);preds.append(m.predict_proba(qx)[:,1]);path=a.out_model_dir/f'action_chunk_without_{held}.ubj';m.save_model(path);rec={'excluded_validation_video':held,'hard_rows':n,'positive_rows':pos,'model':str(path)};models.append(rec);print(json.dumps({'kind':'action_chunk_model',**rec}),flush=True);del dx,dy,m;gc.collect()
 scores=np.mean(np.stack(preds),axis=0).astype(np.float32);write_score_jsonl(a.out_scores,scores,qloc,a.score_field);summary={'model':'pure Action Chunk Bank bidirectional 4-model ensemble','features':tx.shape[1],'train_rows':len(tx),'validation_rows':len(vx),'test_rows':len(qx),'models':models,'fusion_fixed_from_oof':{'mode':'geom-mix','alpha':.2}};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
