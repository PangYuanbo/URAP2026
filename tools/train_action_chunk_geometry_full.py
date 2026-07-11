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
from tools.train_action_chunk_bidir_full import AUX_D,finite,load_aux,percentile
WIDTH=1280.;HEIGHT=960.

def geometry_arrays(pred,forward,backward,labels):
 chunks=[];ys=[];groups=[];loc=[];seqs=[];cursor=0
 for iid,item in pred.items():
  seq,fid,_=image_key(str(iid),0);ds=list(item.get('detections') or []);raw=np.asarray([finite(r.get('score')) for r in ds],np.float32);rank=percentile(raw);gap=(raw.max()-raw) if len(raw) else raw;fv=forward.get_many(seq,fid,len(ds));bv=backward.get_many(seq,fid,len(ds));rows=[];boxes=[]
  gt=np.asarray([r.get('bbox') for r in item.get('labels',[]) if isinstance(r.get('bbox'),list) and len(r['bbox'])==4],np.float32);gt=gt if gt.size else np.zeros((0,4),np.float32)
  for i,r in enumerate(ds):
   box=r.get('bbox')
   if not isinstance(box,list) or len(box)!=4:continue
   x1,y1,x2,y2=[finite(z) for z in box];width=max(1e-3,x2-x1);height=max(1e-3,y2-y1);cx=.5*(x1+x2);cy=.5*(y1+y2);clipped=np.clip(raw[i],1e-6,1-1e-6);border=max(0.,min(cx,WIDTH-cx,cy,HEIGHT-cy))/min(WIDTH,HEIGHT);prefix=np.asarray([raw[i],math.log(clipped/(1-clipped)),rank[i],gap[i],cx/WIDTH,cy/HEIGHT,width/WIDTH,height/HEIGHT,width*height/(WIDTH*HEIGHT),math.log(width/height),border,math.log1p(len(ds))/6.],np.float32);rows.append(np.concatenate((prefix,fv[i],bv[i],np.abs(fv[i]-bv[i]),np.minimum(fv[i],bv[i]),.5*(fv[i]+bv[i]))));boxes.append([x1,y1,x2,y2]);loc.append((str(iid),i));seqs.append(seq)
  if rows:
   chunks.append(np.stack(rows).astype(np.float32));ys.append(greedy_match_qualities(boxes,gt) if labels else np.zeros(len(rows),np.float32));groups.append((cursor,cursor+len(rows)));cursor+=len(rows)
 return np.concatenate(chunks),np.concatenate(ys),groups,loc,np.asarray(seqs)

def hard_rows(x,y,groups,margin=.28,maxneg=20):
 keep=[]
 for st,sp in groups:
  positives=np.flatnonzero(y[st:sp]>=.5);negatives=np.flatnonzero(y[st:sp]<.5)
  if not len(positives) or not len(negatives):continue
  raw=x[st:sp,0];hard=negatives[raw[negatives]>=raw[positives].max()-margin];hard=hard[np.argsort(raw[hard])[::-1][:maxneg]] if len(hard) else negatives[np.argsort(raw[negatives])[::-1][:3]];keep.extend((st+positives).tolist());keep.extend((st+hard).tolist())
 return np.asarray(sorted(set(keep)),np.int64)

def fit(x,y,groups):
 keep=hard_rows(x,y,groups);binary=(y[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos;model=xgb.XGBClassifier(n_estimators=900,max_depth=8,learning_rate=.032,min_child_weight=5,subsample=.88,colsample_bytree=.88,reg_lambda=8,reg_alpha=.08,gamma=.025,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);model.fit(x[keep],binary,verbose=False);return model,len(keep),pos

def main():
 parser=argparse.ArgumentParser(description='Pure Action Chunk Bank with detector geometry context and strict held-video OOF.')
 for name in ('train-pkl','train-forward','train-backward','val-pkl','val-forward','val-backward','test-pkl','test-forward','test-backward','out-val-scores','out-test-scores','out-model-dir','out-summary'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--score-field',default='action_chunk_geometry_score');args=parser.parse_args();ta,tb=load_aux(args.train_forward),load_aux(args.train_backward);tx,ty,tg,_,_=geometry_arrays(load_predictionsgt(args.train_pkl),ta,tb,True);del ta,tb;gc.collect();va,vb=load_aux(args.val_forward),load_aux(args.val_backward);vx,vy,vg,vloc,vseq=geometry_arrays(load_predictionsgt(args.val_pkl),va,vb,True);del va,vb;gc.collect();qa,qb=load_aux(args.test_forward),load_aux(args.test_backward);qx,_,_,qloc,_=geometry_arrays(load_predictionsgt(args.test_pkl),qa,qb,False);del qa,qb;gc.collect();oof=np.zeros(len(vx),np.float32);tests=[];models=[];args.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);model,count,pos=fit(dx,dy,groups);mask=vseq==held;oof[mask]=model.predict_proba(vx[mask])[:,1];tests.append(model.predict_proba(qx)[:,1]);path=args.out_model_dir/f'action_chunk_geometry_without_{held}.ubj';model.save_model(path);record={'excluded_validation_video':held,'hard_rows':count,'positive_rows':pos,'model':str(path)};models.append(record);print(json.dumps({'kind':'action_chunk_geometry_model',**record}),flush=True);del dx,dy,model;gc.collect()
 write_score_jsonl(args.out_val_scores,oof,vloc,args.score_field);write_score_jsonl(args.out_test_scores,np.mean(np.stack(tests),axis=0).astype(np.float32),qloc,args.score_field);summary={'model':'pure Action Chunk Bank plus detector geometry context','image_geometry':[int(WIDTH),int(HEIGHT)],'features':tx.shape[1],'train_rows':len(tx),'validation_rows':len(vx),'test_rows':len(qx),'models':models};args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
