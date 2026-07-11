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
from tools.train_action_chunk_bidir_full import AUX_D,load_aux,finite,percentile

def causal_arrays(pred,forward,backward,labels):
 chunks=[];ys=[];future_weights=[];groups=[];loc=[];seqs=[];cursor=0
 for iid,item in pred.items():
  seq,fid,_=image_key(str(iid),0);ds=list(item.get('detections') or []);raw=np.asarray([finite(r.get('score')) for r in ds],dtype=np.float32);rank=percentile(raw);gap=(raw.max()-raw) if len(raw) else raw;fv=forward.get_many(seq,fid,len(ds));bv=backward.get_many(seq,fid,len(ds));rows=[];boxes=[];row_future=[]
  gt=np.asarray([r.get('bbox') for r in item.get('labels',[]) if isinstance(r.get('bbox'),list) and len(r['bbox'])==4],dtype=np.float32);gt=gt if gt.size else np.zeros((0,4),np.float32)
  for i,r in enumerate(ds):
   box=r.get('bbox')
   if not isinstance(box,list) or len(box)!=4:continue
   clipped=np.clip(raw[i],1e-6,1-1e-6);prefix=np.asarray([raw[i],math.log(clipped/(1-clipped)),rank[i],gap[i]],np.float32);rows.append(np.concatenate((prefix,fv[i])));boxes.append([finite(z) for z in box]);loc.append((str(iid),i));seqs.append(seq)
   future=np.clip(.30*bv[i,0]+.20*bv[i,1]+.15*bv[i,2]+.10*bv[i,3]+.10*bv[i,4]+.15*bv[i,8],0.,1.);row_future.append(future)
  if rows:
   chunks.append(np.stack(rows).astype(np.float32));quality=greedy_match_qualities(boxes,gt) if labels else np.zeros(len(rows),np.float32);ys.append(quality);future_weights.append(np.asarray(row_future,np.float32));groups.append((cursor,cursor+len(rows)));cursor+=len(rows)
 return np.concatenate(chunks),np.concatenate(ys),np.concatenate(future_weights),groups,loc,np.asarray(seqs)

def hard_rows(x,y,groups,margin=.24,maxneg=16):
 keep=[]
 for st,sp in groups:
  p=np.flatnonzero(y[st:sp]>=.5);n=np.flatnonzero(y[st:sp]<.5)
  if not len(p) or not len(n):continue
  raw=x[st:sp,0];hn=n[raw[n]>=raw[p].max()-margin];hn=hn[np.argsort(raw[hn])[::-1][:maxneg]] if len(hn) else n[np.argsort(raw[n])[::-1][:3]];keep.extend((st+p).tolist());keep.extend((st+hn).tolist())
 return np.asarray(sorted(set(keep)),np.int64)

def model_fit(x,y,future,groups):
 keep=hard_rows(x,y,groups);binary=(y[keep]>=.5).astype(np.int32);strength=np.clip(future[keep],0.,1.);weights=np.where(binary>0,1.+1.25*strength,1.+2.25*strength).astype(np.float32);pos=max(1,int(binary.sum()));neg=len(binary)-pos
 model=xgb.XGBClassifier(n_estimators=850,max_depth=7,learning_rate=.035,min_child_weight=5,subsample=.88,colsample_bytree=.9,reg_lambda=7,reg_alpha=.08,gamma=.025,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026)
 model.fit(x[keep],binary,sample_weight=weights,verbose=False);return model,len(keep),pos,float(weights.mean())

def main():
 parser=argparse.ArgumentParser(description='Train causal Action Chunk Bank; backward bank is supervision only and never an inference feature.')
 for name in ('train-pkl','train-forward','train-backward','val-pkl','val-forward','val-backward','test-pkl','test-forward','test-backward','out-val-scores','out-test-scores','out-model-dir','out-summary'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--score-field',default='action_chunk_causal_score');args=parser.parse_args()
 ta,tb=load_aux(args.train_forward),load_aux(args.train_backward);tx,ty,tw,tg,_,_=causal_arrays(load_predictionsgt(args.train_pkl),ta,tb,True);del ta,tb;gc.collect()
 va,vb=load_aux(args.val_forward),load_aux(args.val_backward);vx,vy,vw,vg,vloc,vseq=causal_arrays(load_predictionsgt(args.val_pkl),va,vb,True);del va,vb;gc.collect()
 qa,qb=load_aux(args.test_forward),load_aux(args.test_backward);qx,_,_,_,qloc,_=causal_arrays(load_predictionsgt(args.test_pkl),qa,qb,False);del qa,qb;gc.collect()
 test_predictions=[];oof=np.zeros(len(vx),np.float32);models=[];args.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];future=[tw];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);future.append(vw[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);dw=np.concatenate(future);model,count,pos,mean_weight=model_fit(dx,dy,dw,groups);mask=vseq==held;oof[mask]=model.predict_proba(vx[mask])[:,1];test_predictions.append(model.predict_proba(qx)[:,1]);model_path=args.out_model_dir/f'action_chunk_causal_without_{held}.ubj';model.save_model(model_path);record={'excluded_validation_video':held,'hard_rows':count,'positive_rows':pos,'mean_future_supervision_weight':mean_weight,'model':str(model_path)};models.append(record);print(json.dumps({'kind':'action_chunk_causal_model',**record}),flush=True);del dx,dy,dw,model;gc.collect()
 write_score_jsonl(args.out_val_scores,oof,vloc,args.score_field);test_scores=np.mean(np.stack(test_predictions),axis=0).astype(np.float32);write_score_jsonl(args.out_test_scores,test_scores,qloc,args.score_field)
 summary={'model':'pure causal Action Chunk Bank 4-model ensemble','inference_features':'raw detector + past-only 1s/3s Action Chunk Bank','future_information':'training sample weights only; absent from model input and deployment','features':tx.shape[1],'train_rows':len(tx),'validation_rows':len(vx),'test_rows':len(qx),'models':models};args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
