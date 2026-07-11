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
from tools.train_action_chunk_geometry_full import geometry_arrays

def selected_ranking_rows(x,y,groups,margin=.35,maxneg=28):
 keep=[];qid=[];group_id=0
 for st,sp in groups:
  positives=np.flatnonzero(y[st:sp]>=.5);negatives=np.flatnonzero(y[st:sp]<.5)
  if not len(positives) or not len(negatives):continue
  raw=x[st:sp,0];hard=negatives[raw[negatives]>=raw[positives].max()-margin];hard=hard[np.argsort(raw[hard])[::-1][:maxneg]] if len(hard) else negatives[np.argsort(raw[negatives])[::-1][:4]];local=np.sort(np.concatenate((positives,hard)));keep.extend((st+local).tolist());qid.extend([group_id]*len(local));group_id+=1
 return np.asarray(keep,np.int64),np.asarray(qid,np.int32)

def normalize_by_group(values,groups):
 output=np.zeros(len(values),np.float32)
 for st,sp in groups:
  block=np.asarray(values[st:sp],np.float32)
  if not len(block):continue
  center=float(np.median(block));scale=max(1e-3,float(block.std()));z=np.clip((block-center)/scale,-12,12);output[st:sp]=1./(1.+np.exp(-z))
 return output

def fit(x,y,groups):
 keep,qid=selected_ranking_rows(x,y,groups);relevance=np.where(y[keep]>=.5,1.+4.*np.clip(y[keep]-.5,0.,.5),0.).astype(np.float32);model=xgb.XGBRanker(n_estimators=900,max_depth=8,learning_rate=.03,min_child_weight=4,subsample=.9,colsample_bytree=.9,reg_lambda=8,reg_alpha=.08,gamma=.02,objective='rank:pairwise',eval_metric='ndcg@5',tree_method='hist',device='cuda',max_bin=256,n_jobs=8,random_state=2026);model.fit(x[keep],relevance,qid=qid,verbose=False);return model,len(keep),int((relevance>0).sum()),int(qid.max()+1)

def main():
 parser=argparse.ArgumentParser(description='Pairwise per-frame candidate ranking for the pure Action Chunk Bank.')
 for name in ('train-pkl','train-forward','train-backward','val-pkl','val-forward','val-backward','test-pkl','test-forward','test-backward','out-val-scores','out-test-scores','out-model-dir','out-summary'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--score-field',default='action_chunk_pairwise_score');args=parser.parse_args();ta,tb=load_aux(args.train_forward),load_aux(args.train_backward);tx,ty,tg,_,_=geometry_arrays(load_predictionsgt(args.train_pkl),ta,tb,True);del ta,tb;gc.collect();va,vb=load_aux(args.val_forward),load_aux(args.val_backward);vx,vy,vg,vloc,vseq=geometry_arrays(load_predictionsgt(args.val_pkl),va,vb,True);del va,vb;gc.collect();qa,qb=load_aux(args.test_forward),load_aux(args.test_backward);qx,_,qg,qloc,_=geometry_arrays(load_predictionsgt(args.test_pkl),qa,qb,False);del qa,qb;gc.collect();oof_raw=np.zeros(len(vx),np.float32);test_raw=[];models=[];args.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);model,count,pos,group_count=fit(dx,dy,groups);mask=vseq==held;oof_raw[mask]=model.predict(vx[mask]);test_raw.append(model.predict(qx));path=args.out_model_dir/f'action_chunk_pairwise_without_{held}.ubj';model.save_model(path);record={'excluded_validation_video':held,'rank_rows':count,'positive_rows':pos,'groups':group_count,'model':str(path)};models.append(record);print(json.dumps({'kind':'action_chunk_pairwise_model',**record}),flush=True);del dx,dy,model;gc.collect()
 oof=normalize_by_group(oof_raw,vg);test=normalize_by_group(np.mean(np.stack(test_raw),axis=0),qg);write_score_jsonl(args.out_val_scores,oof,vloc,args.score_field);write_score_jsonl(args.out_test_scores,test,qloc,args.score_field);summary={'model':'pure Action Chunk Bank pairwise frame ranker','normalization':'per-frame robust sigmoid','features':tx.shape[1],'train_rows':len(tx),'validation_rows':len(vx),'test_rows':len(qx),'models':models};args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
