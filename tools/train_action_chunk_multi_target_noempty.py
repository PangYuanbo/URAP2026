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

def selected_rows(x,y,groups,empty_cap=0,margin=.3,maxneg=24):
 keep=[];weights=[]
 for st,sp in groups:
  positives=np.flatnonzero(y[st:sp]>=.5);negatives=np.flatnonzero(y[st:sp]<.5);raw=x[st:sp,0]
  if not len(negatives):chosen=positives
  elif not len(positives):chosen=np.asarray([],dtype=np.int64)
  else:
   hard=negatives[raw[negatives]>=raw[positives].max()-margin];hard=hard[np.argsort(raw[hard])[::-1][:maxneg]] if len(hard) else negatives[np.argsort(raw[negatives])[::-1][:4]];chosen=np.concatenate((positives,hard))
  positive_weight=1.+.65*max(0,len(positives)-1)
  for local in chosen:keep.append(st+int(local));weights.append(positive_weight if local in positives else 1.)
 order=np.argsort(keep);return np.asarray(keep,np.int64)[order],np.asarray(weights,np.float32)[order]
def fit(x,y,groups):
 keep,weights=selected_rows(x,y,groups);binary=(y[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos;model=xgb.XGBClassifier(n_estimators=1100,max_depth=8,learning_rate=.028,min_child_weight=4,subsample=.9,colsample_bytree=.9,reg_lambda=9,reg_alpha=.1,gamma=.02,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);model.fit(x[keep],binary,sample_weight=weights,verbose=False);return model,len(keep),pos,float(weights[binary>0].mean())
def main():
 p=argparse.ArgumentParser(description='Multi-target-aware pure Action Chunk Bank without empty-frame negative sampling.')
 for name in ('train-pkl','train-forward','train-backward','train-neighbor','val-pkl','val-forward','val-backward','val-neighbor','test-pkl','test-forward','test-backward','test-neighbor','out-val-scores','out-test-scores','out-model-dir','out-summary'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--score-field',default='action_chunk_multi_target_noempty_score');a=p.parse_args();ta,tb=load_aux(a.train_forward),load_aux(a.train_backward);tn,names=load_neighbor(a.train_neighbor);tx,ty,tg,_,_=dataset_arrays(load_predictionsgt(a.train_pkl),ta,tb,tn,True);del ta,tb,tn;gc.collect();va,vb=load_aux(a.val_forward),load_aux(a.val_backward);vn,vnames=load_neighbor(a.val_neighbor);assert names==vnames;vx,vy,vg,vloc,vseq=dataset_arrays(load_predictionsgt(a.val_pkl),va,vb,vn,True);del va,vb,vn;gc.collect();qa,qb=load_aux(a.test_forward),load_aux(a.test_backward);qn,qnames=load_neighbor(a.test_neighbor);assert names==qnames;qx,_,_,qloc,_=dataset_arrays(load_predictionsgt(a.test_pkl),qa,qb,qn,False);del qa,qb,qn;gc.collect();oof=np.zeros(len(vx),np.float32);tests=[];models=[];a.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);model,count,pos,pw=fit(dx,dy,groups);mask=vseq==held;oof[mask]=model.predict_proba(vx[mask])[:,1];tests.append(model.predict_proba(qx)[:,1]);path=a.out_model_dir/f'action_chunk_multi_target_noempty_without_{held}.ubj';model.save_model(path);record={'excluded_validation_video':held,'rows':count,'positive_rows':pos,'mean_positive_weight':pw,'model':str(path)};models.append(record);print(json.dumps({'kind':'action_chunk_multi_target_model',**record}),flush=True);del dx,dy,model;gc.collect()
 write_score_jsonl(a.out_val_scores,oof,vloc,a.score_field);write_score_jsonl(a.out_test_scores,np.mean(np.stack(tests),axis=0).astype(np.float32),qloc,a.score_field);summary={'model':'multi-target-aware pure Action Chunk Bank','empty_negative_cap':0,'models':models,'features':tx.shape[1]};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
