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
from tools.train_action_chunk_candidate_context import context_arrays
from tools.train_action_chunk_neighbor_full import load_neighbor

def multi_target_rows(x,y,groups,margin=.3,maxneg=32):
 keep=[];weights=[];selected_groups=0
 for st,sp in groups:
  positives=np.flatnonzero(y[st:sp]>=.5);negatives=np.flatnonzero(y[st:sp]<.5)
  if len(positives)<2:continue
  selected_groups+=1;raw=x[st:sp,0]
  if len(negatives):
   hard=negatives[raw[negatives]>=raw[positives].max()-margin];hard=hard[np.argsort(raw[hard])[::-1][:maxneg]] if len(hard) else negatives[np.argsort(raw[negatives])[::-1][:6]];chosen=np.concatenate((positives,hard))
  else:chosen=positives
  positive_weight=1.+.35*(len(positives)-1)
  positive_set=set(positives.tolist())
  for local in chosen:keep.append(st+int(local));weights.append(positive_weight if int(local) in positive_set else 1.)
 order=np.argsort(keep);return np.asarray(keep,np.int64)[order],np.asarray(weights,np.float32)[order],selected_groups

def fit(x,y,groups):
 keep,weights,selected_groups=multi_target_rows(x,y,groups);binary=(y[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos;model=xgb.XGBClassifier(n_estimators=1300,max_depth=8,learning_rate=.025,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=10,reg_alpha=.12,gamma=.02,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(10,neg/pos),n_jobs=8,random_state=2026);model.fit(x[keep],binary,sample_weight=weights,verbose=False);return model,len(keep),pos,selected_groups

def main():
 p=argparse.ArgumentParser(description='Multi-target specialist for the pure Action Chunk Bank.')
 for name in ('train-pkl','train-forward','train-backward','train-neighbor','val-pkl','val-forward','val-backward','val-neighbor','test-pkl','test-forward','test-backward','test-neighbor','out-val-scores','out-test-scores','out-model-dir','out-summary'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--score-field',default='action_chunk_multi_expert_score');a=p.parse_args();ta,tb=load_aux(a.train_forward),load_aux(a.train_backward);tn,names=load_neighbor(a.train_neighbor);tp=load_predictionsgt(a.train_pkl);tx,ty,tg,_,_=context_arrays(tp,ta,tb,tn,True);del tp,ta,tb,tn;gc.collect();va,vb=load_aux(a.val_forward),load_aux(a.val_backward);vn,vnames=load_neighbor(a.val_neighbor);assert names==vnames;vp=load_predictionsgt(a.val_pkl);vx,vy,vg,vloc,vseq=context_arrays(vp,va,vb,vn,True);del vp,va,vb,vn;gc.collect();qa,qb=load_aux(a.test_forward),load_aux(a.test_backward);qn,qnames=load_neighbor(a.test_neighbor);assert names==qnames;qp=load_predictionsgt(a.test_pkl);qx,_,_,qloc,_=context_arrays(qp,qa,qb,qn,False);del qp,qa,qb,qn;gc.collect();oof=np.zeros(len(vx),np.float32);tests=[];models=[];a.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(vseq)):
  parts=[tx];labels=[ty];groups=list(tg);cursor=len(tx)
  for st,sp in vg:
   if vseq[st]==held:continue
   parts.append(vx[st:sp]);labels.append(vy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  dx=np.concatenate(parts);dy=np.concatenate(labels);model,count,pos,selected=fit(dx,dy,groups);mask=vseq==held;oof[mask]=model.predict_proba(vx[mask])[:,1];tests.append(model.predict_proba(qx)[:,1]);model_path=a.out_model_dir/f'action_chunk_multi_expert_without_{held}.ubj';model.save_model(model_path);record={'excluded_validation_video':held,'rows':count,'positive_rows':pos,'multi_target_frames':selected,'model':str(model_path)};models.append(record);print(json.dumps({'kind':'action_chunk_multi_expert_model',**record}),flush=True);del dx,dy,model;gc.collect()
 write_score_jsonl(a.out_val_scores,oof,vloc,a.score_field);write_score_jsonl(a.out_test_scores,np.mean(np.stack(tests),axis=0).astype(np.float32),qloc,a.score_field);summary={'model':'pure Action Chunk Bank multi-target specialist','features':tx.shape[1],'models':models};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
