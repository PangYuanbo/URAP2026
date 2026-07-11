from __future__ import annotations
import argparse,gc,json,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import FEATURE_NAMES,dataset_arrays,load_auxiliary,write_score_jsonl
D=12;SHORT=8;LONG=16
SCALARS=['raw_score','raw_logit','raw_rank_percentile','raw_gap_to_max','online_action_bank_score','online_action_bank_predicted_iou','online_action_bank_center_similarity','online_action_bank_direction_similarity','online_action_bank_scale_similarity','online_action_bank_track_quality','online_action_bank_track_age_seconds','online_action_bank_acceleration_similarity','online_action_bank_motion_stability','online_action_bank_hypotheses']
def compact(pred,auxp,with_labels=True):
 a,s=load_auxiliary(auxp);x,y,_,groups,locs=dataset_arrays(pred,a,s,{},with_labels);del a
 cols=[x[:,FEATURE_NAMES.index(n)] for n in SCALARS]
 for prefix,count in [('short',SHORT),('long',LONG)]:
  start=FEATURE_NAMES.index(f'online_action_bank_{prefix}_token_0_valid');t=x[:,start:start+count*D].reshape(len(x),count,D);v=t[:,:,0]>.5;den=np.maximum(1,v.sum(1))
  for j in (1,2,3,4,5,6,9,11):cols.append((t[:,:,j]*v).sum(1)/den)
  cols.append(v.mean(1))
 z=np.column_stack(cols).astype(np.float32);del x;gc.collect();return z,y,groups,locs
def sequence(iid):return '_'.join(str(iid).split('_')[:2])
def hard_rows(x,y,groups,margin=.20,maxneg=12):
 keep=[]
 for st,sp in groups:
  pos=np.flatnonzero(y[st:sp]>=.5);neg=np.flatnonzero(y[st:sp]<.5)
  if not len(pos) or not len(neg):continue
  raw=x[st:sp,0];threshold=raw[pos].max()-margin;hn=neg[raw[neg]>=threshold]
  if not len(hn):hn=neg[np.argsort(raw[neg])[::-1][:2]]
  else:hn=hn[np.argsort(raw[hn])[::-1][:maxneg]]
  keep.extend((st+pos).tolist());keep.extend((st+hn).tolist())
 return np.asarray(sorted(set(keep)),dtype=np.int64)
def main():
 p=argparse.ArgumentParser();p.add_argument('--pkl',type=Path,required=True);p.add_argument('--forward',type=Path,required=True);p.add_argument('--backward',type=Path,required=True);p.add_argument('--out-scores',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);p.add_argument('--score-field',default='bidir_oof_score');a=p.parse_args();pred=load_predictionsgt(a.pkl);f,y,g,loc=compact(pred,a.forward);b,y2,g2,loc2=compact(pred,a.backward);assert loc==loc2 and np.allclose(y,y2)
 # Raw frame-relative columns are duplicated; combine motion in both directions and their agreement.
 z=np.column_stack((f,b[:,4:],np.abs(f[:,4:]-b[:,4:]),np.minimum(f[:,4:],b[:,4:]),(f[:,4:]+b[:,4:])*.5)).astype(np.float32);seq=np.asarray([sequence(i) for i,_ in loc]);oof=np.zeros(len(z),np.float32);folds=[]
 for held in sorted(set(seq)):
  train_mask=seq!=held;group_indices=[i for i,(st,sp) in enumerate(g) if train_mask[st]];local_groups=[];cursor=0;parts=[];labels=[]
  for gi in group_indices:
   st,sp=g[gi];parts.append(z[st:sp]);labels.append(y[st:sp]);local_groups.append((cursor,cursor+sp-st));cursor+=sp-st
  tx=np.concatenate(parts);ty=np.concatenate(labels);keep=hard_rows(tx,ty,local_groups);binary=(ty[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos
  model=xgb.XGBClassifier(n_estimators=700,max_depth=7,learning_rate=.04,min_child_weight=5,subsample=.85,colsample_bytree=.8,reg_lambda=6,reg_alpha=.05,gamma=.03,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026)
  model.fit(tx[keep],binary,verbose=False);mask=seq==held;oof[mask]=model.predict_proba(z[mask])[:,1];folds.append({'held_out':held,'train_rows':len(keep),'positive_rows':pos,'eval_rows':int(mask.sum())});print(json.dumps({'kind':'bidir_oof_fold',**folds[-1]}),flush=True)
 write_score_jsonl(a.out_scores,oof,loc,a.score_field);summary={'model':'4-fold leave-one-video-out XGB bidirectional Action Bank','features':z.shape[1],'rows':len(z),'folds':folds,'causal_forward':True,'offline_backward_candidates_only':True,'labels_not_shared_across_held_video':True};a.out_summary.parent.mkdir(parents=True,exist_ok=True);a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
