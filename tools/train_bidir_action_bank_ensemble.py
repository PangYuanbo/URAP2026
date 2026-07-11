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
from tools.train_bidir_action_bank_oof import compact,hard_rows,sequence
def main():
 p=argparse.ArgumentParser()
 for n in ('dev-pkl','dev-forward','dev-backward','test-pkl','test-forward','test-backward','out-scores','out-model-dir','out-summary'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--score-field',default='bidir_ensemble_score');a=p.parse_args();dev_pred=load_predictionsgt(a.dev_pkl);df,dy,dg,dloc=compact(dev_pred,a.dev_forward);db,dy2,dg2,dloc2=compact(dev_pred,a.dev_backward);assert dloc==dloc2;dev=np.column_stack((df,db[:,4:],np.abs(df[:,4:]-db[:,4:]),np.minimum(df[:,4:],db[:,4:]),(df[:,4:]+db[:,4:])*.5)).astype(np.float32);seq=np.asarray([sequence(i) for i,_ in dloc]);del df,db,dev_pred;gc.collect()
 test_pred=load_predictionsgt(a.test_pkl);tf,_,tg,tloc=compact(test_pred,a.test_forward,False);tb,_,tg2,tloc2=compact(test_pred,a.test_backward,False);assert tloc==tloc2;test=np.column_stack((tf,tb[:,4:],np.abs(tf[:,4:]-tb[:,4:]),np.minimum(tf[:,4:],tb[:,4:]),(tf[:,4:]+tb[:,4:])*.5)).astype(np.float32);del tf,tb,test_pred;gc.collect();predictions=[];models=[];a.out_model_dir.mkdir(parents=True,exist_ok=True)
 for held in sorted(set(seq)):
  mask=seq!=held;parts=[];labels=[];groups=[];cursor=0
  for st,sp in dg:
   if not mask[st]:continue
   parts.append(dev[st:sp]);labels.append(dy[st:sp]);groups.append((cursor,cursor+sp-st));cursor+=sp-st
  tx=np.concatenate(parts);ty=np.concatenate(labels);keep=hard_rows(tx,ty,groups);binary=(ty[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos;model=xgb.XGBClassifier(n_estimators=700,max_depth=7,learning_rate=.04,min_child_weight=5,subsample=.85,colsample_bytree=.8,reg_lambda=6,reg_alpha=.05,gamma=.03,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);model.fit(tx[keep],binary,verbose=False);predictions.append(model.predict_proba(test)[:,1]);path=a.out_model_dir/f'model_without_{held}.ubj';model.save_model(path);models.append({'excluded_development_video':held,'train_rows':len(keep),'positive_rows':pos,'model':str(path)});print(json.dumps({'kind':'bidir_ensemble_model',**models[-1]}),flush=True)
 scores=np.mean(np.stack(predictions),axis=0).astype(np.float32);write_score_jsonl(a.out_scores,scores,tloc,a.score_field);summary={'model':'4-model leave-one-development-video-out ensemble','features':dev.shape[1],'development_rows':len(dev),'test_rows':len(test),'models':models,'fusion_fixed_from_oof':{'mode':'geom-mix','alpha':.2},'test_labels_used_for_training':False};a.out_summary.parent.mkdir(parents=True,exist_ok=True);a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
