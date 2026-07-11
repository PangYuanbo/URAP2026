from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/"tools"):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import load_auxiliary,load_native,dataset_arrays,write_score_jsonl,FEATURE_NAMES
def main():
 p=argparse.ArgumentParser();
 for n in ("train-pkl","train-aux","val-pkl","val-aux","test-pkl","test-aux","out-val","out-test","out-model","out-summary"):p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--score-field",default="xgb_motion_score");a=p.parse_args()
 def load(pp,aa,labels):
  aux,sizes=load_auxiliary(aa);pred=load_predictionsgt(pp);arr=dataset_arrays(pred,aux,sizes,{},labels);del aux,pred;return arr
 train_x,train_iou,_,train_groups,_=load(a.train_pkl,a.train_aux,True);val_x,val_iou,_,val_groups,val_locs=load(a.val_pkl,a.val_aux,True);test_x,_,_,_,test_locs=load(a.test_pkl,a.test_aux,False)
 y=(train_iou>=.5).astype(np.int32);vy=(val_iou>=.5).astype(np.int32);pos=max(1,int(y.sum()));neg=len(y)-pos;model=xgb.XGBClassifier(n_estimators=900,max_depth=8,learning_rate=.045,min_child_weight=5,subsample=.85,colsample_bytree=.75,reg_lambda=4.0,reg_alpha=.05,gamma=.02,objective="binary:logistic",eval_metric="aucpr",tree_method="hist",device="cuda",max_bin=256,scale_pos_weight=min(20.0,neg/pos),n_jobs=8,random_state=2026,early_stopping_rounds=80)
 model.fit(train_x,y,eval_set=[(val_x,vy)],verbose=25);val_scores=model.predict_proba(val_x)[:,1];test_scores=model.predict_proba(test_x)[:,1];write_score_jsonl(a.out_val,val_scores,val_locs,a.score_field);write_score_jsonl(a.out_test,test_scores,test_locs,a.score_field);a.out_model.parent.mkdir(parents=True,exist_ok=True);model.save_model(a.out_model);summary={"model":"XGBClassifier GPU hist","features":len(FEATURE_NAMES),"train_rows":len(train_x),"positive_rows":pos,"val_rows":len(val_x),"test_rows":len(test_x),"best_iteration":model.best_iteration,"score_field":a.score_field,"causal_inputs":True};a.out_summary.write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
