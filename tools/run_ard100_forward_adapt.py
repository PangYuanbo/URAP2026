from __future__ import annotations
import gc,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_chunk_bidir_full import load_aux,arrays,hard_rows
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,clone_with_fused_scores
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data
O=Path(r'D:\URAP_vatd_rank_results\ard100_yolomg_generalization_v2');RUN=R/'artifacts'/'detached_ard100_forward_adapt_v1';P=RUN/'progress.json'
def rep(stage,done,**x):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':4,**x},indent=2));print(json.dumps({'stage':stage,**x}),flush=True)
def fit(x,y):
 b=(y>=.5).astype(np.int32);pos=max(1,int(b.sum()));neg=len(b)-pos;m=xgb.XGBClassifier(n_estimators=450,max_depth=7,learning_rate=.045,min_child_weight=5,subsample=.85,colsample_bytree=.8,reg_lambda=7,reg_alpha=.08,gamma=.03,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(15,neg/pos),n_jobs=8,random_state=2026);m.fit(x,b,verbose=False);return m,pos

def main():
 vf=O/'val_forward_features.jsonl';tf=O/'forward_features.jsonl'
 if not(vf.is_file() and tf.is_file()):raise FileNotFoundError('forward features incomplete')
 rep('load_features',0);vp=load_predictionsgt(O/'ard100_yolomg_val_predictionsgt.pkl');tp=load_predictionsgt(O/'ard100_yolomg_predictionsgt.pkl');va=load_aux(vf);vx,vy,vg,vloc,vseq=arrays(vp,va,va,True);del va,vp;gc.collect();ta=load_aux(tf);tx,_,tg,tloc,tseq=arrays(tp,ta,ta,False);del ta,tp;gc.collect();rep('train_folds',1,val_rows=len(vx),test_rows=len(tx),features=vx.shape[1])
 hard=hard_rows(vx,vy,vg,margin=.25,maxneg=16);seqs=sorted(set(vseq.tolist()));folds=[seqs[i::5] for i in range(5)];oof=np.zeros(len(vx),np.float32);test_preds=[];models=[];md=O/'forward_adapt_models';md.mkdir(exist_ok=True)
 for fi,held in enumerate(folds):
  train_idx=hard[~np.isin(vseq[hard],held)];hold_idx=np.flatnonzero(np.isin(vseq,held));m,pos=fit(vx[train_idx],vy[train_idx]);oof[hold_idx]=m.predict_proba(vx[hold_idx])[:,1];test_preds.append(m.predict_proba(tx)[:,1].astype(np.float32));mp=md/f'fold_{fi}.ubj';m.save_model(mp);models.append({'fold':fi,'held':held,'train_rows':len(train_idx),'positive_rows':pos,'model':str(mp)});rep('train_folds',1,fold=fi+1,folds=5,held=held,train_rows=len(train_idx),positive_rows=pos);del m;gc.collect()
 test_score=np.mean(np.stack(test_preds),axis=0).astype(np.float32);vs=O/'val_forward_adapt_scores.jsonl';ts=O/'test_forward_adapt_scores.jsonl';write_score_jsonl(vs,oof,vloc,'forward_adapt_score');write_score_jsonl(ts,test_score,tloc,'forward_adapt_score');rep('select_on_val',2,models=models)
 sweep=O/'val_forward_adapt_sweep.json';cmd=[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(O/'ard100_yolomg_val_predictionsgt.pkl'),'--tracklet-jsonl',str(vs),'--score-field','forward_adapt_score','--per-row-score','--min-tracklet-rows','1','--modes','geom-mix','logit-mix','fp-suppress','replace','--alphas','.05,.1,.2,.3,.4,.5,.6,.7,.8','--missing-score-behaviors','keep','--out-json',str(sweep)];c=subprocess.call(cmd,cwd=R,env={**os.environ,'PYTHONPATH':str(R)})
 if c:raise RuntimeError(f'val sweep failed {c}')
 best=json.loads(sweep.read_text())['best'];score_map,_=load_row_scores(ts,'forward_adapt_score',1);fused=clone_with_fused_scores(load_predictionsgt(O/'ard100_yolomg_predictionsgt.pkl'),score_map,best['mode'],best['alpha'],'keep');metrics=evaluate_data(fused,Path(r'D:\urap_modal_stage\TransVisDrone'),O);base=json.loads((O/'detector_baseline.json').read_text());summary={'protocol':'5-fold by ARD100 validation sequences; no test labels used for fitting or fusion selection','val_best':best,'test':metrics,'test_baseline':base,'gain_points':100*(metrics['map50']-base['map50']),'models':models};(O/'forward_adapt_official_summary.json').write_text(json.dumps(summary,indent=2));rep('done',4,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
