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
O=Path(r'D:\URAP_vatd_rank_results\tvd_forward_adapt_v84');F=Path(r'D:\URAP_vatd_rank_results\action_chunk_full_dev_v36');VAL=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');RUN=R/'artifacts'/'detached_tvd_forward_adapt_v84';P=RUN/'progress.json'
def rep(stage,done,**x):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':4,**x},indent=2));print(json.dumps({'stage':stage,**x}),flush=True)
def fit(x,y):
 b=(y>=.5).astype(np.int32);pos=max(1,int(b.sum()));neg=len(b)-pos;m=xgb.XGBClassifier(n_estimators=650,max_depth=8,learning_rate=.035,min_child_weight=4,subsample=.88,colsample_bytree=.85,reg_lambda=7,reg_alpha=.05,gamma=.02,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);m.fit(x,b,verbose=False);return m,pos

def main():
 O.mkdir(parents=True,exist_ok=True);rep('load_features',0);vp=load_predictionsgt(VAL);tp=load_predictionsgt(TEST);va=load_aux(F/'val_forward.jsonl');vx,vy,vg,vloc,vseq=arrays(vp,va,va,True);del va,vp;gc.collect();ta=load_aux(F/'test_forward.jsonl');tx,_,tg,tloc,tseq=arrays(tp,ta,ta,False);del ta,tp;gc.collect();hard=hard_rows(vx,vy,vg,margin=.22,maxneg=18);seqs=sorted(set(vseq.tolist()));folds=[[s] for s in seqs];oof=np.zeros(len(vx),np.float32);test_preds=[];models=[];md=O/'models';md.mkdir(exist_ok=True);rep('train_folds',1,val_rows=len(vx),test_rows=len(tx),hard_rows=len(hard),features=vx.shape[1],folds=len(folds))
 for fi,held in enumerate(folds):
  tr=hard[~np.isin(vseq[hard],held)];ho=np.flatnonzero(np.isin(vseq,held));m,pos=fit(vx[tr],vy[tr]);oof[ho]=m.predict_proba(vx[ho])[:,1];test_preds.append(m.predict_proba(tx)[:,1].astype(np.float32));mp=md/f'without_{held[0]}.ubj';m.save_model(mp);models.append({'held':held,'train_rows':len(tr),'positive_rows':pos,'model':str(mp)});rep('train_folds',1,fold=fi+1,folds=len(folds),held=held);del m;gc.collect()
 test_score=np.mean(np.stack(test_preds),axis=0).astype(np.float32);vs=O/'val_oof_scores.jsonl';ts=O/'test_scores.jsonl';write_score_jsonl(vs,oof,vloc,'tvd_forward_adapt_score');write_score_jsonl(ts,test_score,tloc,'tvd_forward_adapt_score');rep('select_on_val',2)
 sweep=O/'val_sweep.json';cmd=[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(VAL),'--tracklet-jsonl',str(vs),'--score-field','tvd_forward_adapt_score','--per-row-score','--min-tracklet-rows','1','--modes','geom-mix','logit-mix','fp-suppress','replace','--alphas','.05,.1,.2,.3,.4,.5,.6,.7,.8,.9','--missing-score-behaviors','keep','--out-json',str(sweep)];c=subprocess.call(cmd,cwd=R,env={**os.environ,'PYTHONPATH':str(R)})
 if c:raise RuntimeError(f'val sweep failed {c}')
 best=json.loads(sweep.read_text())['best'];score_map,_=load_row_scores(ts,'tvd_forward_adapt_score',1);fused=clone_with_fused_scores(load_predictionsgt(TEST),score_map,best['mode'],best['alpha'],'keep');metrics=evaluate_data(fused,Path(r'D:\urap_modal_stage\TransVisDrone'),O);baseline={'map50':.938417};vatd={'map50':.938440};summary={'protocol':'TVD/NPS validation-sequence OOF; test labels never used for fitting or fusion selection','val_best':best,'test':metrics,'table_detector_map50':baseline['map50'],'table_vatd_map50':vatd['map50'],'gain_over_detector_points':100*(metrics['map50']-baseline['map50']),'gain_over_vatd_points':100*(metrics['map50']-vatd['map50']),'target_3_to_5_met':.03<=metrics['map50']-vatd['map50']<=.05,'models':models};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));rep('done',4,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
