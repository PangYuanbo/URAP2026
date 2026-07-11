from __future__ import annotations
import gc,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,clone_with_fused_scores
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
TRAIN=Path(r'D:\URAP_nps_train_tvd\route_b_official\tracklets\proposal_tracklets.jsonl');VAL=Path(r'D:\URAP_nps_val_tvd\route_b_official\tracklets\proposal_tracklets.jsonl');TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_tracklets_with_vatd.jsonl');VALPKL=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');TESTPKL=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');O=Path(r'D:\URAP_vatd_rank_results\tvd_track_memory_bank_v88');RUN=R/'artifacts'/'detached_tvd_track_memory_bank_v88';P=RUN/'progress.json'
FIELDS=['num_rows','mean_objectness','max_objectness','mean_final_score','max_final_score','mean_background','final_minus_background_mean','mean_box_side','std_box_side','mean_center_step','max_center_step','std_center_step','track_span_frames','frame_density','weak_detector_temporal_signal','score_above_02_rate','score_slope','objectness_slope','background_slope','final_margin_mean','final_margin_min','final_margin_slope','background_dominance_rate','background_dominance_longest_streak','score_above_02_longest_streak','max_frame_gap','mean_frame_gap','gap_rate','first_final_score','last_final_score','first_background','last_background']
def rep(s,d,**x):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':s,'done':d,'total':4,**x},indent=2));print(json.dumps({'stage':s,**x}),flush=True)
def read(path,labels=True):
 x=[];y=[];seq=[];items=[]
 with path.open(encoding='utf-8-sig') as f:
  for line in f:
   if not line.strip():continue
   item=json.loads(line);m=item.get('meta') or {};x.append([float(m.get(k,0) or 0) for k in FIELDS]);y.append(int(m.get('label',0)) if labels else 0);seq.append(str(m.get('seq','')));items.append(item)
 return np.asarray(x,np.float32),np.asarray(y,np.int32),np.asarray(seq),items
def fit(x,y):
 pos=max(1,int(y.sum()));neg=len(y)-pos;m=xgb.XGBClassifier(n_estimators=950,max_depth=7,learning_rate=.035,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=8,reg_alpha=.08,gamma=.02,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(25,neg/pos),n_jobs=8,random_state=2026);m.fit(x,y,verbose=False);return m,pos
def write(items,scores,path):
 with path.open('w',encoding='utf8') as out:
  for item,score in zip(items,scores):
   meta=item.get('meta') or {};rows=[]
   for row in item.get('rows') or []:rows.append({'seq':row.get('seq') or meta.get('seq'),'frame_id':row.get('frame_id'),'prediction_index':row.get('prediction_index'),'track_memory_score':float(score)})
   out.write(json.dumps({'meta':{'seq':meta.get('seq'),'track_id':meta.get('track_id')},'rows':rows},separators=(',',':'))+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);rep('load_tracklets',0);tx,ty,ts,_=read(TRAIN);vx,vy,vs,vitems=read(VAL);qx,qy,qs,qitems=read(TEST,False);oof=np.zeros(len(vx),np.float32);pred=[];models=[];rep('train_memory',1,train_tracks=len(tx),val_tracks=len(vx),test_tracks=len(qx),train_pos=int(ty.sum()),val_pos=int(vy.sum()))
 for fi,held in enumerate(sorted(set(vs))):
  mask=vs!=held;dx=np.concatenate((tx,vx[mask]));dy=np.concatenate((ty,vy[mask]));m,pos=fit(dx,dy);ho=np.flatnonzero(vs==held);oof[ho]=m.predict_proba(vx[ho])[:,1];pred.append(m.predict_proba(qx)[:,1].astype(np.float32));mp=O/f'without_{held}.ubj';m.save_model(mp);models.append({'held':held,'tracks':len(dx),'positive_tracks':pos,'model':str(mp)});rep('train_memory',1,fold=fi+1,held=held);del dx,dy,m;gc.collect()
 val_scores=O/'val_track_scores.jsonl';test_scores=O/'test_track_scores.jsonl';write(vitems,oof,val_scores);write(qitems,np.mean(np.stack(pred),0),test_scores);rep('select_val_fusion',2)
 sweep=O/'val_sweep.json';cmd=[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(VALPKL),'--tracklet-jsonl',str(val_scores),'--score-field','track_memory_score','--per-row-score','--min-tracklet-rows','1','--modes','geom-mix','logit-mix','fp-suppress','tp-boost','replace','--alphas','.01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.5,.6,.7,.8','--out-json',str(sweep)];c=subprocess.call(cmd,cwd=R,env={**os.environ,'PYTHONPATH':str(R)})
 if c:raise RuntimeError(c)
 best=json.loads(sweep.read_text())['best'];mp,_=load_row_scores(test_scores,'track_memory_score',1);fused=clone_with_fused_scores(load_predictionsgt(TESTPKL),mp,best['mode'],best['alpha'],'keep');metrics=evaluate_data(fused,Path(r'D:\urap_modal_stage\TransVisDrone'),O);summary={'protocol':'track-level memory bank trained on train+OOF val; test labels unused','features':FIELDS,'val_best':best,'test':metrics,'gain_over_vatd_points':100*(metrics['map50']-.93844),'target_3_to_5_met':.03<=metrics['map50']-.93844<=.05,'models':models};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));rep('done',4,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
