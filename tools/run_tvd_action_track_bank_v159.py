from __future__ import annotations
import gc,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import xgboost as xgb
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_frame_budget_v146 import VAL,TEST
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_sequence_calibration_v158 import base as v157_base,calibrated
from tools.run_tvd_track_meta_rank_v156 import TRACKS,TRAIN,track_vector
RUN=ROOT/'artifacts'/'detached_tvd_action_track_bank_v159';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_action_track_bank_v159');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844

def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':5,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)

def load_track_training(path):
 x=[];y=[]
 with path.open(encoding='utf-8-sig') as source:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);meta=item.get('meta') or {};x.append(track_vector(meta));y.append(int(float(meta.get('label',0))>=.5))
 return np.stack(x).astype(np.float32),np.asarray(y,np.int32)

def fit(x,y):
 pos=max(1,int(y.sum()));neg=len(y)-pos;model=xgb.XGBClassifier(n_estimators=1000,max_depth=7,learning_rate=.03,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=10,reg_alpha=.1,gamma=.01,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(20.,neg/pos),n_jobs=8,random_state=2026);model.fit(x,y,verbose=False);return model,pos,neg

def row_track_scores(path,locations,model,aggregation):
 lookup={(seq,fid,idx):i for i,(seq,fid,idx,_iid,_raw) in enumerate(locations)};values=[[] for _ in locations];tracks=0;mapped=0;batch_features=[];batch_members=[]
 with path.open(encoding='utf-8-sig') as source:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);members=[]
   for row in item.get('rows') or []:
    idx=lookup.get((str(row.get('seq') or ''),int(row.get('frame_id',0)),int(row.get('prediction_index',-1))))
    if idx is not None:members.append(idx)
   if members:batch_features.append(track_vector(item.get('meta') or {}));batch_members.append(members);tracks+=1
 scores=model.predict_proba(np.stack(batch_features).astype(np.float32))[:,1]
 for score,members in zip(scores,batch_members):
  for idx in members:values[idx].append(float(score));mapped+=1
 out=np.zeros(len(locations),np.float64);valid=np.zeros(len(locations),bool)
 for i,row in enumerate(values):
  if not row:continue
  valid[i]=True
  if aggregation=='mean':out[i]=float(np.mean(row))
  elif aggregation=='median':out[i]=float(np.median(row))
  else:out[i]=float(np.max(row))
 return out,valid,tracks,mapped

def blend(base,track,valid,alpha,mode):
 out=base.copy();b=np.clip(base[valid],1e-7,1-1e-7);t=np.clip(track[valid],1e-7,1-1e-7)
 if mode=='logit':out[valid]=1/(1+np.exp(-((1-alpha)*np.log(b/(1-b))+alpha*np.log(t/(1-t)))))
 elif mode=='geom':out[valid]=np.exp((1-alpha)*np.log(b)+alpha*np.log(t))
 elif mode=='fp_suppress':out[valid]=b*((1-alpha)+alpha*t)
 elif mode=='replace':out[valid]=t
 return out

def base(split,source):
 c,p,t,loc,labels,score=v157_base(split,source);cfg=json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_sequence_calibration_v158\official_summary.json').read_text(encoding='utf-8'))['validation_selection'];score=calibrated(score,loc,cfg['kind'],float(cfg['alpha']),float(cfg['temperature']),float(cfg['offset']));return c,p,t,loc,labels,score

def main():
 OUT.mkdir(parents=True,exist_ok=True);report('load_train_tracks',0);x,y=load_track_training(TRACKS['train']);report('fit_track_bank',1,tracks=len(y),positives=int(y.sum()),features=x.shape[1]);model,pos,neg=fit(x,y);model.save_model(OUT/'action_track_bank.ubj');del x,y;gc.collect();report('select_validation',2);c,p,t,loc,labels,score=base('val',VAL);rows=[]
 for aggregation in ('max','mean','median'):
  track,valid,tracks,mapped=row_track_scores(TRACKS['val'],loc,model,aggregation)
  for mode in ('logit','geom','fp_suppress','replace'):
   alphas=(1.,) if mode=='replace' else (.005,.01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7)
   for alpha in alphas:rows.append({'aggregation':aggregation,'mode':mode,'alpha':alpha,'track_rows':int(valid.sum()),**metrics(c,blend(score,track,valid,alpha,mode),p,t,TVD)})
 best=max(rows,key=lambda x:float(x['map50']));(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'top':sorted(rows,key=lambda x:-float(x['map50']))[:50],'train_positive_tracks':pos,'train_negative_tracks':neg,'labels':labels},indent=2),encoding='utf-8');report('fixed_test',4,validation_selection=best);qc,qp,qt,qloc,qlabels,qscore=base('test',TEST);qtrack,qvalid,qtracks,qmapped=row_track_scores(TRACKS['test'],qloc,model,best['aggregation']);candidate=blend(qscore,qtrack,qvalid,float(best['alpha']),best['mode']);test={**metrics(qc,candidate,qp,qt,TVD),'labels':qlabels,'detections':len(qloc),'tracks':qtracks,'mapped_track_rows':int(qvalid.sum()),'track_memberships':qmapped};gain=100*(test['map50']-VATD);summary={'protocol':'train-only Action Track Bank; validation-selected fusion after V158; fixed test','validation_selection':best,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',5,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
