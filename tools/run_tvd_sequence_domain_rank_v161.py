from __future__ import annotations
import gc,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_tvd_domain_balanced_action_v129 import TRAIN,VAL,TEST,FULL,NEIGHBOR,SIZE_MAP,fit_balanced
from tools.run_tvd_oof_stack_v130 import metrics
from tools.run_tvd_trainval_refit_v160 import features,advanced_base,combine
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score
from tools.train_action_chunk_neighbor_full import hard_rows
RUN=ROOT/'artifacts'/'detached_tvd_sequence_domain_rank_v161';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_sequence_domain_rank_v161');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844

def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':8,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)

def percentile(values):
 order=np.argsort(values,kind='stable');ranks=np.empty(len(values),np.float32);ranks[order]=(np.arange(len(values),dtype=np.float32)+.5)/len(values);return ranks

def augment(x,sequences):
 extras=np.zeros((len(x),15),np.float32);raw=np.clip(x[:,0].astype(np.float64),1e-7,1-1e-7);logits=np.log(raw/(1-raw))
 for sequence in sorted(set(sequences.tolist())):
  ids=np.flatnonzero(sequences==sequence);r=raw[ids];l=logits[ids];q=np.quantile(r,[.25,.5,.75,.9,.95,.99]);median=float(np.median(l));iqr=max(float(np.quantile(l,.75)-np.quantile(l,.25)),.15);mx=max(float(r.max()),1e-7);mean=float(r.mean());std=max(float(r.std()),1e-6)
  extras[ids]=np.column_stack((percentile(r),r/mx,r/max(float(q[3]),1e-7),r/max(float(q[4]),1e-7),r/max(float(q[5]),1e-7),(l-median)/iqr,(r-mean)/std,np.full(len(ids),mean),np.full(len(ids),std),np.full(len(ids),q[1]),np.full(len(ids),q[3]),np.full(len(ids),q[4]),np.full(len(ids),q[5]),np.full(len(ids),mx),np.full(len(ids),np.log1p(len(ids))/15.)))
 return np.concatenate((x,extras),axis=1)

def align_predictions(locations,predictions,base_locations):
 mapping={(str(image_id),int(index)):float(score) for (image_id,index),score in zip(locations,predictions)};aligned=np.empty(len(base_locations),np.float64);missing=[]
 for row,(_seq,_fid,index,image_id,raw) in enumerate(base_locations):
  key=(str(image_id),int(index));value=mapping.get(key)
  if value is None:missing.append(key);value=float(raw)
  aligned[row]=value
 if missing:raise RuntimeError(f'missing {len(missing)} prediction keys; first={missing[:5]}')
 return aligned

def model_score(raw,learned,alpha=.3):
 return np.asarray([fuse_score(float(r),float(v),alpha,'logit-mix') for r,v in zip(raw,learned)],np.float64)

def main():
 OUT.mkdir(parents=True,exist_ok=True);size_map=json.loads(SIZE_MAP.read_text(encoding='utf-8'));report('load_train_features',0);tx,ty,tg,tloc,tseq,names=features(TRAIN,FULL/'train_forward.jsonl',FULL/'train_backward.jsonl',NEIGHBOR/'train_neighbor_scores.jsonl',True,size_map);tx=augment(tx,tseq);train_keep=hard_rows(tx,ty,tg);report('load_validation_features',1,train_rows=len(tx),train_hard_rows=len(train_keep),features=tx.shape[1]);vx,vy,vg,vloc,vseq,vnames=features(VAL,FULL/'val_forward.jsonl',FULL/'val_backward.jsonl',NEIGHBOR/'val_neighbor_scores.jsonl',True,size_map);assert names==vnames;vx=augment(vx,vseq);report('fit_train_only',2,validation_rows=len(vx));model=fit_balanced(tx[train_keep],ty[train_keep],np.ones(len(train_keep),np.float32),2026161);model.save_model(OUT/'train_only.ubj');learned_val=model.predict_proba(vx)[:,1].astype(np.float64);vc,vp,vt,base_vloc,vlabels,vbase=advanced_base('val',VAL);aligned_val=align_predictions(vloc,learned_val,base_vloc);raw_val=np.asarray([x[4] for x in base_vloc]);rank_val=model_score(raw_val,aligned_val);rows=[]
 for mode in ('logit','geom','fp_suppress','linear'):
  for alpha in (.005,.01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,.85,1.):rows.append({'mode':mode,'alpha':alpha,**metrics(vc,combine(vbase,rank_val,alpha,mode),vp,vt,TVD)})
 best=max(rows,key=lambda x:float(x['map50']));(OUT/'val_sweep.json').write_text(json.dumps({'best':best,'top':sorted(rows,key=lambda x:-float(x['map50']))[:40],'features':tx.shape[1]},indent=2),encoding='utf-8');report('refit_train_validation',4,validation_selection=best);val_keep=hard_rows(vx,vy,vg);fit_x=np.concatenate((tx[train_keep],vx[val_keep]),axis=0);fit_y=np.concatenate((ty[train_keep],vy[val_keep]),axis=0);weights=np.concatenate((np.full(len(train_keep),.75,np.float32),np.ones(len(val_keep),np.float32)));del tx,ty,tg,vx,vy,vg,model;gc.collect();model=fit_balanced(fit_x,fit_y,weights,2026162);model.save_model(OUT/'trainval_refit.ubj');del fit_x,fit_y,weights;gc.collect();report('load_test_features',6,train_hard_rows=len(train_keep),validation_hard_rows=len(val_keep));qx,_qy,_qg,qloc,qseq,qnames=features(TEST,FULL/'test_forward.jsonl',FULL/'test_backward.jsonl',NEIGHBOR/'test_neighbor_scores.jsonl',False,size_map);assert names==qnames;qx=augment(qx,qseq);learned_test=model.predict_proba(qx)[:,1].astype(np.float64);del qx,model;gc.collect();qc,qp,qt,base_qloc,qlabels,qbase=advanced_base('test',TEST);aligned_test=align_predictions(qloc,learned_test,base_qloc);raw_test=np.asarray([x[4] for x in base_qloc]);rank_test=model_score(raw_test,aligned_test);score=combine(qbase,rank_test,float(best['alpha']),best['mode']);test={**metrics(qc,score,qp,qt,TVD),'labels':qlabels,'detections':len(base_qloc),'mapped_scores':len(aligned_test)};gain=100*(test['map50']-VATD);summary={'protocol':'train-only validation selection; sequence-domain normalized Action ranker refit on train+validation; untouched fixed test','features':int(len(names) if names else 0)+15,'validation_selection':best,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',8,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
