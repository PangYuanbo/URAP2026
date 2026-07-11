from __future__ import annotations
import gc,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.run_tvd_domain_balanced_action_v129 import TRAIN,VAL,TEST,FULL,NEIGHBOR,SIZE_MAP,fit_balanced
from tools.run_tvd_oof_stack_v130 import flat_stats,metrics
from tools.run_tvd_sequence_calibration_v158 import base as v157_base,calibrated
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,fuse_score
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import load_neighbor,dataset_arrays,hard_rows
RUN=ROOT/'artifacts'/'detached_tvd_trainval_refit_v160';OUT=Path(r'D:\URAP_vatd_rank_results\tvd_trainval_refit_v160');TVD=Path(r'D:\urap_modal_stage\TransVisDrone');VATD=.93844;V129=Path(r'D:\URAP_vatd_rank_results\tvd_domain_balanced_action_v129')

def report(stage,done,**extra):
 RUN.mkdir(parents=True,exist_ok=True);payload={'stage':stage,'done':done,'total':6,'updated':datetime.now().astimezone().isoformat(),**extra};(RUN/'progress.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)

def features(source,forward_path,backward_path,neighbor_path,labels,size_map):
 forward,backward=load_aux(forward_path),load_aux(backward_path);neighbor,names=load_neighbor(neighbor_path);x,y,groups,locations,sequences=dataset_arrays(load_predictionsgt(source),forward,backward,neighbor,labels,size_map);del forward,backward,neighbor;gc.collect();return x,y,groups,locations,sequences,names

def advanced_base(split,source):
 c,p,t,loc,labels,score=v157_base(split,source);cfg=json.loads(Path(r'D:\URAP_vatd_rank_results\tvd_sequence_calibration_v158\official_summary.json').read_text(encoding='utf-8'))['validation_selection'];score=calibrated(score,loc,cfg['kind'],float(cfg['alpha']),float(cfg['temperature']),float(cfg['offset']));return c,p,t,loc,labels,score

def combine(base,learned,alpha,mode):
 b=np.clip(base,1e-7,1-1e-7);l=np.clip(learned,1e-7,1-1e-7)
 if mode=='logit':return 1/(1+np.exp(-((1-alpha)*np.log(b/(1-b))+alpha*np.log(l/(1-l)))))
 if mode=='geom':return np.exp((1-alpha)*np.log(b)+alpha*np.log(l))
 if mode=='fp_suppress':return b*((1-alpha)+alpha*l)
 return (1-alpha)*b+alpha*l

def main():
 OUT.mkdir(parents=True,exist_ok=True);size_map=json.loads(SIZE_MAP.read_text(encoding='utf-8'));summary129=json.loads((V129/'official_summary.json').read_text(encoding='utf-8'));selected_ratio=float(summary129['selected_ratio']);record=next(row for row in summary129['ratios'] if float(row['ratio'])==selected_ratio);field=record['field'];cfg=summary129['validation_selection']
 report('select_oof_fusion',0,selected_ratio=selected_ratio);vc,vp,vt,vloc,vlabels,vbase=advanced_base('val',VAL);oof_map,_=load_row_scores(Path(record['val_scores']),field,1);oof=np.asarray([fuse_score(raw,float(oof_map.get((seq,fid,idx),raw)),float(cfg['alpha']),str(cfg['mode'])) for seq,fid,idx,_iid,raw in vloc]);rows=[]
 for mode in ('logit','geom','fp_suppress','linear'):
  for alpha in (.01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,.85,1.):rows.append({'mode':mode,'alpha':alpha,**metrics(vc,combine(vbase,oof,alpha,mode),vp,vt,TVD)})
 best=max(rows,key=lambda x:float(x['map50']));(OUT/'val_oof_sweep.json').write_text(json.dumps({'best':best,'top':sorted(rows,key=lambda x:-float(x['map50']))[:40],'v129_config':cfg,'selected_ratio':selected_ratio},indent=2),encoding='utf-8');del vc,vp,vt,vbase,oof,oof_map;gc.collect()
 report('load_train_features',1,validation_selection=best);tx,ty,tg,_tloc,_tseq,names=features(TRAIN,FULL/'train_forward.jsonl',FULL/'train_backward.jsonl',NEIGHBOR/'train_neighbor_scores.jsonl',True,size_map);train_keep=hard_rows(tx,ty,tg);report('load_validation_features',2,train_rows=len(tx),train_hard_rows=len(train_keep));vx,vy,vg,_vloc2,_vseq2,vnames=features(VAL,FULL/'val_forward.jsonl',FULL/'val_backward.jsonl',NEIGHBOR/'val_neighbor_scores.jsonl',True,size_map);assert names==vnames;val_keep=hard_rows(vx,vy,vg);fit_x=np.concatenate((tx[train_keep],vx[val_keep]),axis=0);fit_y=np.concatenate((ty[train_keep],vy[val_keep]),axis=0);weights=np.concatenate((np.full(len(train_keep),.75,np.float32),np.ones(len(val_keep),np.float32)));del tx,ty,tg,vx,vy,vg;gc.collect();report('fit_trainval_model',3,train_hard_rows=len(train_keep),validation_hard_rows=len(val_keep),fit_rows=len(fit_y));model=fit_balanced(fit_x,fit_y,weights,2026160);model.save_model(OUT/'trainval_refit.ubj');del fit_x,fit_y,weights;gc.collect()
 report('load_test_features',4);qx,_qy,_qg,qlocations,_qseq,qnames=features(TEST,FULL/'test_forward.jsonl',FULL/'test_backward.jsonl',NEIGHBOR/'test_neighbor_scores.jsonl',False,size_map);assert names==qnames;learned=model.predict_proba(qx)[:,1].astype(np.float64);del qx,model;gc.collect();qc,qp,qt,base_locations,qlabels,qbase=advanced_base('test',TEST);expected=[(item[3],item[2]) for item in base_locations]
 if qlocations!=expected:raise RuntimeError('test feature/candidate order mismatch')
 refit=np.asarray([fuse_score(raw,float(pred),float(cfg['alpha']),str(cfg['mode'])) for raw,pred in zip((x[4] for x in base_locations),learned)]);score=combine(qbase,refit,float(best['alpha']),best['mode']);test={**metrics(qc,score,qp,qt,TVD),'labels':qlabels,'detections':len(base_locations)};gain=100*(test['map50']-VATD);summary={'protocol':'OOF-selected fusion; final Action ranker refit on train+validation; untouched fixed test','validation_selection':best,'v129_selection':cfg,'selected_ratio':selected_ratio,'test_fixed':test,'vatd_map50':VATD,'gain_over_vatd_points':gain,'target_3_to_5_met':3<=gain<=5};(OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');report('done',6,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
