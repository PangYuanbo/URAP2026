from __future__ import annotations
import gc,json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
import numpy as np
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import load_neighbor,dataset_arrays,fit
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
VAL=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');D=Path(r'D:\URAP_vatd_rank_results\action_chunk_full_dev_v36');N=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_v44');O=Path(r'D:\URAP_vatd_rank_results\tvd_val_full_neighbor_v111');RUN=R/'artifacts'/'detached_tvd_val_full_neighbor_v111';P=RUN/'progress.json'
def rep(stage,done,**extra):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':stage,'done':done,'total':4,'updated':datetime.now().astimezone().isoformat(),**extra},indent=2),encoding='utf8');print(json.dumps({'stage':stage,**extra}),flush=True)
def main():
 O.mkdir(parents=True,exist_ok=True);rep('load_features',0);vf,vb=load_aux(D/'val_forward.jsonl'),load_aux(D/'val_backward.jsonl');vn,names=load_neighbor(N/'val_neighbor_scores.jsonl');vx,vy,vg,vloc,vseq=dataset_arrays(load_predictionsgt(VAL),vf,vb,vn,True);del vf,vb,vn;gc.collect();tf,tb=load_aux(D/'test_forward.jsonl'),load_aux(D/'test_backward.jsonl');tn,test_names=load_neighbor(N/'test_neighbor_scores.jsonl');assert names==test_names;tx,_,tg,tloc,tseq=dataset_arrays(load_predictionsgt(TEST),tf,tb,tn,False);del tf,tb,tn;gc.collect();seqs=sorted(set(vseq.tolist()));oof=np.zeros(len(vx),np.float32);models=[];md=O/'models';md.mkdir(exist_ok=True);rep('train_oof',1,val_rows=len(vx),test_rows=len(tx),features=vx.shape[1],folds=len(seqs))
 for fold,held in enumerate(seqs):
  train_mask=vseq!=held;selected=np.flatnonzero(train_mask);mapping=np.full(len(vx),-1,np.int64);mapping[selected]=np.arange(len(selected));groups=[]
  for start,stop in vg:
   kept=np.arange(start,stop)[train_mask[start:stop]]
   if len(kept):groups.append((int(mapping[kept[0]]),int(mapping[kept[-1]])+1))
  model,count,pos=fit(vx[selected],vy[selected],groups);hold=np.flatnonzero(~train_mask);oof[hold]=model.predict_proba(vx[hold])[:,1];path=md/f'without_{held}.ubj';model.save_model(path);models.append({'held':held,'hard_rows':count,'positive_rows':pos,'model':str(path)});rep('train_oof',1,fold=fold+1,folds=len(seqs),held=held);del model,selected,mapping;gc.collect()
 val_scores=O/'val_oof_scores.jsonl';write_score_jsonl(val_scores,oof,vloc,'tvd_val_full_neighbor_score');rep('select_val',2);sweep=O/'val_sweep.json';command=[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(VAL),'--tracklet-jsonl',str(val_scores),'--per-row-score','--score-field','tvd_val_full_neighbor_score','--modes','geom-mix','logit-mix','fp-suppress','replace','--alphas','.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7','--out-json',str(sweep)];code=subprocess.call(command,cwd=R,env={**os.environ,'PYTHONPATH':str(R)})
 if code:raise RuntimeError(code)
 best=json.loads(sweep.read_text())['best'];rep('train_final',3,val_best=best);model,count,pos=fit(vx,vy,vg);model.save_model(md/'all_validation.ubj');test_scores=model.predict_proba(tx)[:,1].astype(np.float32);test_path=O/'test_scores.jsonl';write_score_jsonl(test_path,test_scores,tloc,'tvd_val_full_neighbor_score');fixed=O/'test_fixed.json';command=[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(TEST),'--tracklet-jsonl',str(test_path),'--per-row-score','--score-field','tvd_val_full_neighbor_score','--modes',best['mode'],'--alphas',str(best['alpha']),'--out-json',str(fixed)];code=subprocess.call(command,cwd=R,env={**os.environ,'PYTHONPATH':str(R)})
 if code:raise RuntimeError(code)
 test=json.loads(fixed.read_text())['best'];summary={'protocol':'full bidirectional 1s/3s and camera-compensated neighbor Action Bank trained only on official validation domain; sequence OOF selection; fixed test','val_best':best,'final_hard_rows':count,'final_positive_rows':pos,'test':test,'gain_over_vatd_points':100*(test['map50']-.93844),'target_3_to_5_met':.03<=test['map50']-.93844<=.05,'models':models};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));rep('done',4,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
