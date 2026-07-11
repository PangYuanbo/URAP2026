from __future__ import annotations
import gc,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
import xgboost as xgb
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import greedy_match_qualities,write_score_jsonl
from tools.sweep_tvd_predictionsgt_action_rescore import image_key,evaluate_data
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,clone_with_fused_scores
VAL=Path(r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl');TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');O=Path(r'D:\URAP_vatd_rank_results\tvd_oof_meta_stack_v87');RUN=R/'artifacts'/'detached_tvd_oof_meta_stack_v87';P=RUN/'progress.json'
SOURCES=[
 ('neighbor',Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\val_oof_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\test_scores.jsonl'),'action_chunk_neighbor_score'),
 ('expert',Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\val_expert_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\test_expert_scores.jsonl'),'action_chunk_multi_expert_score'),
 ('chain',Path(r'D:\URAP_vatd_rank_results\action_chunk_chain_consistency_v76\val_oof_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\action_chunk_chain_consistency_v76\test_scores.jsonl'),'action_chunk_chain_consistency_score'),
 ('bidir',Path(r'D:\URAP_vatd_rank_results\action_chunk_bidir_stable_v81\val_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\action_chunk_bidir_stable_v81\test_scores.jsonl'),'action_chunk_bidir_chain_score'),
 ('future',Path(r'D:\URAP_vatd_rank_results\action_chunk_frame_calibration_v78\val_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\action_chunk_frame_calibration_v78\test_scores.jsonl'),'action_chunk_future_rank'),
 ('neighbor_iou',Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_iou_v47\val_oof_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_iou_v47\test_scores.jsonl'),'action_chunk_neighbor_iou_score'),
 ('forward',Path(r'D:\URAP_vatd_rank_results\tvd_forward_adapt_v84\val_oof_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\tvd_forward_adapt_v84\test_scores.jsonl'),'tvd_forward_adapt_score'),
 ('resolution',Path(r'D:\URAP_vatd_rank_results\tvd_resolution_neighbor_v85\val_scores.jsonl'),Path(r'D:\URAP_vatd_rank_results\tvd_resolution_neighbor_v85\test_scores.jsonl'),'action_chunk_neighbor_score')]
def rep(s,d,**x):RUN.mkdir(parents=True,exist_ok=True);P.write_text(json.dumps({'stage':s,'done':d,'total':4,**x},indent=2));print(json.dumps({'stage':s,**x}),flush=True)
def rank(v):
 if len(v)<=1:return np.ones(len(v),np.float32)
 return np.argsort(np.argsort(v,kind='stable'),kind='stable').astype(np.float32)/(len(v)-1)
def arrays(data,maps,labels):
 xs=[];ys=[];groups=[];loc=[];seqs=[];cur=0
 for iid,item in data.items():
  seq,fid,_=image_key(str(iid),0);ds=item.get('detections') or [];n=len(ds)
  if not n:continue
  raw=np.asarray([float(r.get('score',0)) for r in ds],np.float32);scores=np.stack([np.asarray([float(mp.get((seq,fid,i),raw[i])) for i in range(n)],np.float32) for mp in maps],1);features=[]
  raw_rank=rank(raw);score_ranks=np.stack([rank(scores[:,j]) for j in range(scores.shape[1])],1);mean=scores.mean(1);std=scores.std(1);mn=scores.min(1);mx=scores.max(1);count=np.full(n,math.log1p(n)/6,np.float32)
  for i,row in enumerate(ds):
   vals=np.concatenate(([raw[i],raw_rank[i],raw.max()-raw[i],count[i]],scores[i],score_ranks[i],[mean[i],std[i],mn[i],mx[i],mean[i]-raw[i],mx[i]-mn[i]]));features.append(vals);loc.append((str(iid),i));seqs.append(seq)
  xs.append(np.asarray(features,np.float32));gt=np.asarray([r['bbox'] for r in item.get('labels',[]) if isinstance(r.get('bbox'),list) and len(r['bbox'])==4],np.float32);gt=gt if gt.size else np.zeros((0,4),np.float32);boxes=[r['bbox'] for r in ds];ys.append(greedy_match_qualities(boxes,gt) if labels else np.zeros(n,np.float32));groups.append((cur,cur+n));cur+=n
 return np.concatenate(xs),np.concatenate(ys),groups,loc,np.asarray(seqs)
def hard(x,y,groups):
 out=[]
 for st,sp in groups:
  p=np.flatnonzero(y[st:sp]>=.5);n=np.flatnonzero(y[st:sp]<.5);raw=x[st:sp,0]
  if not len(p):continue
  hn=n[raw[n]>=raw[p].max()-.25];hn=hn[np.argsort(raw[hn])[::-1][:24]] if len(hn) else n[np.argsort(raw[n])[::-1][:6]];out.extend((st+p).tolist());out.extend((st+hn).tolist())
 return np.asarray(sorted(set(out)),np.int64)
def fit(x,y):
 b=(y>=.5).astype(np.int32);pos=max(1,int(b.sum()));neg=len(b)-pos;m=xgb.XGBClassifier(n_estimators=900,max_depth=7,learning_rate=.03,min_child_weight=4,subsample=.88,colsample_bytree=.9,reg_lambda=8,reg_alpha=.06,gamma=.02,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,neg/pos),n_jobs=8,random_state=2026);m.fit(x,b,verbose=False);return m,pos

def main():
 O.mkdir(parents=True,exist_ok=True);rep('load_scores',0);vm=[];tm=[]
 for name,v,t,field in SOURCES:vm.append(load_row_scores(v,field,1)[0]);tm.append(load_row_scores(t,field,1)[0]);rep('load_scores',0,source=name)
 vx,vy,vg,vloc,vseq=arrays(load_predictionsgt(VAL),vm,True);tx,_,tg,tloc,tseq=arrays(load_predictionsgt(TEST),tm,False);del vm,tm;gc.collect();keep=hard(vx,vy,vg);oof=np.zeros(len(vx),np.float32);pred=[];models=[];rep('train_meta',1,val_rows=len(vx),test_rows=len(tx),features=vx.shape[1],hard_rows=len(keep))
 for fi,held in enumerate(sorted(set(vseq))):
  tr=keep[vseq[keep]!=held];ho=np.flatnonzero(vseq==held);m,pos=fit(vx[tr],vy[tr]);oof[ho]=m.predict_proba(vx[ho])[:,1];pred.append(m.predict_proba(tx)[:,1].astype(np.float32));mp=O/f'without_{held}.ubj';m.save_model(mp);models.append({'held':held,'train_rows':len(tr),'positive_rows':pos,'model':str(mp)});rep('train_meta',1,fold=fi+1,held=held);del m;gc.collect()
 vs=O/'val_oof_scores.jsonl';ts=O/'test_scores.jsonl';write_score_jsonl(vs,oof,vloc,'meta_stack_score');write_score_jsonl(ts,np.mean(np.stack(pred),0).astype(np.float32),tloc,'meta_stack_score');rep('select_val',2)
 sweep=O/'val_sweep.json';cmd=[sys.executable,str(R/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',str(VAL),'--tracklet-jsonl',str(vs),'--score-field','meta_stack_score','--per-row-score','--modes','geom-mix','logit-mix','fp-suppress','replace','--alphas','.02,.05,.08,.1,.14,.2,.3,.4,.5,.6,.7,.8,.9','--out-json',str(sweep)];c=subprocess.call(cmd,cwd=R,env={**os.environ,'PYTHONPATH':str(R)})
 if c:raise RuntimeError(c)
 best=json.loads(sweep.read_text())['best'];mp,_=load_row_scores(ts,'meta_stack_score',1);fused=clone_with_fused_scores(load_predictionsgt(TEST),mp,best['mode'],best['alpha'],'keep');metrics=evaluate_data(fused,Path(r'D:\urap_modal_stage\TransVisDrone'),O);summary={'protocol':'8-expert OOF meta stack; leave-one-validation-video; test untouched until final','sources':[x[0] for x in SOURCES],'val_best':best,'test':metrics,'gain_over_vatd_points':100*(metrics['map50']-.93844),'target_3_to_5_met':.03<=metrics['map50']-.93844<=.05,'models':models};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));rep('done',4,summary=summary);return 0
if __name__=='__main__':raise SystemExit(main())
