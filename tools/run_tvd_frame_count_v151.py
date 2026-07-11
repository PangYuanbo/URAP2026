from __future__ import annotations
import json,math,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import xgboost as xgb
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.run_tvd_oof_stack_v130 import VAL,TEST,flat_stats,metrics
from tools.run_tvd_track_memory_v144 import base_scores,load_track_indices,aggregate,fuse,TRACKS
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
RUN=ROOT/"artifacts"/"detached_tvd_frame_count_v151";OUT=Path(r"D:\URAP_vatd_rank_results\tvd_frame_count_v151");TVD=Path(r"D:\urap_modal_stage\TransVisDrone");TRAIN=Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0.pkl");VATD=.93844;CLASSES=5
def report(stage,done,total=5,**x):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**x};(RUN/"progress.json").write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
def frame_feature(item):
 ds=item.get("detections") or [];scores=np.sort(np.asarray([float(r.get("score",0)) for r in ds],np.float32))[::-1];pad=np.zeros(12,np.float32);pad[:min(12,len(scores))]=scores[:12];q=np.quantile(scores,[0,.1,.25,.5,.75,.9,1]) if len(scores) else np.zeros(7);counts=[np.sum(scores>=v) for v in (.005,.01,.02,.05,.1,.2,.4,.6,.8)];areas=[];borders=[]
 for r in ds:
  b=r.get("bbox") or []
  if len(b)==4:
   w=max(0,float(b[2])-float(b[0]));h=max(0,float(b[3])-float(b[1]));areas.append(math.log1p(w*h));borders.append(min(float(b[0]),float(b[1]),max(0,1920-float(b[2])),max(0,1280-float(b[3])))/1080)
 geom=[np.mean(areas) if areas else 0,np.std(areas) if areas else 0,np.mean(borders) if borders else 0,np.std(borders) if borders else 0]
 return np.asarray([math.log1p(len(scores))/6,*pad,*q,*[math.log1p(x)/6 for x in counts],*geom],np.float32)
def frame_arrays(path,labels):
 data=load_predictionsgt(path);ids=sorted(data);x=np.stack([frame_feature(data[i]) for i in ids]);y=np.asarray([min(CLASSES-1,len(data[i].get("labels") or [])) for i in ids],np.int32) if labels else None;return data,ids,x,y
def scored(split,pkl):
 data=load_predictionsgt(pkl);c,p,t,loc,labels=flat_stats(data);raw,base=base_scores(split,data,loc);lookup={(seq,fid,idx):i for i,(seq,fid,idx,iid,r) in enumerate(loc)};tracks,_,_=load_track_indices(TRACKS[split],lookup);memory,valid=aggregate(base,tracks,3.0,"median",2);score,_=fuse(base,memory,valid,.1,"promote");ids=sorted(data);x=np.stack([frame_feature(data[i]) for i in ids]);return data,ids,x,c,p,t,loc,labels,score
def apply(score,loc,ids,expected,margin,min_k,max_k,factor,gate):
 frame_expected={iid:expected[i] for i,iid in enumerate(ids)};out=score.copy();groups={}
 for i,x in enumerate(loc):groups.setdefault(x[3],[]).append(i)
 changed=0;ks=[]
 for iid,members in groups.items():
  k=int(np.clip(np.rint(frame_expected[iid]+margin),min_k,max_k));ks.append(k);idx=np.asarray(members);order=idx[np.argsort(score[idx])[::-1]];mask=order[k:];mask=mask[score[mask]<gate];out[mask]*=factor;changed+=len(mask)
 return out,changed,float(np.mean(ks))
def main():
 OUT.mkdir(parents=True,exist_ok=True);report("load_train_frames",0);_,train_ids,tx,ty=frame_arrays(TRAIN,True);report("train_count_model",1,frames=len(tx),features=tx.shape[1],class_counts=np.bincount(ty,minlength=CLASSES).tolist());model=xgb.XGBClassifier(n_estimators=700,max_depth=7,learning_rate=.035,min_child_weight=8,subsample=.9,colsample_bytree=.9,reg_lambda=12,reg_alpha=.2,objective="multi:softprob",num_class=CLASSES,eval_metric="mlogloss",tree_method="hist",device="cuda",n_jobs=8,random_state=2026);weights=np.where(ty>0,2.0,1.0);model.fit(tx,ty,sample_weight=weights,verbose=False);OUT.mkdir(exist_ok=True);model.save_model(OUT/"frame_count.ubj");report("select_validation",2);vd,vids,vx,vc,vp,vt,vloc,vlabels,vscore=scored("val",VAL);prob=model.predict_proba(vx);expected=prob@np.arange(CLASSES);rows=[]
 for margin in (0,.5,1,1.5,2,2.5,3,4):
  for min_k in (1,2,3,4):
   for max_k in (4,5,6,8,10):
    if min_k>max_k:continue
    for factor in (.3,.4,.5,.6,.7,.8):
     for gate in (.05,.1,.15,.2):
      score,changed,mean_k=apply(vscore,vloc,vids,expected,margin,min_k,max_k,factor,gate);rows.append({"margin":margin,"min_k":min_k,"max_k":max_k,"suppression_factor":factor,"score_gate":gate,"mean_k":mean_k,"changed_rows":changed,**metrics(vc,score,vp,vt,TVD)})
 best=max(rows,key=lambda x:x["map50"]);(OUT/"val_sweep.json").write_text(json.dumps({"best":best,"top":sorted(rows,key=lambda x:-x["map50"])[:50],"expected_count_mean":float(expected.mean()),"labels":vlabels},indent=2));report("fixed_test",4,validation_selection=best);qd,qids,qx,qc,qp,qt,qloc,qlabels,qscore=scored("test",TEST);qexpected=model.predict_proba(qx)@np.arange(CLASSES);final,changed,mean_k=apply(qscore,qloc,qids,qexpected,best["margin"],best["min_k"],best["max_k"],best["suppression_factor"],best["score_gate"]);test={**metrics(qc,final,qp,qt,TVD),"expected_count_mean":float(qexpected.mean()),"mean_k":mean_k,"changed_rows":changed,"labels":qlabels,"detections":len(qloc)};gain=100*(test["map50"]-VATD);summary={"protocol":"train-only learned frame object count; validation-selected candidate budget; fixed test","validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2));report("done",5,summary=summary);return 0
if __name__=="__main__":raise SystemExit(main())

