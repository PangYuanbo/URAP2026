from __future__ import annotations

import gc
import json
import math
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT=Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.run_tvd_domain_balanced_action_v129 import fit_balanced,subset_indices_and_groups
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import CompactAux,candidate_key,finite,load_aux
from tools.train_action_chunk_neighbor_full import dataset_arrays,hard_rows,load_neighbor
from tools.sweep_tvd_predictionsgt_action_rescore import image_key

RUN=ROOT/"artifacts"/"detached_tvd_cmc_action_v137"
OUT=Path(r"D:\URAP_vatd_rank_results\tvd_cmc_action_v137")
FULL=Path(r"D:\URAP_vatd_rank_results\action_chunk_full_dev_v36")
NEIGHBOR=Path(r"D:\URAP_vatd_rank_results\action_chunk_neighbor_v44")
CMC=Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2")
TRAIN=Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0.pkl")
VAL=Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST=Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
SIZE=ROOT/"data_templates"/"nps_sequence_sizes_actual.json"
FIELD="tvd_cmc_action_score"
VATD=.93844

SAFE_FEATURES=[
 "camera_dx_norm","camera_dy_norm","camera_magnitude_norm","camera_motion_validity","camera_gap_norm",
 "cmc_score","cmc_forward_score","cmc_forward_iou","residual_speed_side","residual_speed_diag","cmc_camera_validity",
 "track_length_log","frame_density","mean_objectness","max_objectness","center_step_side","span_seconds","track_speed_side",
]

def report(stage:str,done:int,total:int=5,**extra:object)->None:
 RUN.mkdir(parents=True,exist_ok=True);payload={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**extra};(RUN/"progress.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(payload),flush=True)

def safe(v:Any,default:float=0.0)->float:
 z=finite(v);return z if math.isfinite(z) else default

def load_cmc(path:Path)->tuple[CompactAux,int]:
 key_chunks=[];value_chunks=[];keys=[];values=[];rows_total=0
 with path.open(encoding="utf-8-sig") as source:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);meta=item.get("meta") or {};rows=item.get("rows") or []
   length=max(1.0,safe(meta.get("num_rows"),len(rows)));density=safe(meta.get("frame_density"));mean_obj=safe(meta.get("mean_objectness"));max_obj=safe(meta.get("max_objectness"));mean_side=max(1.0,safe(meta.get("mean_box_side"),1.0));center_step=safe(meta.get("mean_center_step"));span=safe(meta.get("track_span_frames"));track_speed=safe(meta.get("mean_track_speed"))
   for row in rows:
    seq=str(row.get("seq") or "");fid=row.get("frame_id");idx=row.get("prediction_index")
    if not seq or fid is None or idx is None:continue
    width=max(1.0,safe(row.get("image_width"),1920.0));height=max(1.0,safe(row.get("image_height"),1080.0));diag=math.hypot(width,height)
    box=row.get("bbox") or [0,0,1,1];side=max(1.0,math.sqrt(max(1e-6,(safe(box[2])-safe(box[0]))*(safe(box[3])-safe(box[1]))))) if len(box)==4 else 1.0
    dx=safe(row.get("camera_dx"));dy=safe(row.get("camera_dy"));residual=safe(row.get("samurai_cmc_residual_speed"))
    vals=[dx/width,dy/height,math.hypot(dx,dy)/diag,safe(row.get("camera_motion_validity")),safe(row.get("camera_motion_gap_frames"))/30.0,safe(row.get("samurai_cmc_score"),.5),safe(row.get("samurai_cmc_forward_score"),.5),safe(row.get("samurai_cmc_forward_iou")),residual/side,residual/diag,safe(row.get("samurai_cmc_camera_validity")),math.log1p(length)/6.0,density,mean_obj,max_obj,center_step/mean_side,span/30.0,track_speed/mean_side]
    keys.append(candidate_key(seq,int(fid),int(idx)));values.append(vals);rows_total+=1
    if len(keys)>=100000:key_chunks.append(np.asarray(keys,np.uint64));value_chunks.append(np.asarray(values,np.float16));keys=[];values=[]
 if keys:key_chunks.append(np.asarray(keys,np.uint64));value_chunks.append(np.asarray(values,np.float16))
 return CompactAux(np.concatenate(key_chunks),np.concatenate(value_chunks)),rows_total

def cmc_for_locations(aux:CompactAux,locations:list[tuple[str,int]])->tuple[np.ndarray,float]:
 out=np.zeros((len(locations),len(SAFE_FEATURES)),np.float32);matched=0
 by_frame:dict[tuple[str,int],list[tuple[int,int]]]={}
 for output_index,(image_id,prediction_index) in enumerate(locations):
  seq,fid,_=image_key(str(image_id),0);by_frame.setdefault((seq,fid),[]).append((output_index,int(prediction_index)))
 for (seq,fid),members in by_frame.items():
  count=max(index for _,index in members)+1;frame=aux.get_many(seq,fid,count)
  for output_index,index in members:
   out[output_index]=frame[index];matched+=int(np.any(frame[index]!=0))
 return out,matched/max(1,len(locations))

def execute(command:list[str])->None:
 code=subprocess.call(command,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT)}); 
 if code:raise RuntimeError(f"command failed {code}: {command}")

def base_features(pkl:Path,forward_name:str,backward_name:str,neighbor_name:str,labels:bool,sizes:dict[str,list[int]]):
 forward,backward=load_aux(FULL/forward_name),load_aux(FULL/backward_name);neighbor,names=load_neighbor(NEIGHBOR/neighbor_name);x,y,groups,locations,sequences=dataset_arrays(load_predictionsgt(pkl),forward,backward,neighbor,labels,sizes);del forward,backward,neighbor;gc.collect();return x,y,groups,locations,sequences,names

def main()->int:
 OUT.mkdir(parents=True,exist_ok=True);sizes=json.loads(SIZE.read_text(encoding="utf-8"));report("load_base_features",0)
 tx,ty,tg,tloc,_,names=base_features(TRAIN,"train_forward.jsonl","train_backward.jsonl","train_neighbor_scores.jsonl",True,sizes)
 vx,vy,vg,vloc,vseq,vnames=base_features(VAL,"val_forward.jsonl","val_backward.jsonl","val_neighbor_scores.jsonl",True,sizes)
 qx,_,_,qloc,_,qnames=base_features(TEST,"test_forward.jsonl","test_backward.jsonl","test_neighbor_scores.jsonl",False,sizes)
 if names!=vnames or names!=qnames:raise RuntimeError("neighbor feature mismatch")
 report("load_cmc_features",1,base_features=int(tx.shape[1]),cmc_features=len(SAFE_FEATURES))
 tc,tn=load_cmc(CMC/"train_tracklets_causal_cmc.jsonl");vc,vn=load_cmc(CMC/"val_tracklets_causal_cmc.jsonl");qc,qn=load_cmc(CMC/"test_tracklets_causal_cmc.jsonl")
 tf,tcov=cmc_for_locations(tc,tloc);vf,vcov=cmc_for_locations(vc,vloc);qf,qcov=cmc_for_locations(qc,qloc);del tc,vc,qc;gc.collect()
 tx=np.concatenate((tx,tf),axis=1);vx=np.concatenate((vx,vf),axis=1);qx=np.concatenate((qx,qf),axis=1);del tf,vf,qf;gc.collect()
 train_hard=hard_rows(tx,ty,tg);sequences=sorted(set(vseq.tolist()));oof=np.zeros(len(vx),np.float32);tests=[];records=[];(OUT/"models").mkdir(exist_ok=True)
 report("train_oof_models",2,train_rows=len(tx),train_hard_rows=len(train_hard),features=int(tx.shape[1]),coverage={"train":tcov,"val":vcov,"test":qcov},source_rows={"train":tn,"val":vn,"test":qn})
 for fold,held in enumerate(sequences):
  mask=vseq!=held;selected,subgroups=subset_indices_and_groups(mask,vg);val_local=hard_rows(vx[selected],vy[selected],subgroups);val_hard=selected[val_local];fitx=np.concatenate((tx[train_hard],vx[val_hard]));fity=np.concatenate((ty[train_hard],vy[val_hard]));weights=np.concatenate((np.full(len(train_hard),.75,np.float32),np.ones(len(val_hard),np.float32)));model=fit_balanced(fitx,fity,weights,3137+fold);hold=np.flatnonzero(~mask);oof[hold]=model.predict_proba(vx[hold])[:,1];tests.append(model.predict_proba(qx)[:,1].astype(np.float32));mp=OUT/"models"/f"without_{held}.ubj";model.save_model(mp);record={"held":held,"train_hard_rows":len(train_hard),"validation_hard_rows":len(val_hard),"model":str(mp)};records.append(record);report("train_oof_models",2,model=fold+1,models=len(sequences),record=record);del fitx,fity,weights,model;gc.collect()
 vp=OUT/"val_scores.jsonl";qp=OUT/"test_scores.jsonl";write_score_jsonl(vp,oof,vloc,FIELD);write_score_jsonl(qp,np.mean(np.stack(tests),axis=0).astype(np.float32),qloc,FIELD)
 report("select_validation",3);sweep=OUT/"val_sweep.json";execute([sys.executable,str(ROOT/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",str(VAL),"--tracklet-jsonl",str(vp),"--per-row-score","--score-field",FIELD,"--modes","geom-mix","logit-mix","fp-suppress","replace","--alphas",".01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,1","--out-json",str(sweep)]);best=json.loads(sweep.read_text(encoding="utf-8"))["best"]
 report("fixed_test",4,validation_selection=best);fixed=OUT/"test_fixed.json";execute([sys.executable,str(ROOT/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",str(TEST),"--tracklet-jsonl",str(qp),"--per-row-score","--score-field",FIELD,"--modes",str(best["mode"]),"--alphas",str(best["alpha"]),"--out-json",str(fixed)]);test=json.loads(fixed.read_text(encoding="utf-8"))["best"];gain=100*(test["map50"]-VATD);summary={"protocol":"correct-label OOF Action Bank with homography camera compensation and residual motion; fixed test","safe_cmc_features":SAFE_FEATURES,"coverage":{"train":tcov,"val":vcov,"test":qcov},"models":records,"validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");report("done",5,summary=summary);return 0

if __name__=="__main__":raise SystemExit(main())
