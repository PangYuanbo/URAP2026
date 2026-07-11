from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.run_tvd_oof_stack_v130 import VAL,TEST,flat_stats,load_predictionsgt,metrics,blend
from tools.run_tvd_track_memory_v144 import base_scores,load_track_indices,aggregate,fuse,TRACKS
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores
RUN=ROOT/"artifacts"/"detached_tvd_samurai_complement_v145";OUT=Path(r"D:\URAP_vatd_rank_results\tvd_samurai_complement_v145");SAM=Path(r"D:\URAP_vatd_rank_results\tvd_samurai_memory_v93");TVD=Path(r"D:\urap_modal_stage\TransVisDrone");VATD=.93844
FIELDS=("samurai_memory_sym_0p94","samurai_memory_min_0p94","samurai_memory_sym_0p98","samurai_memory_min_0p98","samurai_memory_sym_0p995","samurai_memory_min_0p995","samurai_memory_span_1s","samurai_memory_span_3s","samurai_memory_motion","samurai_memory_motion_sym")
def report(stage,done,total=3,**x):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**x};(RUN/"progress.json").write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
def routes(split,pkl):
 data=load_predictionsgt(pkl);correct,pred,target,loc,labels=flat_stats(data);raw,base=base_scores(split,data,loc);lookup={(seq,fid,idx):i for i,(seq,fid,idx,iid,r) in enumerate(loc)};tracks,_,_=load_track_indices(TRACKS[split],lookup);memory,valid=aggregate(base,tracks,3.0,"median",2);v144,_=fuse(base,memory,valid,.1,"promote");path=SAM/("val_scores.jsonl" if split=="val" else "test_scores.jsonl");sam={}
 for field in FIELDS:
  m,_=load_row_scores(path,field,1);sam[field]=np.asarray([float(m.get((seq,fid,idx),r)) for seq,fid,idx,iid,r in loc])
 return correct,pred,target,loc,labels,v144,sam
def mix(a,b,w,mode):return blend([a,b],(1-w,w),mode)
def main():
 OUT.mkdir(parents=True,exist_ok=True);report("select_validation",0);c,p,t,loc,labels,base,sam=routes("val",VAL);rows=[]
 for field,score in sam.items():
  for mode in ("logit","geom","linear"):
   for w in (0,.01,.02,.03,.05,.08,.1,.14,.2,.3,.4,.5):rows.append({"field":field,"mode":mode,"samurai_weight":w,**metrics(c,mix(base,score,w,mode),p,t,TVD)})
 best=max(rows,key=lambda x:x["map50"]);(OUT/"val_sweep.json").write_text(json.dumps({"best":best,"top":sorted(rows,key=lambda x:-x["map50"])[:50],"labels":labels},indent=2));report("fixed_test",2,validation_selection=best);qc,qp,qt,qloc,qlabels,qbase,qsam=routes("test",TEST);score=mix(qbase,qsam[best["field"]],best["samurai_weight"],best["mode"]);test={**metrics(qc,score,qp,qt,TVD),"labels":qlabels,"detections":len(qloc)};gain=100*(test["map50"]-VATD);summary={"protocol":"validation-selected SAMURAI memory complement on V144; fixed test","validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2));report("done",3,summary=summary);return 0
if __name__=="__main__":raise SystemExit(main())
