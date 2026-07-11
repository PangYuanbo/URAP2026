from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.run_tvd_oof_stack_v130 import VAL,TEST,flat_stats,load_predictionsgt,metrics
from tools.run_tvd_track_memory_v144 import base_scores,load_track_indices,aggregate,fuse,TRACKS
RUN=ROOT/"artifacts"/"detached_tvd_frame_budget_refine_v147";OUT=Path(r"D:\URAP_vatd_rank_results\tvd_frame_budget_refine_v147");TVD=Path(r"D:\urap_modal_stage\TransVisDrone");VATD=.93844
def report(stage,done,total=3,**x):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**x};(RUN/"progress.json").write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
def load(split,pkl):
 data=load_predictionsgt(pkl);c,p,t,loc,labels=flat_stats(data);raw,base=base_scores(split,data,loc);lookup={(seq,fid,idx):i for i,(seq,fid,idx,iid,r) in enumerate(loc)};tracks,_,_=load_track_indices(TRACKS[split],lookup);memory,valid=aggregate(base,tracks,3.0,"median",2);score,_=fuse(base,memory,valid,.1,"promote");return c,p,t,loc,labels,score
def budget(score,loc,k,factor,gate):
 out=score.copy();frames={}
 for i,x in enumerate(loc):frames.setdefault(x[3],[]).append(i)
 changed=0
 for ids in frames.values():
  ids=np.asarray(ids);eligible=ids[score[ids]<gate];order=ids[np.argsort(score[ids])[::-1]];keep=set(order[:k].tolist());mask=np.asarray([i for i in eligible if i not in keep],np.int64)
  out[mask]*=factor;changed+=len(mask)
 return out,changed
def main():
 OUT.mkdir(parents=True,exist_ok=True);report("select_validation",0);c,p,t,loc,labels,base=load("val",VAL);rows=[]
 for k in (3,4,5,6):
  for factor in tuple(x/100 for x in range(40,81,2)):
   for gate in (.05,.06,.07,.08,.09,.1,.11,.12,.14,.16,.18,.2):
    score,changed=budget(base,loc,k,factor,gate);rows.append({"top_k":k,"suppression_factor":factor,"score_gate":gate,"changed_rows":changed,**metrics(c,score,p,t,TVD)})
 best=max(rows,key=lambda x:x["map50"]);(OUT/"val_sweep.json").write_text(json.dumps({"best":best,"top":sorted(rows,key=lambda x:-x["map50"])[:50],"labels":labels},indent=2));report("fixed_test",2,validation_selection=best);qc,qp,qt,qloc,qlabels,qbase=load("test",TEST);qscore,changed=budget(qbase,qloc,best["top_k"],best["suppression_factor"],best["score_gate"]);test={**metrics(qc,qscore,qp,qt,TVD),"changed_rows":changed,"labels":qlabels,"detections":len(qloc)};gain=100*(test["map50"]-VATD);summary={"protocol":"validation-selected refined per-frame candidate budget on V144; fixed test","validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2));report("done",3,summary=summary);return 0
if __name__=="__main__":raise SystemExit(main())

