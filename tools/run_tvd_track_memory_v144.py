from __future__ import annotations
import json,math,sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.run_tvd_oof_stack_v130 import VAL,TEST,flat_stats,load_final_scores,load_predictionsgt,metrics,blend
from tools.run_tvd_asymmetric_gate_v140 import gated,logit,sigmoid
RUN=ROOT/"artifacts"/"detached_tvd_track_memory_v144";OUT=Path(r"D:\URAP_vatd_rank_results\tvd_track_memory_v144");TVD=Path(r"D:\urap_modal_stage\TransVisDrone");FPS=json.loads((ROOT/"data_templates"/"nps_sequence_fps.json").read_text());VATD=.93844
TRACKS={"val":Path(r"D:\URAP_nps_val_tvd\route_b_official\tracklets\proposal_tracklets.jsonl"),"test":Path(r"D:\URAP_vatd_rank_inputs\nps_tracklets_with_vatd.jsonl")}
def report(stage,done,total=4,**x):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**x};(RUN/"progress.json").write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
def base_scores(split,data,loc):
 s=json.loads(Path(r"D:\URAP_vatd_rank_results\tvd_oof_stack_v130\official_summary.json").read_text())["validation_selection"];action=blend(list(load_final_scores(split,data,loc)),(s["weights"]["v53"],s["weights"]["v126"],s["weights"]["v129"]),s["mode"]);raw=np.asarray([x[4] for x in loc]);return raw,gated(raw,action,.01,0,1.0)[0]
def load_track_indices(path,lookup):
 tracks=[];mapped=0;rows=0
 with path.open(encoding="utf-8-sig") as source:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);members=[]
   for r in item.get("rows") or []:
    rows+=1;key=(str(r.get("seq") or ""),int(r.get("frame_id",0)),int(r.get("prediction_index",-1)));idx=lookup.get(key)
    if idx is not None:members.append((int(r.get("frame_id",0)),idx));mapped+=1
   if len(members)>=2:tracks.append(sorted(members))
 return tracks,mapped,rows
def aggregate(base,tracks,window_s,kind,min_rows):
 total=np.zeros(len(base));count=np.zeros(len(base),np.int32)
 for track in tracks:
  seq_fps=float(FPS.get("Clip_"+str(0),30));frames=np.asarray([x[0] for x in track]);ids=np.asarray([x[1] for x in track]);seq_window=None
  for pos,(fid,idx) in enumerate(track):
   if seq_window is None:
    seq_window=max(1,int(round(window_s*30.0)))
   mask=np.abs(frames-fid)<=seq_window;values=base[ids[mask]]
   if len(values)<min_rows:continue
   if kind=="mean":v=float(values.mean())
   elif kind=="median":v=float(np.median(values))
   elif kind=="q75":v=float(np.quantile(values,.75))
   else:v=float(values.max())
   total[idx]+=v;count[idx]+=1
 out=base.copy();valid=count>0;out[valid]=total[valid]/count[valid];return out,valid
def fuse(base,memory,valid,strength,mode):
 out=base.copy();mask=valid.copy()
 if mode=="promote":mask&=memory>base
 elif mode=="demote":mask&=memory<base
 out[mask]=sigmoid((1-strength)*logit(base[mask])+strength*logit(memory[mask]));return out,int(mask.sum())
def load(split,pkl):
 data=load_predictionsgt(pkl);correct,pred,target,loc,labels=flat_stats(data);raw,base=base_scores(split,data,loc);lookup={(seq,fid,idx):i for i,(seq,fid,idx,iid,r) in enumerate(loc)};tracks,mapped,source_rows=load_track_indices(TRACKS[split],lookup);return correct,pred,target,loc,labels,raw,base,tracks,mapped,source_rows
def main():
 OUT.mkdir(parents=True,exist_ok=True);report("load_validation_tracks",0);c,p,t,loc,labels,raw,base,tracks,mapped,source_rows=load("val",VAL);report("select_validation",1,tracks=len(tracks),mapped=mapped,source_rows=source_rows);rows=[]
 for window in (1.0,3.0):
  for kind in ("mean","median","q75","max"):
   for minimum in (2,3,5):
    memory,valid=aggregate(base,tracks,window,kind,minimum)
    for mode in ("promote","demote","symmetric"):
     for strength in (.1,.2,.3,.5,.75,1.0):
      score,changed=fuse(base,memory,valid,strength,mode);rows.append({"window_seconds":window,"aggregation":kind,"min_rows":minimum,"mode":mode,"strength":strength,"memory_rows":int(valid.sum()),"changed_rows":changed,**metrics(c,score,p,t,TVD)})
 best=max(rows,key=lambda x:x["map50"]);(OUT/"val_sweep.json").write_text(json.dumps({"best":best,"top":sorted(rows,key=lambda x:-x["map50"])[:50],"labels":labels,"tracks":len(tracks),"mapped":mapped},indent=2));report("load_test_tracks",2,validation_selection=best);qc,qp,qt,qloc,qlabels,qraw,qbase,qtracks,qmapped,qsource=load("test",TEST);memory,valid=aggregate(qbase,qtracks,best["window_seconds"],best["aggregation"],best["min_rows"]);score,changed=fuse(qbase,memory,valid,best["strength"],best["mode"]);test={**metrics(qc,score,qp,qt,TVD),"tracks":len(qtracks),"mapped":qmapped,"source_rows":qsource,"memory_rows":int(valid.sum()),"changed_rows":changed,"labels":qlabels,"detections":len(qloc)};gain=100*(test["map50"]-VATD);summary={"protocol":"validation-selected 1s/3s proposal-track memory on V140; fixed test","validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2));report("done",4,summary=summary);return 0
if __name__=="__main__":raise SystemExit(main())
