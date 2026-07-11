from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.run_tvd_oof_stack_v130 import VAL,TEST,flat_stats,load_final_scores,load_predictionsgt,metrics,blend
RUN=ROOT/"artifacts"/"detached_tvd_asymmetric_gate_v140";OUT=Path(r"D:\URAP_vatd_rank_results\tvd_asymmetric_gate_v140");TVD=Path(r"D:\urap_modal_stage\TransVisDrone");VATD=.93844

def report(stage,done,total=3,**extra):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**extra};(RUN/"progress.json").write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))
def logit(x):x=np.clip(x,1e-9,1-1e-9);return np.log(x/(1-x))
def v130_scores(split,data,locations):
 summary=json.loads(Path(r"D:\URAP_vatd_rank_results\tvd_oof_stack_v130\official_summary.json").read_text())["validation_selection"];weights=(float(summary["weights"]["v53"]),float(summary["weights"]["v126"]),float(summary["weights"]["v129"]));return blend(list(load_final_scores(split,data,locations)),weights,str(summary["mode"]))
def gated(raw,action,promote_low,demote_low,strength):
 out=raw.copy();promote=(action>raw)&(raw>=promote_low);demote=(action<=raw)&(raw>=demote_low);mask=promote|demote
 out[mask]=sigmoid((1-strength)*logit(raw[mask])+strength*logit(action[mask]));return out,int(mask.sum()),int(promote.sum()),int(demote.sum())
def main():
 OUT.mkdir(parents=True,exist_ok=True);report("select_validation",0);data=load_predictionsgt(VAL);correct,pred,target,loc,labels=flat_stats(data);raw=np.asarray([x[4] for x in loc],np.float64);action=v130_scores("val",data,loc);rows=[]
 for promote_low in (0,.002,.005,.0075,.01,.0125,.015,.02,.03,.05,.1):
  for demote_low in (0,.002,.005,.0075,.01,.015,.02,.03):
   for strength in (.5,.75,1.0):
    score,changed,promoted,demoted=gated(raw,action,promote_low,demote_low,strength);rows.append({"promote_low":promote_low,"demote_low":demote_low,"strength":strength,"changed_rows":changed,"promoted_rows":promoted,"demoted_rows":demoted,**metrics(correct,score,pred,target,TVD)})
 best=max(rows,key=lambda r:float(r["map50"]));(OUT/"val_sweep.json").write_text(json.dumps({"best":best,"top":sorted(rows,key=lambda r:-float(r["map50"]))[:50],"labels":labels},indent=2));report("fixed_test",2,validation_selection=best);q=load_predictionsgt(TEST);qc,qp,qt,qloc,qlabels=flat_stats(q);qraw=np.asarray([x[4] for x in qloc],np.float64);qaction=v130_scores("test",q,qloc);score,changed,promoted,demoted=gated(qraw,qaction,float(best["promote_low"]),float(best["demote_low"]),float(best["strength"]));test={**metrics(qc,score,qp,qt,TVD),"changed_rows":changed,"promoted_rows":promoted,"demoted_rows":demoted,"labels":qlabels,"detections":len(qloc)};gain=100*(test["map50"]-VATD);summary={"protocol":"validation-selected asymmetric promotion/demotion routing of V130; fixed test","validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2));report("done",3,summary=summary);return 0

if __name__=="__main__":raise SystemExit(main())




