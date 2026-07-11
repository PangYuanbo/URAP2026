from __future__ import annotations
import json,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.run_tvd_oof_stack_v130 import VAL,TEST,flat_stats,load_final_scores,load_predictionsgt,metrics,blend
from tools.run_tvd_asymmetric_gate_v140 import gated
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,fuse_score
RUN=ROOT/"artifacts"/"detached_tvd_sequence_router_v143";OUT=Path(r"D:\URAP_vatd_rank_results\tvd_sequence_router_v143");TVD=Path(r"D:\urap_modal_stage\TransVisDrone");VATD=.93844;V142=Path(r"D:\URAP_vatd_rank_results\tvd_hard_domain_action_v142")
def report(stage,done,total=3,**x):RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**x};(RUN/"progress.json").write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
def scores(split,data,loc):
 s=json.loads(Path(r"D:\URAP_vatd_rank_results\tvd_oof_stack_v130\official_summary.json").read_text())["validation_selection"];action=blend(list(load_final_scores(split,data,loc)),(s["weights"]["v53"],s["weights"]["v126"],s["weights"]["v129"]),s["mode"]);raw=np.asarray([x[4] for x in loc]);base=gated(raw,action,.01,0,1.0)[0];cfg=json.loads((V142/"official_summary.json").read_text())["validation_selection"];m,_=load_row_scores(V142/("val_scores.jsonl" if split=="val" else "test_scores.jsonl"),"tvd_hard_domain_action_score",1);spec=np.asarray([fuse_score(r,float(m.get((seq,fid,idx),r)),cfg["alpha"],cfg["mode"]) for seq,fid,idx,iid,r in loc]);return raw,base,spec
def features(loc,raw):
 frames=defaultdict(list)
 for i,(seq,fid,idx,iid,r) in enumerate(loc):frames[(seq,iid)].append(i)
 seqs=defaultdict(list)
 for (seq,iid),ids in frames.items():seqs[seq].append(ids)
 out={}
 for seq,fs in seqs.items():
  vals=np.concatenate([raw[x] for x in fs]);mx=np.asarray([raw[x].max() if len(x) else 0 for x in fs]);out[seq]={"detections_per_image":len(vals)/len(fs),"mean_score":float(vals.mean()),"active_fraction":float((mx>=.1).mean()),"mean_frame_max":float(mx.mean()),"low_fraction":float((vals<.01).mean())}
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);report("select_validation",0);d=load_predictionsgt(VAL);c,p,t,loc,labels=flat_stats(d);raw,base,spec=scores("val",d,loc);feat=features(loc,raw);rows=[]
 for name in next(iter(feat.values())):
  vals=sorted(set(v[name] for v in feat.values()));thresholds=[vals[0]-1e-9,vals[-1]+1e-9]+[(a+b)/2 for a,b in zip(vals,vals[1:])]
  for op in ("le","ge"):
   for th in thresholds:
    routed={seq for seq,v in feat.items() if (v[name]<=th if op=="le" else v[name]>=th)};score=np.asarray([spec[i] if loc[i][0] in routed else base[i] for i in range(len(loc))]);rows.append({"feature":name,"op":op,"threshold":th,"routed_sequences":sorted(routed),**metrics(c,score,p,t,TVD)})
 best=max(rows,key=lambda x:x["map50"]);(OUT/"val_sweep.json").write_text(json.dumps({"best":best,"top":sorted(rows,key=lambda x:-x["map50"])[:30],"sequence_features":feat},indent=2));report("fixed_test",2,validation_selection=best);q=load_predictionsgt(TEST);qc,qp,qt,qloc,qlabels=flat_stats(q);qraw,qbase,qspec=scores("test",q,qloc);qfeat=features(qloc,qraw);routed={seq for seq,v in qfeat.items() if (v[best["feature"]]<=best["threshold"] if best["op"]=="le" else v[best["feature"]]>=best["threshold"])};qscore=np.asarray([qspec[i] if qloc[i][0] in routed else qbase[i] for i in range(len(qloc))]);test={**metrics(qc,qscore,qp,qt,TVD),"routed_sequences":sorted(routed),"sequence_features":qfeat,"labels":qlabels,"detections":len(qloc)};gain=100*(test["map50"]-VATD);summary={"protocol":"validation-selected label-free sequence router between V140 and hard-domain V142; fixed test","validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2));report("done",3,summary=summary);return 0
if __name__=="__main__":raise SystemExit(main())
