from __future__ import annotations
import copy,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.run_tvd_oof_stack_v130 import TEST,load_predictionsgt,load_final_scores,blend
from tools.analyze_tvd_candidate_oracle import greedy_matches
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
from tools.run_tvd_asymmetric_gate_v140 import gated
OUT=Path(r"D:\URAP_vatd_rank_results\tvd_residual_diagnostics_v141")
data=load_predictionsgt(TEST);summary=json.loads(Path(r"D:\URAP_vatd_rank_results\tvd_oof_stack_v130\official_summary.json").read_text())["validation_selection"];components=list(load_final_scores("test",data,[(image_key(str(i),0)[0],image_key(str(i),0)[1],j,str(i),float(r.get("score",0))) for i,item in sorted(data.items()) for j,r in enumerate(item.get("detections") or [])]));action=blend(components,(summary["weights"]["v53"],summary["weights"]["v126"],summary["weights"]["v129"]),summary["mode"]);cursor=0;groups=defaultdict(dict)
for iid,item in sorted(data.items()):
 ds=item.get("detections") or [];n=len(ds);raw=np.asarray([float(r.get("score",0)) for r in ds]);score,_,_,_=gated(raw,action[cursor:cursor+n],.01,0,1.0);cursor+=n;seq=image_key(str(iid),0)[0];v=copy.deepcopy(item);v["detections"]=[{**r,"score":float(s)} for r,s in zip(ds,score)];oracle=copy.deepcopy(item);matches=greedy_matches(ds,item.get("labels") or [],.5);oracle["detections"]=[{**ds[d],"score":1-1e-7*k} for k,(_,d,_) in enumerate(matches)];groups[seq][str(iid)]={"raw":item,"best":v,"oracle":oracle}
OUT.mkdir(parents=True,exist_ok=True);rows=[]
for seq,items in sorted(groups.items()):
 raw={k:v["raw"] for k,v in items.items()};best={k:v["best"] for k,v in items.items()};oracle={k:v["oracle"] for k,v in items.items()};rm=evaluate_data(raw,Path(r"D:\urap_modal_stage\TransVisDrone"),OUT);bm=evaluate_data(best,Path(r"D:\urap_modal_stage\TransVisDrone"),OUT);om=evaluate_data(oracle,Path(r"D:\urap_modal_stage\TransVisDrone"),OUT);labels=sum(len(x.get("labels") or []) for x in raw.values());rows.append({"sequence":seq,"images":len(raw),"labels":labels,"raw_map50":rm["map50"],"best_map50":bm["map50"],"oracle_map50":om["map50"],"gain":bm["map50"]-rm["map50"],"remaining_to_oracle":om["map50"]-bm["map50"]});print(json.dumps(rows[-1]),flush=True)
result={"method":"V140 asymmetric V130","rows":rows,"priority":sorted(rows,key=lambda r:-(r["labels"]*r["remaining_to_oracle"]))};(OUT/"summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
