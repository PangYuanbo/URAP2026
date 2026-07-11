from __future__ import annotations
import gc,json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(r"C:\Users\aaron\Desktop\URAP");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import load_aux
from tools.train_action_chunk_neighbor_full import dataset_arrays,hard_rows,load_neighbor
from tools.run_tvd_domain_balanced_action_v119 import fit_balanced
RUN=ROOT/"artifacts"/"detached_tvd_hard_domain_action_v142";OUT=Path(r"D:\URAP_vatd_rank_results\tvd_hard_domain_action_v142");FULL=Path(r"D:\URAP_vatd_rank_results\action_chunk_full_dev_v36");N=Path(r"D:\URAP_vatd_rank_results\action_chunk_neighbor_v44");TRAIN=Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0.pkl");VAL=Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl");TEST=Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl");SIZE=ROOT/"data_templates"/"nps_sequence_sizes_actual.json";FIELD="tvd_hard_domain_action_score";VATD=.93844
def report(stage,done,total=4,**x):
 RUN.mkdir(parents=True,exist_ok=True);p={"stage":stage,"done":done,"total":total,"updated":datetime.now().astimezone().isoformat(),**x};(RUN/"progress.json").write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
def execute(cmd):
 c=subprocess.call(cmd,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT)}); 
 if c:raise RuntimeError(f"command failed {c}")
def main():
 OUT.mkdir(parents=True,exist_ok=True);sizes=json.loads(SIZE.read_text())
 report("load_features",0)
 tf,tb=load_aux(FULL/"train_forward.jsonl"),load_aux(FULL/"train_backward.jsonl");tn,names=load_neighbor(N/"train_neighbor_scores.jsonl");tx,ty,tg,_,tseq=dataset_arrays(load_predictionsgt(TRAIN),tf,tb,tn,True,sizes);del tf,tb,tn;gc.collect()
 vf,vb=load_aux(FULL/"val_forward.jsonl"),load_aux(FULL/"val_backward.jsonl");vn,vnames=load_neighbor(N/"val_neighbor_scores.jsonl");assert names==vnames;vx,_,_,vloc,_=dataset_arrays(load_predictionsgt(VAL),vf,vb,vn,False,sizes);del vf,vb,vn;gc.collect()
 qf,qb=load_aux(FULL/"test_forward.jsonl"),load_aux(FULL/"test_backward.jsonl");qn,qnames=load_neighbor(N/"test_neighbor_scores.jsonl");assert names==qnames;qx,_,_,qloc,_=dataset_arrays(load_predictionsgt(TEST),qf,qb,qn,False,sizes);del qf,qb,qn;gc.collect()
 hard=set("Clip_30 Clip_36 Clip_29 Clip_11 Clip_23 Clip_16 Clip_10 Clip_9 Clip_31 Clip_2 Clip_22 Clip_35 Clip_17".split());keep=hard_rows(tx,ty,tg);keep=keep[np.asarray([str(tseq[i]) in hard for i in keep])];binary=(ty[keep]>=.5).astype(np.int32);pos=max(1,int(binary.sum()));neg=len(binary)-pos;weights=np.ones(len(keep),np.float32);report("train_models",1,train_rows=len(tx),hard_rows=len(keep),positives=pos)
 vals=[];tests=[];records=[];(OUT/"models").mkdir(exist_ok=True)
 for i,seed in enumerate((2026,2048,4096,8192),1):
  model=fit_balanced(tx[keep],ty[keep],weights,seed);vals.append(model.predict_proba(vx)[:,1].astype(np.float32));tests.append(model.predict_proba(qx)[:,1].astype(np.float32));mp=OUT/"models"/f"seed_{seed}.ubj";model.save_model(mp);records.append(str(mp));report("train_models",1,model=i,models=4,seed=seed);del model;gc.collect()
 vs=np.mean(np.stack(vals),axis=0);qs=np.mean(np.stack(tests),axis=0);vp=OUT/"val_scores.jsonl";qp=OUT/"test_scores.jsonl";write_score_jsonl(vp,vs,vloc,FIELD);write_score_jsonl(qp,qs,qloc,FIELD)
 report("select_validation",2)
 sweep=OUT/"val_sweep.json";execute([sys.executable,str(ROOT/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",str(VAL),"--tracklet-jsonl",str(vp),"--per-row-score","--score-field",FIELD,"--modes","geom-mix","logit-mix","fp-suppress","replace","--alphas",".01,.02,.04,.06,.08,.1,.14,.2,.3,.4,.55,.7,1","--out-json",str(sweep)]);best=json.loads(sweep.read_text())["best"]
 report("fixed_test",3,validation_selection=best);fixed=OUT/"test_fixed.json";execute([sys.executable,str(ROOT/"tools"/"sweep_tvd_predictionsgt_score_fusion.py"),"--tvd-root",r"D:\urap_modal_stage\TransVisDrone","--predictionsgt-pkl",str(TEST),"--tracklet-jsonl",str(qp),"--per-row-score","--score-field",FIELD,"--modes",str(best["mode"]),"--alphas",str(best["alpha"]),"--out-json",str(fixed)]);test=json.loads(fixed.read_text())["best"];gain=100*(test["map50"]-VATD);summary={"protocol":"hard-domain train-only four-seed Action Bank; validation selection; fixed test","models":records,"train_hard_rows":len(keep),"validation_selection":best,"test_fixed":test,"vatd_map50":VATD,"gain_over_vatd_points":gain,"target_3_to_5_met":3<=gain<=5};(OUT/"official_summary.json").write_text(json.dumps(summary,indent=2));report("done",4,summary=summary);return 0
if __name__=="__main__":raise SystemExit(main())

