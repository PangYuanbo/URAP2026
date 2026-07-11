from __future__ import annotations
import argparse, json, math, pickle, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO/"tools") not in sys.path: sys.path.insert(0,str(REPO/"tools"))
from eval_tvd_predictionsgt_pkl import load_predictionsgt
from sweep_tvd_predictionsgt_action_rescore import evaluate_data, image_key
from sweep_tvd_predictionsgt_score_fusion import clone_with_fused_scores
FIELDS=("bank","quality","stability","short_iou","short_compat","long_iou","long_compat","motion_combo")
TOKEN_DIM=12
def finite(v):
 try: x=float(v)
 except (TypeError,ValueError): return 0.0
 return x if math.isfinite(x) else 0.0
def token_stats(values):
 a=np.asarray(values or [],dtype=np.float32)
 if not len(a) or len(a)%TOKEN_DIM: return (0.0,0.0,0.0)
 a=a.reshape(-1,TOKEN_DIM); valid=a[:,0]>.5
 if not valid.any(): return (0.0,0.0,0.0)
 ages=np.arange(len(a),dtype=np.float32); w=np.exp(-2*ages/max(1,len(a)-1))[valid]; w/=w.sum()
 return float((a[valid,9]*w).sum()),float((a[valid,11]*w).sum()),float(valid.mean())
def load_maps(path):
 maps={f:{} for f in FIELDS}
 with Path(path).open(encoding="utf-8-sig") as source:
  for line in source:
   if not line.strip(): continue
   item=json.loads(line); meta=item.get("meta") or {}
   for r in item.get("rows") or []:
    seq=str(r.get("seq") or meta.get("seq") or ""); fid=r.get("frame_id"); idx=r.get("prediction_index")
    if not seq or fid is None or idx is None: continue
    key=(seq,int(float(fid)),int(float(idx)))
    si,sc,sv=token_stats(r.get("online_action_bank_short_tokens")); li,lc,lv=token_stats(r.get("online_action_bank_long_tokens"))
    bank=finite(r.get("online_action_bank_score")); quality=finite(r.get("online_action_bank_track_quality")); stability=finite(r.get("online_action_bank_motion_stability"))
    vals={"bank":bank,"quality":quality,"stability":stability,"short_iou":si,"short_compat":sc,"long_iou":li,"long_compat":lc,"motion_combo":max(0,min(1,.25*bank+.15*quality+.10*stability+.20*si+.15*sc+.10*li+.05*lc))}
    for f,v in vals.items(): maps[f][key]=v
 return maps
def main():
 p=argparse.ArgumentParser(); p.add_argument("--val-pkl",type=Path,required=True);p.add_argument("--test-pkl",type=Path,required=True);p.add_argument("--val-jsonl",type=Path,required=True);p.add_argument("--test-jsonl",type=Path,required=True);p.add_argument("--tvd-root",type=Path,required=True);p.add_argument("--out-json",type=Path,required=True);a=p.parse_args()
 val=load_predictionsgt(a.val_pkl); vm=load_maps(a.val_jsonl); modes=("linear-mix","logit-mix","geom-mix","fp-suppress","tp-boost"); alphas=(.01,.02,.04,.06,.08,.10,.14,.20,.30,.40,.55,.70); rows=[]
 for field in FIELDS:
  for mode in modes:
   for alpha in alphas:
    metrics=evaluate_data(clone_with_fused_scores(val,vm[field],mode,alpha,"keep"),a.tvd_root,a.out_json.parent); rows.append({"field":field,"mode":mode,"alpha":alpha,**metrics})
 best=max(rows,key=lambda r:r["map50"]); test=load_predictionsgt(a.test_pkl); tm=load_maps(a.test_jsonl); fixed=evaluate_data(clone_with_fused_scores(test,tm[best["field"]],best["mode"],best["alpha"],"keep"),a.tvd_root,a.out_json.parent)
 result={"validation_best":best,"test_fixed":fixed,"target_map50":.97,"target_met":fixed["map50"]>=.97,"top":sorted(rows,key=lambda r:r["map50"],reverse=True)[:20]};a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2));return 0
if __name__=="__main__": raise SystemExit(main())
