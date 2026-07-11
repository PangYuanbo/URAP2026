from __future__ import annotations
import argparse,json,math,pickle,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO/"tools") not in sys.path:sys.path.insert(0,str(REPO/"tools"))
from eval_tvd_predictionsgt_pkl import load_predictionsgt
from sweep_tvd_predictionsgt_action_rescore import image_key
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);z=max(0,x2-x1)*max(0,y2-y1);u=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-z;return z/max(u,1e-9)
def main():
 p=argparse.ArgumentParser();p.add_argument("--pkl",type=Path,required=True);p.add_argument("--out-jsonl",type=Path,required=True);a=p.parse_args();data=load_predictionsgt(a.pkl);a.out_jsonl.parent.mkdir(parents=True,exist_ok=True);rows_total=0
 with a.out_jsonl.open("w",encoding="utf-8") as out:
  for image_id,item in data.items():
   ds=item.get("detections") or [];scores=np.asarray([float(d.get("score",0)) for d in ds],dtype=np.float32);rows=[];seq,fid,_=image_key(str(image_id),0)
   for i,d in enumerate(ds):
    overlaps=np.asarray([iou(d["bbox"],x["bbox"]) for x in ds],dtype=np.float32) if ds else np.zeros(0);neighbors=(overlaps>=.3);strong=(overlaps>=.5);weighted=float(np.sum(scores*overlaps)-scores[i]);support=max(0,int(neighbors.sum())-1);strong_support=max(0,int(strong.sum())-1);neighbor_max=float(np.max(scores[np.arange(len(ds))!=i])) if len(ds)>1 else 0.0;local_max=float(np.max(scores[neighbors])) if neighbors.any() else float(scores[i]);prob=1.0
    for j in np.where(neighbors)[0]:prob*=1.0-float(scores[j])*float(overlaps[j])
    noisy_or=1.0-prob;density=1.0-math.exp(-max(0.0,weighted));consensus=max(0.0,min(1.0,.40*noisy_or+.30*density+.20*local_max+.10*min(1.0,strong_support/3)))
    rows.append({"seq":seq,"frame_id":fid,"prediction_index":i,"consensus_noisy_or":noisy_or,"consensus_density":density,"consensus_local_max":local_max,"consensus_support":min(1.0,support/5),"consensus_strong_support":min(1.0,strong_support/3),"consensus_score":consensus})
   out.write(json.dumps({"meta":{"seq":seq,"image_id":str(image_id)},"rows":rows},separators=(",",":"))+"\n");rows_total+=len(rows)
 print(json.dumps({"kind":"candidate_consensus_done","images":len(data),"rows":rows_total,"output":str(a.out_jsonl)}));return 0
if __name__=="__main__":raise SystemExit(main())
