from __future__ import annotations
import json,sys
from collections import defaultdict
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
P=Path(r'D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl');O=Path(r'D:\URAP_vatd_rank_results\tvd_hard_sequence_v90\train_sequence_metrics.json')
def main():
 data=load_predictionsgt(P);g=defaultdict(dict)
 for iid,item in data.items():g[image_key(str(iid),0)[0]][iid]=item
 rows=[]
 for seq,b in sorted(g.items()):
  m=evaluate_data(b,Path(r'D:\urap_modal_stage\TransVisDrone'),O.parent);images=len(b);labels=sum(len(x.get('labels',[])) for x in b.values());dets=sum(len(x.get('detections',[])) for x in b.values());raw=[float(r.get('score',0)) for x in b.values() for r in x.get('detections',[])];row={'sequence':seq,'images':images,'labels':labels,'detections':dets,'detections_per_image':dets/images,'positive_fraction':labels/images,'mean_score':sum(raw)/max(1,len(raw)),**m};rows.append(row);print(json.dumps(row),flush=True)
 O.parent.mkdir(parents=True,exist_ok=True);O.write_text(json.dumps({'rows':rows},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
