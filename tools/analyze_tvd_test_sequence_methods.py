from __future__ import annotations
import json,math,sys
from collections import defaultdict
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,fuse_score
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map
P=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');V46=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\test_scores.jsonl');V52=Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\test_expert_scores.jsonl');V94=Path(r'D:\URAP_vatd_rank_results\tvd_resolution_specialist_v94\test_scores.jsonl');FPS=R/'data_templates'/'nps_sequence_fps.json';O=Path(r'D:\URAP_vatd_rank_results\tvd_sequence_diagnostics_v96')
def main():
 data=load_predictionsgt(P);base,_=load_row_scores(V46,'action_chunk_neighbor_score',1);expert,_=load_row_scores(V52,'action_chunk_multi_expert_score',1);special,_=load_row_scores(V94,'action_chunk_1920_score',1);gates=temporal_gate_map(data,.3,3.,.75,json.loads(FPS.read_text()));groups=defaultdict(dict)
 for image_id,item in data.items():groups[image_key(str(image_id),0)[0]][image_id]=item
 rows=[];O.mkdir(parents=True,exist_ok=True)
 for seq,subset in sorted(groups.items()):
  v53={};v94={}
  for image_id,item in subset.items():
   _,fid,_=image_key(str(image_id),0);a=[];b=[]
   for index,row in enumerate(item.get('detections') or []):
    key=(seq,fid,index);raw=float(row.get('score',0.));z=dict(row);q=dict(row);bs=max(1e-9,float(base.get(key,raw)));es=max(1e-9,float(expert.get(key,bs)));aux=math.exp(.5*math.log(bs)+.5*math.log(es)) if gates.get(str(image_id),False) else bs;z['score']=fuse_score(raw,aux,.4,'geom-mix');q['score']=fuse_score(raw,float(special.get(key,raw)),.4,'logit-mix');a.append(z);b.append(q)
   v53[image_id]={'labels':item.get('labels',[]),'detections':a};v94[image_id]={'labels':item.get('labels',[]),'detections':b}
  raw_m=evaluate_data(subset,Path(r'D:\urap_modal_stage\TransVisDrone'),O);v53_m=evaluate_data(v53,Path(r'D:\urap_modal_stage\TransVisDrone'),O);v94_m=evaluate_data(v94,Path(r'D:\urap_modal_stage\TransVisDrone'),O);row={'sequence':seq,'raw_map50':raw_m['map50'],'v53_map50':v53_m['map50'],'v94_map50':v94_m['map50'],'v53_delta':v53_m['map50']-raw_m['map50'],'v94_delta':v94_m['map50']-raw_m['map50']};rows.append(row);print(json.dumps(row),flush=True)
 (O/'summary.json').write_text(json.dumps({'rows':rows},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
