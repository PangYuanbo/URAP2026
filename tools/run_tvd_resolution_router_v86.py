from __future__ import annotations
import json,math,sys
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,fuse_score
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map
TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');V46=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\test_scores.jsonl');V52=Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\test_expert_scores.jsonl');V85=Path(r'D:\URAP_vatd_rank_results\tvd_resolution_neighbor_v85\test_scores.jsonl');SIZES=R/'data_templates'/'nps_sequence_sizes.json';FPS=R/'data_templates'/'nps_sequence_fps.json';O=Path(r'D:\URAP_vatd_rank_results\tvd_resolution_router_v86')
def main():
 data=load_predictionsgt(TEST);old,_=load_row_scores(V46,'action_chunk_neighbor_score',1);expert,_=load_row_scores(V52,'action_chunk_multi_expert_score',1);new,_=load_row_scores(V85,'action_chunk_neighbor_score',1);sizes=json.loads(SIZES.read_text());fps=json.loads(FPS.read_text());gates=temporal_gate_map(data,.3,3.,.75,fps);out={};r1280=r1920=0
 for iid,item in data.items():
  seq,fid,_=image_key(str(iid),0);is1920=int(sizes.get(seq,[1280,960])[0])>=1900;rows=[]
  for i,row in enumerate(item.get('detections') or []):
   key=(seq,fid,i);raw=float(row.get('score',0));z=dict(row)
   if is1920:
    b=max(1e-9,float(old.get(key,raw)));e=max(1e-9,float(expert.get(key,b)));aux=math.exp(.5*math.log(b)+.5*math.log(e)) if gates.get(str(iid),False) else b;z['score']=fuse_score(raw,aux,.4,'geom-mix');r1920+=1
   else:
    aux=float(new.get(key,raw));z['score']=fuse_score(raw,aux,.2,'logit-mix');r1280+=1
   rows.append(z)
  out[iid]={'labels':item.get('labels',[]),'detections':rows}
 metrics=evaluate_data(out,Path(r'D:\urap_modal_stage\TransVisDrone'),O);summary={'protocol':'label-free resolution router: 1280 uses resolution-aware V85 selected on val; 1920 uses stable V53 selected on val','rows_1280':r1280,'rows_1920':r1920,'test':metrics,'gain_over_vatd_points':100*(metrics['map50']-.93844),'target_3_to_5_met':.03<=metrics['map50']-.93844<=.05};O.mkdir(parents=True,exist_ok=True);(O/'official_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
