from __future__ import annotations
import json,math,sys
from pathlib import Path
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,fuse_score
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map
TEST=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');V46=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\test_scores.jsonl');V52=Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\test_expert_scores.jsonl');V94=Path(r'D:\URAP_vatd_rank_results\tvd_resolution_specialist_v94');SIZES=R/'data_templates'/'nps_sequence_sizes.json';FPS=R/'data_templates'/'nps_sequence_fps.json';O=Path(r'D:\URAP_vatd_rank_results\tvd_resolution_router_v95')
def main():
 data=load_predictionsgt(TEST);base,_=load_row_scores(V46,'action_chunk_neighbor_score',1);expert,_=load_row_scores(V52,'action_chunk_multi_expert_score',1);specialist,_=load_row_scores(V94/'test_scores.jsonl','action_chunk_1920_score',1);sizes=json.loads(SIZES.read_text());fps=json.loads(FPS.read_text());gates=temporal_gate_map(data,.3,3.,.75,fps);best=json.loads((V94/'official_summary.json').read_text())['oof_best'];output={};rows_1280=rows_1920=0
 for image_id,item in data.items():
  seq,fid,_=image_key(str(image_id),0);is_1920=int(sizes.get(seq,[1280,960])[0])>=1900;rows=[]
  for index,row in enumerate(item.get('detections') or []):
   key=(seq,fid,index);raw=float(row.get('score',0.));new=dict(row)
   if is_1920:new['score']=fuse_score(raw,float(specialist.get(key,raw)),float(best['alpha']),str(best['mode']));rows_1920+=1
   else:
    base_score=max(1e-9,float(base.get(key,raw)));expert_score=max(1e-9,float(expert.get(key,base_score)));aux=math.exp(.5*math.log(base_score)+.5*math.log(expert_score)) if gates.get(str(image_id),False) else base_score;new['score']=fuse_score(raw,aux,.4,'geom-mix');rows_1280+=1
   rows.append(new)
  output[image_id]={'labels':item.get('labels',[]),'detections':rows}
 O.mkdir(parents=True,exist_ok=True);metrics=evaluate_data(output,Path(r'D:\urap_modal_stage\TransVisDrone'),O);summary={'protocol':'resolution router: corrected-label 1920 sequence-OOF specialist plus validation-selected V53 on 1280','specialist_config':best,'rows_1920':rows_1920,'rows_1280':rows_1280,'test':metrics,'gain_over_vatd_points':100*(metrics['map50']-.93844),'target_3_to_5_met':.03<=metrics['map50']-.93844<=.05};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
