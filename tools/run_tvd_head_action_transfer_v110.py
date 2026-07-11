from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
R=Path(r'C:\Users\aaron\Desktop\URAP');sys.path[:0]=[str(R),str(R/'tools')]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores,fuse_score
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map
OLD=Path(r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl');NEW=Path(r'D:\URAP_vatd_rank_results\tvd_head_calibrate_v108_eval\test_last\predictionsgt\predictionsgt_split_0.pkl');V46=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46\test_scores.jsonl');V52=Path(r'D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52\test_expert_scores.jsonl');FPS=R/'data_templates'/'nps_sequence_fps.json';O=Path(r'D:\URAP_vatd_rank_results\tvd_head_action_transfer_v110')
def iou_matrix(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)),np.float32)
 ix1=np.maximum(a[:,None,0],b[None,:,0]);iy1=np.maximum(a[:,None,1],b[None,:,1]);ix2=np.minimum(a[:,None,2],b[None,:,2]);iy2=np.minimum(a[:,None,3],b[None,:,3]);inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1);aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-6)
def main():
 old=load_predictionsgt(OLD);new=load_predictionsgt(NEW);base,_=load_row_scores(V46,'action_chunk_neighbor_score',1);expert,_=load_row_scores(V52,'action_chunk_multi_expert_score',1);gates=temporal_gate_map(old,.3,3.,.75,json.loads(FPS.read_text()));old_by_key={image_key(str(iid),0)[:2]:item for iid,item in old.items()};output={};matched=total=0
 for image_id,item in new.items():
  seq,fid,_=image_key(str(image_id),0);old_item=old_by_key.get((seq,fid),{'detections':[]});old_rows=old_item.get('detections') or [];old_scores=[]
  for index,row in enumerate(old_rows):
   raw=float(row.get('score',0.));key=(seq,fid,index);b=max(1e-9,float(base.get(key,raw)));e=max(1e-9,float(expert.get(key,b)));aux=math.exp(.5*math.log(b)+.5*math.log(e)) if gates.get(str(image_id),False) else b;old_scores.append(fuse_score(raw,aux,.4,'geom-mix'))
  old_boxes=np.asarray([row['bbox'] for row in old_rows],np.float32).reshape(-1,4) if old_rows else np.zeros((0,4),np.float32);new_rows=item.get('detections') or [];new_boxes=np.asarray([row['bbox'] for row in new_rows],np.float32).reshape(-1,4) if new_rows else np.zeros((0,4),np.float32);ious=iou_matrix(new_boxes,old_boxes);rows=[]
  for index,row in enumerate(new_rows):
   z=dict(row);raw=float(row.get('score',0.));total+=1
   if len(old_rows):
    best=int(np.argmax(ious[index]));quality=float(ious[index,best])
    if quality>=.35:
     prior=float(old_scores[best]);weight=.35*min(1.,max(0.,(quality-.35)/.45));z['score']=fuse_score(raw,prior,weight,'logit-mix');matched+=1
   rows.append(z)
  output[image_id]={'labels':item.get('labels',[]),'detections':rows}
 O.mkdir(parents=True,exist_ok=True);metrics=evaluate_data(output,Path(r'D:\urap_modal_stage\TransVisDrone'),O);summary={'protocol':'head-calibrated detector plus V53 validation-selected Action Bank transferred by camera-space IoU; fixed transfer rule; no test-label fitting','matched_rows':matched,'total_rows':total,'match_fraction':matched/max(1,total),'test':metrics,'gain_over_vatd_points':100*(metrics['map50']-.93844),'target_3_to_5_met':.03<=metrics['map50']-.93844<=.05};(O/'official_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
