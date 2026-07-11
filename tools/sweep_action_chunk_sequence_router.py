from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
 if str(candidate) not in sys.path:sys.path.insert(0,str(candidate))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_action_chunk_context_temporal_gate import gates,logit,sigmoid
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key,parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score,load_row_scores

def build_scores(data,v46,v51,v52,frame_gate,sequence_threshold):
 counts=defaultdict(lambda:[0,0])
 for image_id in data:
  sequence,_,_=image_key(str(image_id),0);counts[sequence][0]+=int(frame_gate.get(str(image_id),False));counts[sequence][1]+=1
 active={sequence:(on/max(1,total)>=sequence_threshold) for sequence,(on,total) in counts.items()}
 output={}
 for image_id,item in data.items():
  sequence,_,_=image_key(str(image_id),0);use_temporal=active[sequence];enabled=frame_gate.get(str(image_id),False);rows=[]
  for index,row in enumerate(item.get('detections') or []):
   key=image_key(str(image_id),index);raw=float(row.get('score',0));neighbor=float(v46.get(key,raw));context=float(v51.get(key,raw));expert=float(v52.get(key,raw))
   if use_temporal:
    auxiliary=math.sqrt(max(1e-9,neighbor)*max(1e-9,expert)) if enabled else neighbor
    score=fuse_score(raw,auxiliary,.4,'geom-mix')
   else:score=sigmoid(.8*logit(raw)+.1*logit(neighbor)+.1*logit(context))
   updated=dict(row);updated['score']=score;rows.append(updated)
  output[image_id]={'labels':item.get('labels',[]),'detections':rows}
 return output,active,{sequence:on/max(1,total) for sequence,(on,total) in counts.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--tvd-root',type=Path,required=True);p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--v46',type=Path,required=True);p.add_argument('--v51',type=Path,required=True);p.add_argument('--v52',type=Path,required=True);p.add_argument('--fps-json',type=Path,required=True);p.add_argument('--sequence-thresholds',default='.001,.01,.025,.05,.1,.2,.3');p.add_argument('--fixed-config-json',type=Path);p.add_argument('--out-json',type=Path,required=True);a=p.parse_args()
 data=load_predictionsgt(a.predictionsgt_pkl);v46,_=load_row_scores(a.v46,'action_chunk_neighbor_score',1);v51,_=load_row_scores(a.v51,'action_chunk_candidate_context_score',1);v52,_=load_row_scores(a.v52,'action_chunk_multi_expert_score',1);fps=json.loads(a.fps_json.read_text(encoding='utf8'));frame_gate=gates(data,.3,3.,.75,fps)
 if a.fixed_config_json:thresholds=[json.loads(a.fixed_config_json.read_text(encoding='utf8'))['best']['sequence_active_threshold']]
 else:thresholds=parse_csv_floats(a.sequence_thresholds)
 rows=[]
 for threshold in thresholds:
  output,active,fractions=build_scores(data,v46,v51,v52,frame_gate,threshold);metrics=evaluate_data(output,a.tvd_root,a.out_json.parent);record={'sequence_active_threshold':threshold,'temporal_sequences':sum(active.values()),'sequences':len(active),'sequence_gate_fractions':fractions,**metrics};rows.append(record);print(json.dumps(record),flush=True)
 best=max(rows,key=lambda row:row['map50']);a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps({'best':best,'rows':rows},indent=2),encoding='utf8');return 0
if __name__=='__main__':raise SystemExit(main())
