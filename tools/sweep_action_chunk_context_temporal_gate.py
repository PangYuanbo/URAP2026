from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
 if str(candidate) not in sys.path:sys.path.insert(0,str(candidate))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_action_chunk_multiplicity_expert import cluster_count
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key,parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores

def logit(value):
 value=min(1-1e-6,max(1e-6,float(value)));return math.log(value/(1-value))
def sigmoid(value):return 1/(1+math.exp(-max(-30,min(30,value))))
def gates(data,threshold,seconds,fraction,fps_map):
 grouped=defaultdict(list)
 for image_id,item in data.items():
  seq,fid,_=image_key(str(image_id),0);grouped[seq].append((fid,str(image_id),cluster_count(item.get('detections') or [],threshold)>=2))
 output={}
 for seq,items in grouped.items():
  items.sort();window=max(1,int(round(seconds*float(fps_map.get(seq,30)))));history=deque();active=0
  for fid,image_id,signal in items:
   history.append((fid,signal));active+=int(signal)
   while history and fid-history[0][0]>=window:active-=int(history.popleft()[1])
   output[image_id]=active/max(1,len(history))>=fraction
 return output
def evaluate(data,v46,v51,v52,gate,expert_weight,tvd_root,out_dir):
 output={}
 for image_id,item in data.items():
  rows=[];enabled=gate.get(str(image_id),False)
  for index,row in enumerate(item.get('detections') or []):
   key=image_key(str(image_id),index);raw=float(row.get('score',0));a=float(v46.get(key,raw));b=float(v51.get(key,raw));expert=float(v52.get(key,raw));base=sigmoid(.8*logit(raw)+.1*logit(a)+.1*logit(b));score=sigmoid((1-expert_weight)*logit(base)+expert_weight*logit(expert)) if enabled else base;new=dict(row);new['score']=score;rows.append(new)
  output[image_id]={'labels':item.get('labels',[]),'detections':rows}
 return evaluate_data(output,tvd_root,out_dir)
def main():
 p=argparse.ArgumentParser();p.add_argument('--tvd-root',type=Path,required=True);p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--v46',type=Path,required=True);p.add_argument('--v51',type=Path,required=True);p.add_argument('--v52',type=Path,required=True);p.add_argument('--fps-json',type=Path,required=True);p.add_argument('--thresholds',default='.3,.4');p.add_argument('--windows',default='1,3');p.add_argument('--fractions',default='.5,.75');p.add_argument('--weights',default='.02,.04,.08,.12,.2');p.add_argument('--fixed-config-json',type=Path);p.add_argument('--out-json',type=Path,required=True);a=p.parse_args();data=load_predictionsgt(a.predictionsgt_pkl);v46,_=load_row_scores(a.v46,'action_chunk_neighbor_score',1);v51,_=load_row_scores(a.v51,'action_chunk_candidate_context_score',1);v52,_=load_row_scores(a.v52,'action_chunk_multi_expert_score',1);fps=json.loads(a.fps_json.read_text(encoding='utf8'))
 if a.fixed_config_json:
  best=json.loads(a.fixed_config_json.read_text(encoding='utf8'))['best'];configs=[(best['threshold'],best['window_seconds'],best['min_fraction'],best['expert_weight'])]
 else:configs=[(t,w,f,e) for t in parse_csv_floats(a.thresholds) for w in parse_csv_floats(a.windows) for f in parse_csv_floats(a.fractions) for e in parse_csv_floats(a.weights)]
 cache={};rows=[]
 for threshold,window,fraction,weight in configs:
  key=(threshold,window,fraction);gate=cache.setdefault(key,gates(data,threshold,window,fraction,fps));metrics=evaluate(data,v46,v51,v52,gate,weight,a.tvd_root,a.out_json.parent);record={'threshold':threshold,'window_seconds':window,'min_fraction':fraction,'expert_weight':weight,'gated_images':sum(gate.values()),**metrics};rows.append(record);print(json.dumps(record),flush=True)
 best=max(rows,key=lambda row:row['map50']);a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps({'best':best,'rows':rows},indent=2),encoding='utf8');return 0
if __name__=='__main__':raise SystemExit(main())
