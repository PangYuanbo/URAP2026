from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_action_chunk_multiplicity_expert import cluster_count
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key,parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score,load_row_scores

def temporal_gate_map(data,score_threshold,window_seconds,min_fraction,fps_map):
 sequences=defaultdict(list)
 for image_id,item in data.items():
  seq,fid,_=image_key(str(image_id),0);signal=cluster_count(item.get('detections') or [],score_threshold)>=2;sequences[seq].append((fid,str(image_id),signal))
 gates={}
 for seq,items in sequences.items():
  items.sort();fps=float(fps_map.get(seq,25.));window=max(1,int(round(window_seconds*fps)));history=deque();active=0
  for fid,image_id,signal in items:
   history.append((fid,signal));active+=int(signal)
   while history and fid-history[0][0]>=window:active-=int(history.popleft()[1])
   gates[image_id]=(active/len(history))>=min_fraction
 return gates

def evaluate_config(data,base,expert,gates,weight,alpha,tvd_root,out_dir):
 output={}
 for image_id,item in data.items():
  seq,fid,_=image_key(str(image_id),0);gate=gates.get(str(image_id),False);rows=[]
  for index,row in enumerate(item.get('detections') or []):
   key=(seq,fid,index);raw=float(row.get('score',0.));base_score=max(1e-9,float(base.get(key,raw)));expert_score=max(1e-9,float(expert.get(key,base_score)));aux=math.exp((1.-weight)*math.log(base_score)+weight*math.log(expert_score)) if gate else base_score;new=dict(row);new['score']=fuse_score(raw,aux,alpha,'geom-mix');rows.append(new)
  output[image_id]={'labels':item.get('labels',[]),'detections':rows}
 return evaluate_data(output,tvd_root,out_dir)

def main():
 p=argparse.ArgumentParser(description='Causal true-time multiplicity Action Bank gate.');p.add_argument('--tvd-root',type=Path,required=True);p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--base-jsonl',type=Path,required=True);p.add_argument('--base-field',required=True);p.add_argument('--expert-jsonl',type=Path,required=True);p.add_argument('--expert-field',required=True);p.add_argument('--sequence-fps-json',type=Path,required=True);p.add_argument('--thresholds',default='.3,.4');p.add_argument('--windows',default='1,3');p.add_argument('--fractions',default='.25,.5,.75');p.add_argument('--expert-weights',default='.5');p.add_argument('--alphas',default='.3,.4');p.add_argument('--fixed-config-json',type=Path);p.add_argument('--out-json',type=Path,required=True);a=p.parse_args();data=load_predictionsgt(a.predictionsgt_pkl);base,_=load_row_scores(a.base_jsonl,a.base_field,1);expert,_=load_row_scores(a.expert_jsonl,a.expert_field,1);fps_map=json.loads(a.sequence_fps_json.read_text(encoding='utf8'));rows=[]
 if a.fixed_config_json:
  best=json.loads(a.fixed_config_json.read_text(encoding='utf8'))['best'];configs=[(float(best['threshold']),float(best['window_seconds']),float(best['min_fraction']),float(best['expert_weight']),float(best['alpha']))]
 else:configs=[(t,w,f,ew,alpha) for t in parse_csv_floats(a.thresholds) for w in parse_csv_floats(a.windows) for f in parse_csv_floats(a.fractions) for ew in parse_csv_floats(a.expert_weights) for alpha in parse_csv_floats(a.alphas)]
 gate_cache={}
 for threshold,window,fraction,weight,alpha in configs:
  cache_key=(threshold,window,fraction);gates=gate_cache.setdefault(cache_key,temporal_gate_map(data,threshold,window,fraction,fps_map));metrics=evaluate_config(data,base,expert,gates,weight,alpha,a.tvd_root,a.out_json.parent);row={'threshold':threshold,'window_seconds':window,'min_fraction':fraction,'expert_weight':weight,'alpha':alpha,'gated_images':sum(gates.values()),**metrics};rows.append(row);print(json.dumps(row),flush=True)
 best=max(rows,key=lambda row:float(row['map50']));a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps({'best':best,'rows':rows},indent=2),encoding='utf8');return 0
if __name__=='__main__':raise SystemExit(main())
