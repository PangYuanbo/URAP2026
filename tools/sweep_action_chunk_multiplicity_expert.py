from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tools.action_chunk_candidate_context import _iou_matrix
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key,parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score,load_row_scores

def cluster_count(detections,score_threshold,iou_threshold=.3):
 valid=[row for row in detections if isinstance(row.get('bbox'),list) and len(row['bbox'])==4 and float(row.get('score',0.))>=score_threshold]
 if not valid:return 0
 scores=np.asarray([float(row.get('score',0.)) for row in valid]);boxes=np.asarray([row['bbox'] for row in valid],np.float32);overlaps=_iou_matrix(boxes);leaders=[]
 for candidate in np.argsort(scores)[::-1]:
  if not any(overlaps[candidate,leader]>iou_threshold for leader in leaders):leaders.append(int(candidate))
 return len(leaders)

def evaluate_config(data,base,expert,threshold,weight,alpha,tvd_root,out_dir):
 output={}
 for image_id,item in data.items():
  seq,fid,_=image_key(str(image_id),0);gate=cluster_count(item.get('detections') or [],threshold)>=2;rows=[]
  for index,row in enumerate(item.get('detections') or []):
   key=(seq,fid,index);raw=float(row.get('score',0.));base_score=max(1e-9,float(base.get(key,raw)));expert_score=max(1e-9,float(expert.get(key,base_score)));aux=math.exp((1.-weight)*math.log(base_score)+weight*math.log(expert_score)) if gate else base_score;new=dict(row);new['score']=fuse_score(raw,aux,alpha,'geom-mix');rows.append(new)
  output[image_id]={'labels':item.get('labels',[]),'detections':rows}
 return evaluate_data(output,tvd_root,out_dir)

def main():
 p=argparse.ArgumentParser(description='Select a label-free multiplicity gate for a multi-target Action Chunk expert.');p.add_argument('--tvd-root',type=Path,required=True);p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--base-jsonl',type=Path,required=True);p.add_argument('--base-field',required=True);p.add_argument('--expert-jsonl',type=Path,required=True);p.add_argument('--expert-field',required=True);p.add_argument('--thresholds',default='.1,.2,.3,.4');p.add_argument('--expert-weights',default='.5,1');p.add_argument('--alphas',default='.2,.3,.4');p.add_argument('--fixed-config-json',type=Path);p.add_argument('--out-json',type=Path,required=True);a=p.parse_args();data=load_predictionsgt(a.predictionsgt_pkl);base,_=load_row_scores(a.base_jsonl,a.base_field,1);expert,_=load_row_scores(a.expert_jsonl,a.expert_field,1);rows=[]
 if a.fixed_config_json:
  config=json.loads(a.fixed_config_json.read_text(encoding='utf8'))['best'];configs=[(float(config['threshold']),float(config['expert_weight']),float(config['alpha']))]
 else:configs=[(t,w,alpha) for t in parse_csv_floats(a.thresholds) for w in parse_csv_floats(a.expert_weights) for alpha in parse_csv_floats(a.alphas)]
 for threshold,weight,alpha in configs:
  metrics=evaluate_config(data,base,expert,threshold,weight,alpha,a.tvd_root,a.out_json.parent);row={'threshold':threshold,'expert_weight':weight,'alpha':alpha,**metrics};rows.append(row);print(json.dumps(row),flush=True)
 best=max(rows,key=lambda row:float(row['map50']));summary={'best':best,'rows':rows};a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps(summary,indent=2),encoding='utf8');return 0
if __name__=='__main__':raise SystemExit(main())
