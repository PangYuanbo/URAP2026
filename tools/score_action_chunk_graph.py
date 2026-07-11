from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from qstr_dronedet.action_chunk_camera_motion import ActionChunkCameraMotionCache
from qstr_dronedet.camera_motion import transform_bbox_xyxy
from qstr_dronedet.candidates.merge import bbox_iou
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores

def finite(v):
 try:x=float(v)
 except(TypeError,ValueError):return 0.
 return x if math.isfinite(x) else 0.
def sigmoid(x):return 1./(1.+np.exp(-np.clip(x,-30,30)))
def logit(x):x=np.clip(x,1e-5,1-1e-5);return np.log(x/(1-x))
def center(box):return .5*(box[0]+box[2]),.5*(box[1]+box[3])
def size(box):return max(1.,box[2]-box[0]),max(1.,box[3]-box[1])
def compatibility(previous,current,matrix,validity):
 transformed=tuple(float(x) for x in transform_bbox_xyxy(previous,matrix));pc=center(transformed);cc=center(current);pw,ph=size(transformed);cw,ch=size(current);ref=max(5.,.25*(pw+ph+cw+ch));distance=math.hypot(cc[0]-pc[0],cc[1]-pc[1])/ref;center_score=math.exp(-distance);scale=math.exp(-abs(math.log(cw/pw))-abs(math.log(ch/ph)));overlap=bbox_iou(transformed,current);return float(np.clip((.45*center_score+.35*overlap+.20*scale)*(.75+.25*validity),1e-4,1.))
def normalize(values):
 if not len(values):return values
 median=float(np.median(values));scale=max(.25,float(np.std(values)));return np.clip((values-median)/scale,-8,8)
def transition_matrices(frames,cache,sequence,reverse):
 ordered=list(reversed(range(len(frames)))) if reverse else list(range(len(frames)));matrices={};previous_index=None
 for index in ordered:
  boxes=frames[index][2]
  if previous_index is not None:
   previous_boxes=frames[previous_index][2];matrix,validity=cache.between(sequence,frames[previous_index][0],frames[index][0]);values=np.empty((len(boxes),len(previous_boxes)),np.float32)
   for ci,current in enumerate(boxes):
    for pi,previous in enumerate(previous_boxes):values[ci,pi]=compatibility(previous,current,matrix,validity)
   matrices[index]=values
  previous_index=index
 return matrices
def pass_scores(frames,matrices,reverse,decay,transition_weight):
 ordered=list(reversed(range(len(frames)))) if reverse else list(range(len(frames)));output=[None]*len(frames);previous_state=None
 for index in ordered:
  unary=frames[index][3];transition=matrices.get(index)
  if previous_state is None or transition is None or not len(unary) or not transition.shape[1]:state=logit(unary)
  else:state=logit(unary)+decay*np.max(normalize(previous_state)[None,:]+transition_weight*logit(transition),axis=1)
  output[index]=normalize(state);previous_state=state
 return output
def main():
 p=argparse.ArgumentParser(description='Bidirectional camera-compensated Action Chunk candidate graph scorer.');p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--unary-scores',type=Path,required=True);p.add_argument('--unary-field',required=True);p.add_argument('--homography-cache',type=Path,required=True);p.add_argument('--out-jsonl',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);p.add_argument('--decays',default='.35,.55,.75');p.add_argument('--transition-weights',default='.4,.8,1.2');args=p.parse_args();pred=load_predictionsgt(args.predictionsgt_pkl);unary_map,_=load_row_scores(args.unary_scores,args.unary_field,1);grouped={}
 for iid,item in pred.items():seq,fid,_=image_key(str(iid),0);grouped.setdefault(seq,[]).append((fid,str(iid),item))
 for values in grouped.values():values.sort(key=lambda x:x[0])
 cache=ActionChunkCameraMotionCache(Path('.'),args.homography_cache,320);decays=[float(x) for x in args.decays.split(',')];weights=[float(x) for x in args.transition_weights.split(',')];fields=[];total=0;args.out_jsonl.parent.mkdir(parents=True,exist_ok=True)
 with args.out_jsonl.open('w',encoding='utf8') as target:
  for seq,items in sorted(grouped.items()):
   frames=[]
   for fid,iid,item in items:
    boxes=[tuple(float(z) for z in row['bbox']) for row in item.get('detections',[]) if isinstance(row.get('bbox'),list) and len(row['bbox'])==4];unary=np.asarray([unary_map.get((seq,fid,i),finite(row.get('score'))) for i,row in enumerate(item.get('detections',[])) if isinstance(row.get('bbox'),list) and len(row['bbox'])==4],np.float32);frames.append((fid,iid,boxes,unary))
   variants={};forward_matrices=transition_matrices(frames,cache,seq,False);backward_matrices=transition_matrices(frames,cache,seq,True)
   for decay in decays:
    for weight in weights:
     name=f'action_chunk_graph_d{int(round(decay*100)):02d}_t{int(round(weight*100)):03d}';forward=pass_scores(frames,forward_matrices,False,decay,weight);backward=pass_scores(frames,backward_matrices,True,decay,weight);variants[name]=[sigmoid(.5*(forward[i]+backward[i])) for i in range(len(frames))];fields.append(name)
   for fi,(fid,iid,boxes,_unary) in enumerate(frames):
    rows=[]
    for ci in range(len(boxes)):
     row={'seq':seq,'frame_id':fid,'prediction_index':ci}
     for name,values in variants.items():row[name]=float(values[fi][ci])
     rows.append(row);total+=1
    target.write(json.dumps({'meta':{'seq':seq,'image_id':iid},'rows':rows},separators=(',',':'))+'\n')
   print(json.dumps({'kind':'action_chunk_graph_sequence','sequence':seq,'frames':len(frames),'candidates':sum(len(x[2]) for x in frames)}),flush=True)
 summary={'kind':'action_chunk_graph_done','rows':total,'fields':sorted(set(fields)),'unary_field':args.unary_field,'decays':decays,'transition_weights':weights};args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
