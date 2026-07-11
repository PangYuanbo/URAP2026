from __future__ import annotations
import argparse,bisect,json,math,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
 if str(candidate) not in sys.path:sys.path.insert(0,str(candidate))
from qstr_dronedet.action_chunk_camera_motion import ActionChunkCameraMotionCache
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores
VARIANTS={'short':(.22,.45,(1.,0.,0.)),'balanced':(.34,.60,(1.,.7,.35)),'long':(.38,.58,(.7,.8,.65)),'strong':(.48,.78,(1.,.75,.4))}
LAGS=(.25,1.,3.)
def finite(value):
 try:return float(value)
 except (TypeError,ValueError):return 0.0
def logit(values):
 values=np.clip(values,1e-6,1-1e-6);return np.log(values/(1-values))
def sigmoid(values):return 1/(1+np.exp(-np.clip(values,-30,30)))
def normalize(values):
 if not len(values):return values
 return np.clip((values-float(np.median(values)))/max(.25,float(np.std(values))),-8,8)
def transform_boxes(boxes,matrix):
 if not len(boxes):return boxes.copy()
 corners=np.stack((boxes[:,[0,1]],boxes[:,[2,1]],boxes[:,[2,3]],boxes[:,[0,3]]),axis=1);hom=np.concatenate((corners,np.ones((*corners.shape[:2],1))),axis=2);mapped=hom@matrix.T;mapped=mapped[...,:2]/np.clip(mapped[...,2:3],1e-9,None);mins=mapped.min(axis=1);maxs=mapped.max(axis=1);return np.concatenate((mins,maxs),axis=1)
def compatibility(previous,current,matrix,dt):
 if not len(previous) or not len(current):return np.zeros((len(current),len(previous)),np.float32)
 p=transform_boxes(previous,matrix);pc=.5*(p[:,:2]+p[:,2:]);cc=.5*(current[:,:2]+current[:,2:]);ps=np.maximum(1e-3,p[:,2:]-p[:,:2]);cs=np.maximum(1e-3,current[:,2:]-current[:,:2]);ref=.25*(ps[:,0]+ps[:,1])[None,:]+.25*(cs[:,0]+cs[:,1])[:,None];distance=np.sqrt(((cc[:,None,:]-pc[None,:,:])**2).sum(2))/np.maximum(5.,ref*(1.+1.5*dt));center=np.exp(-distance);scale=np.exp(-np.abs(np.log(cs[:,None,0]/ps[None,:,0]))-np.abs(np.log(cs[:,None,1]/ps[None,:,1])));ix1=np.maximum(current[:,None,0],p[None,:,0]);iy1=np.maximum(current[:,None,1],p[None,:,1]);ix2=np.minimum(current[:,None,2],p[None,:,2]);iy2=np.minimum(current[:,None,3],p[None,:,3]);inter=np.maximum(0.,ix2-ix1)*np.maximum(0.,iy2-iy1);ca=(cs[:,0]*cs[:,1])[:,None];pa=(ps[:,0]*ps[:,1])[None,:];iou=inter/np.maximum(1e-6,ca+pa-inter);return np.clip(.5*center+.3*iou+.2*scale,1e-4,1.).astype(np.float32)
def nearest_index(frame_ids,current_index,lag_frames,reverse):
 target=frame_ids[current_index]+lag_frames if reverse else frame_ids[current_index]-lag_frames
 if reverse:
  index=bisect.bisect_left(frame_ids,target,current_index+1)
  return index if index<len(frame_ids) else None
 index=bisect.bisect_right(frame_ids,target,0,current_index)-1
 return index if index>=0 else None
def pass_scores(sequence,frames,fps,cache,reverse):
 frame_ids=[frame[0] for frame in frames];order=range(len(frames)-1,-1,-1) if reverse else range(len(frames));states={name:[None]*len(frames) for name in VARIANTS}
 for index in order:
  fid,_,boxes,unary=frames[index];links=[]
  for lag in LAGS:
   previous_index=nearest_index(frame_ids,index,max(1,int(round(lag*fps))),reverse)
   if previous_index is None:links.append(None);continue
   previous_fid=frame_ids[previous_index];matrix,_=cache.between(sequence,previous_fid,fid);dt=abs(fid-previous_fid)/fps;links.append((previous_index,compatibility(frames[previous_index][2],boxes,matrix,dt)))
  for name,(decay,transition_weight,lag_weights) in VARIANTS.items():
   supports=[];weights=[]
   for link,lag_weight in zip(links,lag_weights):
    if link is None or lag_weight<=0:continue
    previous_index,transition=link;previous_state=states[name][previous_index]
    if previous_state is None or not transition.shape[1]:continue
    support=np.max(normalize(previous_state)[None,:]+transition_weight*logit(transition),axis=1);supports.append(support);weights.append(lag_weight)
   state=logit(unary)
   if supports:state=state+decay*sum(weight*support for weight,support in zip(weights,supports))/sum(weights)
   states[name][index]=state.astype(np.float32)
 return states
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--unary-scores',type=Path,required=True);p.add_argument('--unary-field',required=True);p.add_argument('--homography-cache',type=Path,required=True);p.add_argument('--fps-json',type=Path,required=True);p.add_argument('--out-jsonl',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);a=p.parse_args();pred=load_predictionsgt(a.predictionsgt_pkl);unary_map,_=load_row_scores(a.unary_scores,a.unary_field,1);fps_map=json.loads(a.fps_json.read_text(encoding='utf8'));grouped={}
 for iid,item in pred.items():seq,fid,_=image_key(str(iid),0);grouped.setdefault(seq,[]).append((fid,str(iid),item))
 for values in grouped.values():values.sort(key=lambda x:x[0])
 cache=ActionChunkCameraMotionCache(Path('.'),a.homography_cache,320);a.out_jsonl.parent.mkdir(parents=True,exist_ok=True);total=0
 with a.out_jsonl.open('w',encoding='utf8') as target:
  for sequence,items in sorted(grouped.items()):
   frames=[]
   for fid,iid,item in items:
    boxes=np.asarray([row.get('bbox') for row in item.get('detections') or []],np.float32);boxes=boxes if boxes.size else np.zeros((0,4),np.float32);unary=np.asarray([unary_map.get((sequence,fid,index),finite(row.get('score'))) for index,row in enumerate(item.get('detections') or [])],np.float32);frames.append((fid,iid,boxes,unary))
   fps=float(fps_map.get(sequence,30.));forward=pass_scores(sequence,frames,fps,cache,False);backward=pass_scores(sequence,frames,fps,cache,True)
   for index,(fid,iid,boxes,_unary) in enumerate(frames):
    rows=[]
    unary_logit=logit(_unary)
    for candidate_index in range(len(boxes)):
     row={'seq':sequence,'frame_id':fid,'prediction_index':candidate_index}
     for name in VARIANTS:
      f=float(sigmoid(normalize(forward[name][index]))[candidate_index]);b=float(sigmoid(normalize(backward[name][index]))[candidate_index]);fr=float(sigmoid(normalize(forward[name][index]-unary_logit))[candidate_index]);br=float(sigmoid(normalize(backward[name][index]-unary_logit))[candidate_index]);row[f'action_chunk_msgraph_{name}_forward']=f;row[f'action_chunk_msgraph_{name}_bidir']=math.sqrt(max(1e-6,f)*max(1e-6,b));row[f'action_chunk_msgraph_{name}_residual_forward']=fr;row[f'action_chunk_msgraph_{name}_residual_bidir']=math.sqrt(max(1e-6,fr)*max(1e-6,br))
     rows.append(row);total+=1
    target.write(json.dumps({'meta':{'seq':sequence,'image_id':iid,'fps':fps},'rows':rows},separators=(',',':'))+'\n')
   print(json.dumps({'kind':'action_chunk_msgraph_sequence','sequence':sequence,'fps':fps,'frames':len(frames),'candidates':sum(len(frame[2]) for frame in frames)}),flush=True)
 summary={'kind':'action_chunk_msgraph_done','rows':total,'lags_seconds':LAGS,'variants':VARIANTS,'fields':[f'action_chunk_msgraph_{name}_{direction}' for name in VARIANTS for direction in ('forward','bidir','residual_forward','residual_bidir')]};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
