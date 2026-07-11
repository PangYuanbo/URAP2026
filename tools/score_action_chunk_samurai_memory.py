from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for path in (REPO,REPO/'tools'):
 if str(path) not in sys.path:sys.path.insert(0,str(path))
from qstr_dronedet.action_chunk_camera_motion import ActionChunkCameraMotionCache
from qstr_dronedet.camera_motion import transform_bbox_xyxy
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key

DECAYS=(.94,.98,.995)

def valid_rows(item):
 rows=[];indices=[]
 for index,row in enumerate(item.get('detections') or []):
  box=row.get('bbox')
  if isinstance(box,list) and len(box)==4:
   rows.append(row);indices.append(index)
 boxes=np.asarray([row['bbox'] for row in rows],np.float32).reshape(-1,4)
 scores=np.asarray([float(row.get('score',0.)) for row in rows],np.float32)
 return boxes,scores,indices

def transition(previous,current,matrix,validity):
 if not len(previous) or not len(current):return np.zeros((len(previous),len(current)),np.float32),np.zeros((len(previous),len(current)),np.float32)
 projected=np.asarray([transform_bbox_xyxy(tuple(map(float,box)),matrix) for box in previous],np.float32).reshape(-1,4)
 pc=np.stack(((projected[:,0]+projected[:,2])*.5,(projected[:,1]+projected[:,3])*.5),1)
 cc=np.stack(((current[:,0]+current[:,2])*.5,(current[:,1]+current[:,3])*.5),1)
 pw=np.maximum(1.,projected[:,2]-projected[:,0]);ph=np.maximum(1.,projected[:,3]-projected[:,1])
 cw=np.maximum(1.,current[:,2]-current[:,0]);ch=np.maximum(1.,current[:,3]-current[:,1])
 dx=pc[:,None,0]-cc[None,:,0];dy=pc[:,None,1]-cc[None,:,1]
 side=np.maximum(5.,.25*(pw[:,None]+ph[:,None]+cw[None,:]+ch[None,:]))
 residual=np.sqrt(dx*dx+dy*dy)/side
 center=np.exp(-residual)
 scale=np.exp(-np.abs(np.log(pw[:,None]/cw[None,:]))-np.abs(np.log(ph[:,None]/ch[None,:])))
 ix1=np.maximum(projected[:,None,0],current[None,:,0]);iy1=np.maximum(projected[:,None,1],current[None,:,1])
 ix2=np.minimum(projected[:,None,2],current[None,:,2]);iy2=np.minimum(projected[:,None,3],current[None,:,3])
 inter=np.maximum(0.,ix2-ix1)*np.maximum(0.,iy2-iy1)
 union=pw[:,None]*ph[:,None]+cw[None,:]*ch[None,:]-inter
 iou=inter/np.maximum(union,1e-6)
 similarity=(.55*center+.25*scale+.20*iou)*(.8+.2*float(validity))
 return similarity.astype(np.float32),residual.astype(np.float32)

def directional(items,cache,reverse=False):
 order=range(len(items)-1,-1,-1) if reverse else range(len(items))
 states=[None]*len(items);ages=[None]*len(items);motions=[None]*len(items)
 previous_index=None
 for item_index in order:
  fid,_,item=items[item_index];boxes,raw,_=valid_rows(item)
  current_states=np.repeat(raw[:,None],len(DECAYS),axis=1) if len(raw) else np.zeros((0,len(DECAYS)),np.float32)
  current_ages=np.ones(len(raw),np.float32);current_motion=np.zeros(len(raw),np.float32)
  if previous_index is not None and len(raw):
   previous_fid,_,previous_item=items[previous_index];previous_boxes,_,_=valid_rows(previous_item)
   if len(previous_boxes):
    matrix,validity=cache.between(str(image_key(str(items[item_index][1]),0)[0]),previous_fid,fid)
    similarity,residual=transition(previous_boxes,boxes,matrix,validity)
    best_indices=np.argmax(similarity,axis=0);best_similarity=similarity[best_indices,np.arange(len(raw))]
    linked=best_similarity>=.42
    previous_states=states[previous_index];previous_ages=ages[previous_index];previous_motion=motions[previous_index]
    for decay_index,decay in enumerate(DECAYS):
     propagated=previous_states[best_indices,decay_index]*decay*(.55+.45*best_similarity)
     current_states[:,decay_index]=np.maximum(raw,propagated)
    current_ages=np.where(linked,previous_ages[best_indices]+1.,1.)
    transition_motion=1.-np.exp(-residual[best_indices,np.arange(len(raw))]/.35)
    current_motion=np.where(linked,.85*previous_motion[best_indices]+.15*transition_motion,0.)
  states[item_index]=current_states;ages[item_index]=current_ages;motions[item_index]=current_motion;previous_index=item_index
 return states,ages,motions

def main():
 parser=argparse.ArgumentParser(description='SAMURAI-style high-confidence candidate memory with camera-compensated bidirectional propagation.')
 parser.add_argument('--predictionsgt-pkl',type=Path,required=True);parser.add_argument('--homography-cache',type=Path,required=True);parser.add_argument('--out-jsonl',type=Path,required=True);parser.add_argument('--out-summary',type=Path,required=True)
 args=parser.parse_args();data=load_predictionsgt(args.predictionsgt_pkl);grouped={}
 for image_id,item in data.items():
  seq,fid,_=image_key(str(image_id),0);grouped.setdefault(seq,[]).append((fid,str(image_id),item))
 for items in grouped.values():items.sort(key=lambda value:value[0])
 cache=ActionChunkCameraMotionCache(Path('.'),args.homography_cache,320);args.out_jsonl.parent.mkdir(parents=True,exist_ok=True);row_count=0
 fields=[f'samurai_memory_sym_{str(decay).replace(".","p")}' for decay in DECAYS]+[f'samurai_memory_min_{str(decay).replace(".","p")}' for decay in DECAYS]+['samurai_memory_span_1s','samurai_memory_span_3s','samurai_memory_motion','samurai_memory_motion_sym']
 with args.out_jsonl.open('w',encoding='utf8') as output:
  for seq,items in sorted(grouped.items()):
   forward,forward_age,forward_motion=directional(items,cache,False);backward,backward_age,backward_motion=directional(items,cache,True)
   fps=29.97
   for item_index,(fid,image_id,item) in enumerate(items):
    _,raw,indices=valid_rows(item);rows=[]
    for local_index,prediction_index in enumerate(indices):
     row={'seq':seq,'frame_id':fid,'prediction_index':prediction_index}
     for decay_index,decay in enumerate(DECAYS):
      tag=str(decay).replace('.','p');f=float(forward[item_index][local_index,decay_index]);b=float(backward[item_index][local_index,decay_index]);row[f'samurai_memory_sym_{tag}']=math.sqrt(max(1e-9,f*b));row[f'samurai_memory_min_{tag}']=min(f,b)
     span=float(forward_age[item_index][local_index]+backward_age[item_index][local_index]-1.);motion=.5*float(forward_motion[item_index][local_index]+backward_motion[item_index][local_index]);row['samurai_memory_span_1s']=float(1.-math.exp(-span/max(1.,fps)));row['samurai_memory_span_3s']=float(1.-math.exp(-span/max(1.,3.*fps)));row['samurai_memory_motion']=motion;row['samurai_memory_motion_sym']=float(row['samurai_memory_sym_0p98']*(.35+.65*motion));rows.append(row);row_count+=1
    output.write(json.dumps({'meta':{'seq':seq,'image_id':image_id},'rows':rows},separators=(',',':'))+'\n')
   print(json.dumps({'kind':'samurai_memory_sequence','sequence':seq,'frames':len(items),'rows':sum(len(valid_rows(item)[1]) for _,_,item in items)}),flush=True)
 summary={'kind':'samurai_memory_done','rows':row_count,'sequences':len(grouped),'fields':fields,'decays':DECAYS,'homography_cache':str(args.homography_cache)};args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
