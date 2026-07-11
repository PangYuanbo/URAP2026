from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
for p in (REPO,REPO/'tools'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from qstr_dronedet.action_chunk_camera_motion import ActionChunkCameraMotionCache
from qstr_dronedet.camera_motion import transform_bbox_xyxy
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key

def box_array(rows):return np.asarray([row['bbox'] for row in rows if isinstance(row.get('bbox'),list) and len(row['bbox'])==4],np.float32).reshape(-1,4)
def descriptors(current,reference,reference_scores):
 n=len(current)
 if not n or not len(reference):return np.zeros((n,6),np.float32)
 cc=np.stack(((current[:,0]+current[:,2])*.5,(current[:,1]+current[:,3])*.5),axis=1);rc=np.stack(((reference[:,0]+reference[:,2])*.5,(reference[:,1]+reference[:,3])*.5),axis=1);cw=np.maximum(1.,current[:,2]-current[:,0]);ch=np.maximum(1.,current[:,3]-current[:,1]);rw=np.maximum(1.,reference[:,2]-reference[:,0]);rh=np.maximum(1.,reference[:,3]-reference[:,1]);dx=cc[:,None,0]-rc[None,:,0];dy=cc[:,None,1]-rc[None,:,1];refside=np.maximum(5.,.25*(cw[:,None]+ch[:,None]+rw[None,:]+rh[None,:]));residual=np.sqrt(dx*dx+dy*dy)/refside;center=np.exp(-residual);scale=np.exp(-np.abs(np.log(cw[:,None]/rw[None,:]))-np.abs(np.log(ch[:,None]/rh[None,:])));ix1=np.maximum(current[:,None,0],reference[None,:,0]);iy1=np.maximum(current[:,None,1],reference[None,:,1]);ix2=np.minimum(current[:,None,2],reference[None,:,2]);iy2=np.minimum(current[:,None,3],reference[None,:,3]);inter=np.maximum(0.,ix2-ix1)*np.maximum(0.,iy2-iy1);union=cw[:,None]*ch[:,None]+rw[None,:]*rh[None,:]-inter;iou=inter/np.maximum(union,1e-6);match=.50*center+.25*scale+.15*iou+.10*reference_scores[None,:];order=np.argsort(match,axis=1);best=order[:,-1];second=order[:,-2] if len(reference)>1 else best;rows=np.arange(n);return np.stack((iou[rows,best],center[rows,best],scale[rows,best],residual[rows,best],reference_scores[best],match[rows,best]-match[rows,second]),axis=1).astype(np.float32)
def main():
 p=argparse.ArgumentParser(description='Real-time Action Chunk neighbor-bank features at true-time offsets.');p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--homography-cache',type=Path,required=True);p.add_argument('--out-jsonl',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);p.add_argument('--sequence-fps-json',type=Path,required=True);p.add_argument('--seconds',default='.25,1,3');p.add_argument('--bidirectional',action='store_true');a=p.parse_args();pred=load_predictionsgt(a.predictionsgt_pkl);fps_map=json.loads(a.sequence_fps_json.read_text());grouped={}
 for iid,item in pred.items():seq,fid,_=image_key(str(iid),0);grouped.setdefault(seq,[]).append((fid,str(iid),item))
 for values in grouped.values():values.sort(key=lambda x:x[0])
 seconds=[float(x) for x in a.seconds.split(',')];cache=ActionChunkCameraMotionCache(Path('.'),a.homography_cache,320);total=0;a.out_jsonl.parent.mkdir(parents=True,exist_ok=True)
 with a.out_jsonl.open('w',encoding='utf8') as target:
  for seq,items in sorted(grouped.items()):
   fps=float(fps_map.get(seq,30.));lookup={fid:(iid,item) for fid,iid,item in items}
   for fid,iid,item in items:
    detections=item.get('detections',[]);current=box_array(detections);features={}
    directions=(-1,1) if a.bidirectional else (-1,)
    for direction in directions:
     prefix='past' if direction<0 else 'future'
     for sec in seconds:
      offset=max(1,int(round(sec*fps)));ref_fid=fid+direction*offset;ref_entry=lookup.get(ref_fid)
      if ref_entry is None:values=np.zeros((len(current),6),np.float32)
      else:
       ref_rows=ref_entry[1].get('detections',[]);reference=box_array(ref_rows);scores=np.asarray([float(row.get('score',0)) for row in ref_rows if isinstance(row.get('bbox'),list) and len(row['bbox'])==4],np.float32);matrix,validity=cache.between(seq,ref_fid,fid);reference=np.asarray([transform_bbox_xyxy(tuple(map(float,box)),matrix) for box in reference],np.float32).reshape(-1,4);values=descriptors(current,reference,scores);values[:,1:3]*=(.75+.25*validity)
      tag=str(sec).replace('.','p')
      for col,name in enumerate(('iou','center','scale','residual','reference_score','margin')):features[f'action_chunk_neighbor_{prefix}_{tag}_{name}']=values[:,col]
    rows=[]
    for index in range(len(current)):
     row={'seq':seq,'frame_id':fid,'prediction_index':index}
     for name,values in features.items():row[name]=float(values[index])
     rows.append(row);total+=1
    target.write(json.dumps({'meta':{'seq':seq,'image_id':iid,'fps':fps},'rows':rows},separators=(',',':'))+'\n')
   print(json.dumps({'kind':'action_chunk_neighbor_sequence','sequence':seq,'frames':len(items)}),flush=True)
 summary={'kind':'action_chunk_neighbor_done','rows':total,'seconds':seconds,'bidirectional':a.bidirectional};a.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
