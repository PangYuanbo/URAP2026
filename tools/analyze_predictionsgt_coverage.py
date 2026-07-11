from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np

def iou(a,b):
 ax1,ay1,ax2,ay2=map(float,a);bx1,by1,bx2,by2=map(float,b)
 inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
 return inter/max(1e-9,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

def main():
 p=argparse.ArgumentParser();p.add_argument('--predictionsgt-pkl',type=Path,required=True);p.add_argument('--out-json',type=Path,required=True);a=p.parse_args()
 with a.predictionsgt_pkl.open('rb') as h:data=pickle.load(h)
 thresholds=np.linspace(.5,.95,10);covered=np.zeros(10,dtype=np.int64);labels_total=dets_total=frames_none=frames_all=0;scores=[];best_ious=[]
 for item in data.values():
  dets=item.get('detections',[]);labels=item.get('labels',[]);dets_total+=len(dets);labels_total+=len(labels);frame_ok=True
  for label in labels:
   same=(d for d in dets if int(d.get('category_id',0))==int(label.get('category_id',0)))
   candidates=[(iou(label['bbox'],d['bbox']),float(d.get('score',0))) for d in same]
   best_iou,best_score=max(candidates,default=(0.0,0.0),key=lambda x:x[0]);covered += best_iou>=thresholds;best_ious.append(best_iou);scores.append(best_score);frame_ok &= best_iou>=.5
  if labels and frame_ok:frames_all+=1
  if labels and not any(iou(label['bbox'],d['bbox'])>=.5 for label in labels for d in dets if int(d.get('category_id',0))==int(label.get('category_id',0))):frames_none+=1
 out={'images':len(data),'labels':labels_total,'detections':dets_total,'coverage':[{'iou':float(t),'covered':int(c),'ratio':float(c/max(1,labels_total))} for t,c in zip(thresholds,covered)],'ranking_only_map50_upper_bound':float(covered[0]/max(1,labels_total)),'recoverable_map50_points_over_baseline_14_3754':float(100*(covered[0]/max(1,labels_total)-.14375435579806786)),'frames_all_labels_covered_iou50':frames_all,'frames_no_candidate_iou50':frames_none,'best_iou_mean':float(np.mean(best_ious)),'matched_candidate_score_mean':float(np.mean([s for s,b in zip(scores,best_ious) if b>=.5])) if any(b>=.5 for b in best_ious) else None}
 a.out_json.parent.mkdir(parents=True,exist_ok=True);a.out_json.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
