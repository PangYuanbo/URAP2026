from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from qstr_dronedet.tracking.online_action_bank import OnlineActionTrack
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.score_tracklets_samurai_cmc import HomographyCache, bbox_iou
from tools.sweep_tvd_predictionsgt_action_rescore import image_key

def finite(v:Any)->float:
 try:x=float(v)
 except (TypeError,ValueError):return 0.0
 return x if math.isfinite(x) else 0.0
def box(row):return tuple(float(v) for v in row["bbox"])
def logit(v):
 v=min(1-1e-6,max(1e-6,float(v)));return math.log(v/(1-v))
def sigmoid(v):return 1/(1+math.exp(-max(-30,min(30,float(v)))))
def maturity(track):return min(1.0,max(0.0,track.observations-1)/8.0)
def track_rank(track):return track.quality*(0.12+0.88*maturity(track))
def main():
 p=argparse.ArgumentParser();p.add_argument("--predictionsgt-pkl",type=Path,required=True);p.add_argument("--frame-root",type=Path,required=True);p.add_argument("--homography-cache",type=Path,required=True);p.add_argument("--out-jsonl",type=Path,required=True);p.add_argument("--out-summary",type=Path,required=True);p.add_argument("--sequence-fps-json",type=Path,required=True);p.add_argument("--short-seconds",type=float,default=1.0);p.add_argument("--long-seconds",type=float,default=3.0);p.add_argument("--max-tracks",type=int,default=8);p.add_argument("--birth-gate",type=float,default=.18);p.add_argument("--assign-motion-gate",type=float,default=.42);p.add_argument("--assign-utility-gate",type=float,default=.12);p.add_argument("--motion-weight",type=float,default=.65);p.add_argument("--birth-nms",type=float,default=.55);a=p.parse_args()
 data=load_predictionsgt(a.predictionsgt_pkl);fps_map=json.loads(a.sequence_fps_json.read_text());cache=HomographyCache(a.frame_root,a.homography_cache,320);grouped={}
 for image_id,item in data.items():
  seq,fid,_=image_key(str(image_id),0);grouped.setdefault(seq,[]).append((fid,str(image_id),item))
 for frames in grouped.values():frames.sort(key=lambda x:x[0])
 a.out_jsonl.parent.mkdir(parents=True,exist_ok=True);sequence_summaries=[];total_rows=assigned_total=birth_total=0
 with a.out_jsonl.open("w",encoding="utf-8") as out:
  for seq,frames in sorted(grouped.items()):
   fps=float(fps_map.get(seq,30.0));tracks=[];seq_assigned=seq_births=0
   for fid,image_id,item in frames:
    timestamp=fid/fps;detections=list(item.get("detections") or []);active=[t for t in tracks if timestamp-t.timestamp<=a.long_seconds];transforms=[];validities=[]
    for track in active:
     transform,valid=cache.between(seq,track.frame_id,fid);transforms.append(transform);validities.append(valid)
    pair_scores=np.zeros((len(active),len(detections)),dtype=np.float32);motion_scores=np.zeros_like(pair_scores);details=[[None for _ in detections] for _ in active]
    for ti,track in enumerate(active):
     for ci,det in enumerate(detections):
      detail=track.score_candidate(box(det),timestamp,transforms[ti],validities[ti],a.short_seconds,a.long_seconds);details[ti][ci]=detail;motion_scores[ti,ci]=detail.score
      raw=finite(det.get("score"));pair_scores[ti,ci]=sigmoid((1-a.motion_weight)*logit(raw)+a.motion_weight*logit(detail.score))*(.35+.65*maturity(track))
    assignments=[]
    if len(active) and len(detections):
     row_idx,col_idx=linear_sum_assignment(-pair_scores)
     for ti,ci in zip(row_idx.tolist(),col_idx.tolist()):
      if motion_scores[ti,ci]>=a.assign_motion_gate and pair_scores[ti,ci]>=a.assign_utility_gate:assignments.append((ti,ci))
    assigned_tracks={ti for ti,_ in assignments};assigned_candidates={ci for _,ci in assignments};updated=[]
    candidate_track=np.full((len(detections),),-1,dtype=np.int32);candidate_motion=np.zeros((len(detections),),dtype=np.float32);candidate_utility=np.zeros((len(detections),),dtype=np.float32);candidate_quality=np.zeros((len(detections),),dtype=np.float32)
    for ti,ci in assignments:
     track=active[ti].clone();detail=details[ti][ci];raw=finite(detections[ci].get("score"));track.update(fid,timestamp,box(detections[ci]),raw,detail.score,transforms[ti],a.long_seconds);updated.append(track);candidate_track[ci]=ti;candidate_motion[ci]=detail.score;candidate_utility[ci]=pair_scores[ti,ci];candidate_quality[ci]=track_rank(track);seq_assigned+=1
    unmatched=[track for ti,track in enumerate(active) if ti not in assigned_tracks]
    selected_boxes=[box(detections[ci]) for ci in assigned_candidates];birth_indices=[]
    for ci in sorted((i for i in range(len(detections)) if i not in assigned_candidates),key=lambda i:finite(detections[i].get("score")),reverse=True):
     raw=finite(detections[ci].get("score"));candidate=box(detections[ci])
     if raw<a.birth_gate:break
     if any(bbox_iou(candidate,kept)>=a.birth_nms for kept in selected_boxes):continue
     birth_indices.append(ci);selected_boxes.append(candidate);updated.append(OnlineActionTrack(fid,timestamp,candidate,raw));candidate_utility[ci]=raw;candidate_quality[ci]=.12*raw;seq_births+=1
     if len(updated)+len(unmatched)>=a.max_tracks:break
    birth_set=set(birth_indices);selected_set=assigned_candidates|birth_set
    survivors=updated+sorted(unmatched,key=track_rank,reverse=True);tracks=sorted(survivors,key=track_rank,reverse=True)[:a.max_tracks]
    rows=[]
    for ci,det in enumerate(detections):
     candidate=box(det);raw=finite(det.get("score"));duplicate=max((bbox_iou(candidate,box(detections[s])) for s in selected_set if s!=ci),default=0.0);assigned=ci in assigned_candidates;birth=ci in birth_set;selected=assigned or birth
     motion=float(candidate_motion[ci]);utility=float(candidate_utility[ci]);quality=float(candidate_quality[ci]);selection_conf=utility if assigned else (raw if birth else 0.0);joint=max(0.0,min(1.0,(.55*selection_conf+.25*motion+.20*quality) if selected else raw*(1-.85*duplicate)*.25));suppress=max(0.0,min(1.0,joint*(1-duplicate) if not selected else joint))
     rows.append({"seq":seq,"frame_id":fid,"prediction_index":ci,"joint_assigned":float(assigned),"joint_birth":float(birth),"joint_selected":float(selected),"joint_motion":motion,"joint_utility":utility,"joint_track_quality":quality,"joint_duplicate_iou":float(duplicate),"joint_score":joint,"joint_suppress_score":suppress,"joint_active_tracks":len(active),"joint_selected_count":len(selected_set)})
    out.write(json.dumps({"meta":{"seq":seq,"image_id":image_id,"fps":fps},"rows":rows},separators=(",",":"))+"\n");total_rows+=len(rows)
   assigned_total+=seq_assigned;birth_total+=seq_births;summary={"sequence":seq,"frames":len(frames),"assigned":seq_assigned,"births":seq_births};sequence_summaries.append(summary);print(json.dumps({"kind":"joint_action_bank_sequence",**summary}),flush=True)
 summary={"kind":"joint_action_bank_done","rows":total_rows,"assigned":assigned_total,"births":birth_total,"max_tracks":a.max_tracks,"parameters":{"birth_gate":a.birth_gate,"assign_motion_gate":a.assign_motion_gate,"assign_utility_gate":a.assign_utility_gate,"motion_weight":a.motion_weight,"birth_nms":a.birth_nms},"sequences":sequence_summaries};a.out_summary.parent.mkdir(parents=True,exist_ok=True);a.out_summary.write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
