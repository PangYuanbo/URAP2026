from __future__ import annotations
import argparse,json,math,sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from qstr_dronedet.tracking.online_action_bank import OnlineActionTrack
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.score_tracklets_samurai_cmc import HomographyCache,bbox_iou
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
def finite(v:Any)->float:
 try:x=float(v)
 except(TypeError,ValueError):return 0.0
 return x if math.isfinite(x) else 0.0
def box(r):return tuple(float(x) for x in r["bbox"])
def logit(v):v=min(1-1e-6,max(1e-6,float(v)));return math.log(v/(1-v))
def sigmoid(v):return 1/(1+math.exp(-max(-30,min(30,float(v)))))
@dataclass
class Hypothesis:
 track:OnlineActionTrack
 born:float
 hits:int=1
 consecutive:int=1
 misses:int=0
 evidence:float=.0
 def clone_update(self,fid,t,candidate,raw,motion,transform,long_seconds):
  tr=self.track.clone();tr.update(fid,t,candidate,raw,motion,transform,long_seconds);return Hypothesis(tr,self.born,self.hits+1,self.consecutive+1,0,.82*self.evidence+.18*(.55*raw+.45*motion))
 def persistence(self,now):
  age=max(1e-3,now-self.born);density=min(1.0,self.hits/max(2.0,age*15.0));duration=1-math.exp(-age/1.0);streak=1-math.exp(-self.consecutive/6.0);return max(0,min(1,.35*density+.30*duration+.25*streak+.10*self.evidence))
def main():
 p=argparse.ArgumentParser();p.add_argument("--predictionsgt-pkl",type=Path,required=True);p.add_argument("--frame-root",type=Path,required=True);p.add_argument("--homography-cache",type=Path,required=True);p.add_argument("--out-jsonl",type=Path,required=True);p.add_argument("--out-summary",type=Path,required=True);p.add_argument("--sequence-fps-json",type=Path,required=True);p.add_argument("--short-seconds",type=float,default=1.0);p.add_argument("--long-seconds",type=float,default=3.0);p.add_argument("--max-tracks",type=int,default=64);p.add_argument("--birth-gate",type=float,default=.025);p.add_argument("--motion-gate",type=float,default=.28);p.add_argument("--utility-gate",type=float,default=.04);p.add_argument("--birth-nms",type=float,default=.75);a=p.parse_args();data=load_predictionsgt(a.predictionsgt_pkl);fpsmap=json.loads(a.sequence_fps_json.read_text());cache=HomographyCache(a.frame_root,a.homography_cache,320);grouped={}
 for iid,item in data.items():seq,fid,_=image_key(str(iid),0);grouped.setdefault(seq,[]).append((fid,str(iid),item))
 for frames in grouped.values():frames.sort(key=lambda z:z[0])
 a.out_jsonl.parent.mkdir(parents=True,exist_ok=True);summaries=[];total=0
 with a.out_jsonl.open("w",encoding="utf-8") as out:
  for seq,frames in sorted(grouped.items()):
   fps=float(fpsmap.get(seq,30));hyps=[];matched_total=birth_total=0
   for fid,iid,item in frames:
    now=fid/fps;ds=list(item.get("detections") or []);active=[h for h in hyps if now-h.track.timestamp<=a.long_seconds];trans=[];valid=[]
    for h in active:m,v=cache.between(seq,h.track.frame_id,fid);trans.append(m);valid.append(v)
    utility=np.zeros((len(active),len(ds)),np.float32);motion=np.zeros_like(utility);details=[[None]*len(ds) for _ in active]
    for hi,h in enumerate(active):
     persist=h.persistence(now)
     for ci,d in enumerate(ds):
      z=h.track.score_candidate(box(d),now,trans[hi],valid[hi],a.short_seconds,a.long_seconds);details[hi][ci]=z;motion[hi,ci]=z.score;raw=finite(d.get("score"));utility[hi,ci]=sigmoid(.48*logit(raw)+.52*logit(z.score))*(.25+.75*persist)
    pairs=[]
    if len(active) and len(ds):
     rr,cc=linear_sum_assignment(-utility)
     pairs=[(h,c) for h,c in zip(rr.tolist(),cc.tolist()) if motion[h,c]>=a.motion_gate and utility[h,c]>=a.utility_gate]
    ah={h for h,_ in pairs};ac={c for _,c in pairs};new=[];candidate_h=[None]*len(ds)
    for hi,ci in pairs:
     h=active[hi].clone_update(fid,now,box(ds[ci]),finite(ds[ci].get("score")),float(motion[hi,ci]),trans[hi],a.long_seconds);new.append(h);candidate_h[ci]=h;matched_total+=1
    for hi,h in enumerate(active):
     if hi in ah:continue
     h.misses+=1;h.consecutive=0
     if now-h.track.timestamp<=a.long_seconds:new.append(h)
    occupied=[h.track.bbox for h in new]
    for ci in sorted((i for i in range(len(ds)) if i not in ac),key=lambda i:finite(ds[i].get("score")),reverse=True):
     raw=finite(ds[ci].get("score"));candidate=box(ds[ci])
     if raw<a.birth_gate:break
     if any(bbox_iou(candidate,b)>=a.birth_nms for b in occupied):continue
     h=Hypothesis(OnlineActionTrack(fid,now,candidate,raw),now,1,1,0,raw);new.append(h);candidate_h[ci]=h;occupied.append(candidate);birth_total+=1
     if len(new)>=a.max_tracks:break
    hyps=sorted(new,key=lambda h:(h.persistence(now),h.track.quality,-h.misses),reverse=True)[:a.max_tracks];rows=[]
    for ci,d in enumerate(ds):
     h=candidate_h[ci];raw=finite(d.get("score"));persist=h.persistence(now) if h else 0.;hit=min(1.,h.hits/12.) if h else 0.;duration=min(1.,(now-h.born)/3.) if h else 0.;streak=min(1.,h.consecutive/12.) if h else 0.;mot=float(motion[next((hi for hi,c in pairs if c==ci),0),ci]) if h and ci in ac and len(active) else 0.;score=max(0,min(1,.45*persist+.20*hit+.15*duration+.10*streak+.10*mot));rows.append({"seq":seq,"frame_id":fid,"prediction_index":ci,"persistence_score":score,"persistence":persist,"persistence_hits":hit,"persistence_duration":duration,"persistence_streak":streak,"persistence_motion":mot,"persistence_matched":float(ci in ac),"persistence_active_tracks":len(active)})
    out.write(json.dumps({"meta":{"seq":seq,"image_id":iid,"fps":fps},"rows":rows},separators=(",",":"))+"\n");total+=len(rows)
   s={"sequence":seq,"frames":len(frames),"matched":matched_total,"births":birth_total};summaries.append(s);print(json.dumps({"kind":"persistence_action_bank_sequence",**s}),flush=True)
 summary={"kind":"persistence_action_bank_done","rows":total,"max_tracks":a.max_tracks,"sequences":summaries};a.out_summary.parent.mkdir(parents=True,exist_ok=True);a.out_summary.write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
