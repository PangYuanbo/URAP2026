from __future__ import annotations
import argparse,json,math
from pathlib import Path

def finite(value):
 try:return float(value)
 except (TypeError,ValueError):return 0.0
def load(path):
 output={}
 with path.open(encoding='utf8') as source:
  for line in source:
   payload=json.loads(line);meta=payload.get('meta') or {}
   for row in payload.get('rows') or []:
    key=(str(row.get('seq') or meta.get('seq')),int(row.get('frame_id')),int(row.get('prediction_index')));output[key]=row
 return output
def geom(a,b):return math.sqrt(max(1e-6,a)*max(1e-6,b))
def main():
 p=argparse.ArgumentParser();p.add_argument('--forward-jsonl',type=Path,required=True);p.add_argument('--reverse-jsonl',type=Path,required=True);p.add_argument('--out-jsonl',type=Path,required=True);a=p.parse_args();reverse=load(a.reverse_jsonl);a.out_jsonl.parent.mkdir(parents=True,exist_ok=True);frames=rows_total=matched=0
 with a.forward_jsonl.open(encoding='utf8') as source,a.out_jsonl.open('w',encoding='utf8') as target:
  for line in source:
   payload=json.loads(line);meta=payload.get('meta') or {};output=[]
   for forward in payload.get('rows') or []:
    key=(str(forward.get('seq') or meta.get('seq')),int(forward.get('frame_id')),int(forward.get('prediction_index')));backward=reverse.get(key,{});matched+=int(bool(backward));fs=finite(forward.get('action_chunk_bank_score'));bs=finite(backward.get('action_chunk_bank_score'));fi=finite(forward.get('action_chunk_bank_predicted_iou'));bi=finite(backward.get('action_chunk_bank_predicted_iou'));fq=finite(forward.get('action_chunk_bank_track_quality'));bq=finite(backward.get('action_chunk_bank_track_quality'));fm=min(1.0,finite(forward.get('action_chunk_bank_track_observations'))/8.0)*min(1.0,finite(forward.get('action_chunk_bank_chain_duration_seconds'))/1.0);bm=min(1.0,finite(backward.get('action_chunk_bank_track_observations'))/8.0)*min(1.0,finite(backward.get('action_chunk_bank_chain_duration_seconds'))/1.0);fstable=finite(forward.get('action_chunk_bank_motion_stability'));bstable=finite(backward.get('action_chunk_bank_motion_stability'));base=geom(fs,bs);compat=geom(max(fi,1e-4),max(bi,1e-4));maturity=geom(max(fm,1e-4),max(bm,1e-4));quality=geom(max(fq,1e-4),max(bq,1e-4));stability=geom(max(fstable,1e-4),max(bstable,1e-4));output.append({'seq':key[0],'frame_id':key[1],'prediction_index':key[2],'action_chunk_bidir_chain_score':base,'action_chunk_bidir_chain_min':min(fs,bs),'action_chunk_bidir_chain_compatibility':geom(base,compat),'action_chunk_bidir_chain_mature':geom(base,maturity),'action_chunk_bidir_chain_quality':geom(base,quality),'action_chunk_bidir_chain_stable':geom(geom(base,maturity),stability),'action_chunk_bidir_chain_full':(base*compat*maturity*quality*stability)**.2});rows_total+=1
   target.write(json.dumps({'meta':meta,'rows':output},separators=(',',':'))+'\n');frames+=1
 print(json.dumps({'kind':'bidir_stable_chain_done','frames':frames,'rows':rows_total,'matched':matched,'output':str(a.out_jsonl)}),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
