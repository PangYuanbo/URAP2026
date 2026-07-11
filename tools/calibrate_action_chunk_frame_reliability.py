from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def sigmoid(value):return 1.0/(1.0+math.exp(-max(-30.0,min(30.0,float(value)))))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-jsonl',type=Path,required=True);p.add_argument('--input-field',required=True);p.add_argument('--output-jsonl',type=Path,required=True);a=p.parse_args();a.output_jsonl.parent.mkdir(parents=True,exist_ok=True);frames=rows_total=0
 with a.input_jsonl.open(encoding='utf8') as source,a.output_jsonl.open('w',encoding='utf8') as target:
  for line in source:
   payload=json.loads(line);rows=payload.get('rows') or [];values=np.asarray([float(row.get(a.input_field,0.0)) for row in rows],np.float64)
   if len(values):
    order=np.argsort(values,kind='stable');ranks=np.empty(len(values),np.float64);ranks[order]=np.arange(len(values),dtype=np.float64);percentile=(ranks+.5)/len(values);median=float(np.median(values));scale=max(1e-6,float(np.std(values)));robust=np.asarray([sigmoid((value-median)/scale) for value in values],np.float64)
   else:percentile=robust=np.zeros(0,np.float64)
   output=[]
   for index,row in enumerate(rows):
    updated={k:v for k,v in row.items() if k!=a.input_field};updated['action_chunk_future_rank']=float(percentile[index]);updated['action_chunk_future_frame_z']=float(robust[index]);output.append(updated);rows_total+=1
   target.write(json.dumps({'meta':payload.get('meta') or {},'rows':output},separators=(',',':'))+'\n');frames+=1
 print(json.dumps({'kind':'frame_reliability_calibration_done','frames':frames,'rows':rows_total,'output':str(a.output_jsonl)}),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
