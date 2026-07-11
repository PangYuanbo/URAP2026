from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def percentile(values):
 if len(values)<=1:return np.full(len(values),.5,np.float32)
 order=np.argsort(np.argsort(values,kind='stable'),kind='stable');return order.astype(np.float32)/(len(values)-1)
def sigmoid_z(values):
 median=float(np.median(values));scale=max(1e-4,float(np.std(values)));return 1/(1+np.exp(-np.clip((values-median)/scale,-12,12)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--field',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();items=[];seq_groups=defaultdict(list)
 for line in a.input.open(encoding='utf8'):
  item=json.loads(line);items.append(item);meta=item.get('meta') or {};seq=str(meta.get('seq') or '')
  for row in item.get('rows') or []:seq_groups[str(row.get('seq') or seq)].append(row)
 for rows in seq_groups.values():
  values=np.asarray([float(row[a.field]) for row in rows],np.float32);rank=percentile(values);z=sigmoid_z(values)
  for row,r,s in zip(rows,rank,z):row[a.field+'_seq_rank']=float(r);row[a.field+'_seq_z']=float(s)
 for item in items:
  rows=item.get('rows') or [];values=np.asarray([float(row[a.field]) for row in rows],np.float32);rank=percentile(values);z=sigmoid_z(values)
  for row,r,s in zip(rows,rank,z):row[a.field+'_frame_rank']=float(r);row[a.field+'_frame_z']=float(s)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',encoding='utf8') as target:
  for item in items:target.write(json.dumps(item,separators=(',',':'))+'\n')
 print(json.dumps({'rows':sum(len(i.get('rows') or []) for i in items),'sequences':len(seq_groups)}));return 0
if __name__=='__main__':raise SystemExit(main())
