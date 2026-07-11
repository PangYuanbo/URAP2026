from __future__ import annotations
import argparse,json,math
from pathlib import Path
TOKEN_DIM=12
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-jsonl',type=Path,required=True);p.add_argument('--output-jsonl',type=Path,required=True);p.add_argument('--center',type=float,default=.8);p.add_argument('--width',type=float,default=.4);a=p.parse_args();a.output_jsonl.parent.mkdir(parents=True,exist_ok=True);rows=0
 with a.input_jsonl.open(encoding='utf-8-sig') as source,a.output_jsonl.open('w',encoding='utf-8') as target:
  for line in source:
   if not line.strip():continue
   item=json.loads(line);out=[]
   for r in item.get('rows') or []:
    values=[]
    for name in ('online_action_bank_short_tokens','online_action_bank_long_tokens'):
     token=r.get(name) or []
     for start in range(0,len(token)-TOKEN_DIM+1,TOKEN_DIM):
      if float(token[start])>.5:values.extend((float(token[start+1]),float(token[start+2])))
    magnitude=math.sqrt(sum(value*value for value in values));score=math.exp(-((magnitude-a.center)/max(1e-6,a.width))**2)
    out.append({'seq':r.get('seq'),'frame_id':r.get('frame_id'),'prediction_index':r.get('prediction_index'),'dynamic_band_score':score,'dynamic_residual_magnitude':magnitude});rows+=1
   target.write(json.dumps({'meta':item.get('meta') or {},'rows':out},separators=(',',':'))+'\n')
 print(json.dumps({'rows':rows,'center':a.center,'width':a.width,'causal':True},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
