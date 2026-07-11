from __future__ import annotations
import argparse,json,math
from pathlib import Path

def gmean(values):
 values=[max(1e-6,min(1.,float(v))) for v in values];return math.exp(sum(math.log(v) for v in values)/len(values))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();count=0;a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.input.open(encoding='utf8') as source,a.output.open('w',encoding='utf8') as target:
  for line in source:
   item=json.loads(line)
   for row in item.get('rows') or []:
    def v(direction,sec,name):return float(row.get(f'action_chunk_neighbor_{direction}_{sec}_{name}',0.))
    p25,f25=v('past','0p25','reference_score'),v('future','0p25','reference_score');p1,f1=v('past','1p0','reference_score'),v('future','1p0','reference_score');p3,f3=v('past','3p0','reference_score'),v('future','3p0','reference_score');r25=.5*(v('past','0p25','residual')+v('future','0p25','residual'));i25=.5*(v('past','0p25','iou')+v('future','0p25','iou'));short=gmean((p25,f25));medium=gmean((p1,f1));long=gmean((p3,f3));row['action_chunk_neighbor_support_short']=short;row['action_chunk_neighbor_support_short_medium']=gmean((short,medium));row['action_chunk_neighbor_support_multiscale']=gmean((short,medium,long));row['action_chunk_neighbor_support_motion']=max(0.,min(1.,short*math.exp(-.025*r25)*(1.-.5*i25)));row['action_chunk_neighbor_support_past']=gmean((p25,p1));count+=1
   target.write(json.dumps(item,separators=(',',':'))+'\n')
 print(json.dumps({'rows':count}));return 0
if __name__=='__main__':raise SystemExit(main())
