from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any

import numpy as np

REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
    if str(candidate) not in sys.path:
        sys.path.insert(0,str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key,parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores


def rank_percentile(values: np.ndarray) -> np.ndarray:
    if len(values)<=1:
        return np.ones_like(values,dtype=np.float32)
    order=np.argsort(np.argsort(values,kind='stable'),kind='stable')
    return order.astype(np.float32)/float(len(values)-1)


def transplant_frame(rows: list[dict[str,Any]],sequence: str,frame_id: int,score_map: dict,weight: float,band: float) -> tuple[list[dict[str,Any]],int]:
    if len(rows)<=1:
        return rows,0
    raw=np.asarray([float(row.get('score',0.0)) for row in rows],np.float64)
    auxiliary=np.asarray([float(score_map.get((sequence,frame_id,index),raw[index])) for index in range(len(rows))],np.float64)
    threshold=raw.max()-band if band<1.0 else -np.inf
    eligible=np.flatnonzero(raw>=threshold)
    if len(eligible)<=1:
        return rows,0
    raw_rank=rank_percentile(raw[eligible])
    aux_rank=rank_percentile(auxiliary[eligible])
    composite=(1.0-weight)*raw_rank+weight*aux_rank
    target_order=eligible[np.argsort(composite,kind='stable')[::-1]]
    sorted_scores=np.sort(raw[eligible])[::-1]
    assigned=raw.copy()
    assigned[target_order]=sorted_scores
    output=[]
    changed=0
    for index,row in enumerate(rows):
        new_row=dict(row)
        new_row['score']=float(assigned[index])
        changed+=int(abs(assigned[index]-raw[index])>1e-15)
        output.append(new_row)
    return output,changed


def evaluate_config(data,score_map,weight,band,tvd_root,out_dir):
    output={}
    changed=0
    preservation_error=0.0
    for image_id,item in data.items():
        sequence,frame_id,_=image_key(str(image_id),0)
        rows=list(item.get('detections') or [])
        rescored,count=transplant_frame(rows,sequence,frame_id,score_map,weight,band)
        before=sorted(float(row.get('score',0.0)) for row in rows)
        after=sorted(float(row.get('score',0.0)) for row in rescored)
        if before:
            preservation_error=max(preservation_error,max(abs(left-right) for left,right in zip(before,after)))
        changed+=count
        output[image_id]={'labels':item.get('labels',[]),'detections':rescored}
    return {'changed_rows':changed,'max_frame_score_multiset_error':preservation_error,**evaluate_data(output,tvd_root,out_dir)}


def main() -> int:
    parser=argparse.ArgumentParser(description='Domain-robust Action Chunk frame-rank transplant preserving each frame score multiset.')
    parser.add_argument('--tvd-root',type=Path,required=True)
    parser.add_argument('--predictionsgt-pkl',type=Path,required=True)
    parser.add_argument('--sources-json',type=Path,required=True)
    parser.add_argument('--weights',default='.25,.5,.75,1')
    parser.add_argument('--bands',default='.03,.05,.1,.2,1')
    parser.add_argument('--fixed-config-json',type=Path)
    parser.add_argument('--out-json',type=Path,required=True)
    args=parser.parse_args()
    sources=json.loads(args.sources_json.read_text(encoding='utf8'))
    data=load_predictionsgt(args.predictionsgt_pkl)
    if args.fixed_config_json:
        selected=json.loads(args.fixed_config_json.read_text(encoding='utf8'))['best']
        configs=[(str(selected['source']),float(selected['weight']),float(selected['band']))]
    else:
        configs=[(name,weight,band) for name in sources for weight in parse_csv_floats(args.weights) for band in parse_csv_floats(args.bands)]
    maps={}
    rows=[]
    for source,weight,band in configs:
        if source not in maps:
            item=sources[source]
            maps[source],_=load_row_scores(Path(item['path']),str(item['field']),1)
        metrics=evaluate_config(data,maps[source],weight,band,args.tvd_root,args.out_json.parent)
        row={'source':source,'weight':weight,'band':band,**metrics}
        rows.append(row)
        print(json.dumps(row),flush=True)
    best=max(rows,key=lambda row:float(row['map50']))
    summary={'best':best,'rows':rows}
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(summary,indent=2),encoding='utf8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
