from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
from typing import Any

import numpy as np

REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
    if str(candidate) not in sys.path:
        sys.path.insert(0,str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data
from tools.sweep_tvd_predictionsgt_score_fusion import clone_with_fused_scores

TOKEN_DIM=12
FIELDS=('predicted_iou','track_quality','age_support','short_compat','long_compat','causal_continuity','causal_stability')


def finite(value: Any) -> float:
    try:
        result=float(value)
    except (TypeError,ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def token_stats(values: Any) -> tuple[float,float,float]:
    array=np.asarray(values or [],np.float32)
    if not len(array) or len(array)%TOKEN_DIM:
        return 0.0,0.0,0.0
    array=array.reshape(-1,TOKEN_DIM)
    valid=array[:,0]>.5
    if not valid.any():
        return 0.0,0.0,0.0
    ages=np.arange(len(array),dtype=np.float32)
    weights=np.exp(-2.0*ages/max(1,len(array)-1))[valid]
    weights/=weights.sum()
    return float((array[valid,9]*weights).sum()),float((array[valid,11]*weights).sum()),float(valid.mean())


def geometric(values: tuple[float,...]) -> float:
    clipped=[max(1e-6,min(1.0,value)) for value in values]
    return math.exp(sum(math.log(value) for value in clipped)/len(clipped))


def load_maps(path: Path) -> dict[str,dict]:
    maps={field:{} for field in FIELDS}
    with path.open(encoding='utf-8-sig') as source:
        for line in source:
            if not line.strip():
                continue
            item=json.loads(line)
            meta=item.get('meta') or {}
            for row in item.get('rows') or []:
                seq=str(row.get('seq') or meta.get('seq') or '')
                frame_id=row.get('frame_id')
                prediction_index=row.get('prediction_index')
                if not seq or frame_id is None or prediction_index is None:
                    continue
                key=(seq,int(frame_id),int(prediction_index))
                predicted_iou=max(0.0,min(1.0,finite(row.get('action_chunk_bank_predicted_iou'))))
                quality=max(0.0,min(1.0,finite(row.get('action_chunk_bank_track_quality'))))
                stability=max(0.0,min(1.0,finite(row.get('action_chunk_bank_motion_stability'))))
                age=max(0.0,finite(row.get('action_chunk_bank_track_age_seconds')))
                age_support=1.0-math.exp(-age)
                short_iou,short_compat,short_valid=token_stats(row.get('action_chunk_bank_short_tokens'))
                long_iou,long_compat,long_valid=token_stats(row.get('action_chunk_bank_long_tokens'))
                continuity=geometric((predicted_iou,quality,max(age_support,.01),max(short_compat,.01)))
                stable=geometric((max(short_compat,.01),max(long_compat,.01),max(stability,.01),max(.5*(short_valid+long_valid),.01)))
                values={'predicted_iou':predicted_iou,'track_quality':quality,'age_support':age_support,'short_compat':short_compat,'long_compat':long_compat,'causal_continuity':continuity,'causal_stability':stable}
                for field,value in values.items():
                    maps[field][key]=value
    return maps


def main() -> int:
    parser=argparse.ArgumentParser(description='Label-free strictly causal heuristic sweep over Action Chunk persistence evidence.')
    parser.add_argument('--val-pkl',type=Path,required=True)
    parser.add_argument('--test-pkl',type=Path,required=True)
    parser.add_argument('--val-jsonl',type=Path,required=True)
    parser.add_argument('--test-jsonl',type=Path,required=True)
    parser.add_argument('--tvd-root',type=Path,required=True)
    parser.add_argument('--out-json',type=Path,required=True)
    args=parser.parse_args()
    validation=load_predictionsgt(args.val_pkl)
    validation_maps=load_maps(args.val_jsonl)
    rows=[]
    for field in FIELDS:
        for mode in ('geom-mix','fp-suppress','tp-boost'):
            for alpha in (.01,.02,.04,.06,.08,.10,.14,.20,.30):
                metrics=evaluate_data(clone_with_fused_scores(validation,validation_maps[field],mode,alpha,'keep'),args.tvd_root,args.out_json.parent)
                row={'field':field,'mode':mode,'alpha':alpha,**metrics}
                rows.append(row)
                print(json.dumps(row),flush=True)
    best=max(rows,key=lambda row:float(row['map50']))
    test=load_predictionsgt(args.test_pkl)
    test_maps=load_maps(args.test_jsonl)
    fixed=evaluate_data(clone_with_fused_scores(test,test_maps[best['field']],best['mode'],best['alpha'],'keep'),args.tvd_root,args.out_json.parent)
    result={'protocol':'strict causal label-free Action Chunk continuity heuristic; validation selection; fixed test','validation_selection':best,'test_fixed':fixed,'target_map50':.97,'target_met':fixed['map50']>=.97,'top':sorted(rows,key=lambda row:float(row['map50']),reverse=True)[:20]}
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
