from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
    if str(candidate) not in sys.path:
        sys.path.insert(0,str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key,parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score,load_row_scores


def logit(value: float) -> float:
    value=min(1.0-1e-6,max(1e-6,float(value)))
    return math.log(value/(1.0-value))


def sigmoid(value: float) -> float:
    if value>=0.0:
        z=math.exp(-value)
        return 1.0/(1.0+z)
    z=math.exp(value)
    return z/(1.0+z)


def bounded_aux(base: float,residual: float,mode: str,cap: float,weight: float) -> float:
    delta=logit(residual)-logit(base)
    if mode=='boost-only':
        delta=max(0.0,delta)
    elif mode=='suppress-only':
        delta=min(0.0,delta)
    elif mode!='symmetric':
        raise ValueError(f'unknown residual mode: {mode}')
    delta=max(-cap,min(cap,delta))
    return sigmoid(logit(base)+weight*delta)


def evaluate_config(data: dict[str,Any],base: dict,residual: dict,mode: str,cap: float,weight: float,alpha: float,tvd_root: Path,out_dir: Path) -> dict[str,Any]:
    output={}
    changed=0
    for image_id,item in data.items():
        detections=[]
        for index,row in enumerate(item.get('detections') or []):
            key=image_key(str(image_id),index)
            raw=float(row.get('score',0.0))
            base_score=float(base.get(key,raw))
            residual_score=float(residual.get(key,base_score))
            aux=bounded_aux(base_score,residual_score,mode,cap,weight)
            changed+=int(abs(aux-base_score)>1e-12)
            new_row=dict(row)
            new_row['score']=fuse_score(raw,aux,alpha,'geom-mix')
            detections.append(new_row)
        output[image_id]={'labels':item.get('labels',[]),'detections':detections}
    return {'changed_rows':changed,**evaluate_data(output,tvd_root,out_dir)}


def main() -> int:
    parser=argparse.ArgumentParser(description='Bounded residual fusion for stable immediate and persistent Action Chunk banks.')
    parser.add_argument('--tvd-root',type=Path,required=True)
    parser.add_argument('--predictionsgt-pkl',type=Path,required=True)
    parser.add_argument('--base-jsonl',type=Path,required=True)
    parser.add_argument('--base-field',required=True)
    parser.add_argument('--residual-jsonl',type=Path,required=True)
    parser.add_argument('--residual-field',required=True)
    parser.add_argument('--modes',default='boost-only,symmetric')
    parser.add_argument('--caps',default='.25,.5,1')
    parser.add_argument('--weights',default='.25,.5')
    parser.add_argument('--alphas',default='.2,.3,.4')
    parser.add_argument('--fixed-config-json',type=Path)
    parser.add_argument('--out-json',type=Path,required=True)
    args=parser.parse_args()
    data=load_predictionsgt(args.predictionsgt_pkl)
    base,base_summary=load_row_scores(args.base_jsonl,args.base_field,1)
    residual,residual_summary=load_row_scores(args.residual_jsonl,args.residual_field,1)
    if args.fixed_config_json:
        best=json.loads(args.fixed_config_json.read_text(encoding='utf8'))['best']
        configs=[(str(best['residual_mode']),float(best['cap']),float(best['weight']),float(best['alpha']))]
    else:
        modes=[value.strip() for value in args.modes.split(',') if value.strip()]
        configs=[(mode,cap,weight,alpha) for mode in modes for cap in parse_csv_floats(args.caps) for weight in parse_csv_floats(args.weights) for alpha in parse_csv_floats(args.alphas)]
    rows=[]
    for mode,cap,weight,alpha in configs:
        metrics=evaluate_config(data,base,residual,mode,cap,weight,alpha,args.tvd_root,args.out_json.parent)
        row={'residual_mode':mode,'cap':cap,'weight':weight,'alpha':alpha,**metrics}
        rows.append(row)
        print(json.dumps(row),flush=True)
    best=max(rows,key=lambda row:float(row['map50']))
    summary={'base':base_summary,'residual':residual_summary,'best':best,'rows':rows}
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(summary,indent=2),encoding='utf8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
