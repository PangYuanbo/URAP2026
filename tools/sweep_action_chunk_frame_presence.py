from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict,deque
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
    if str(candidate) not in sys.path:
        sys.path.insert(0,str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import evaluate_data,image_key,parse_csv_floats
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score,load_row_scores


def causal_presence(data,threshold,window_seconds,min_fraction,fps_map):
    sequences=defaultdict(list)
    for image_id,item in data.items():
        seq,frame_id,_=image_key(str(image_id),0)
        scores=[float(row.get('score',0.0)) for row in item.get('detections') or []]
        signal=max(scores,default=0.0)>=threshold
        sequences[seq].append((frame_id,str(image_id),signal))
    presence={}
    for seq,items in sequences.items():
        items.sort()
        fps=float(fps_map.get(seq,30.0))
        frame_window=max(1,int(round(window_seconds*fps)))
        history=deque()
        active=0
        for frame_id,image_id,signal in items:
            history.append((frame_id,signal))
            active+=int(signal)
            while history and frame_id-history[0][0]>=frame_window:
                active-=int(history.popleft()[1])
            fraction=active/max(1,len(history))
            presence[image_id]=min(1.0,fraction/max(1e-6,min_fraction))
    return presence


def evaluate_config(data,base,presence,alpha,tvd_root,out_dir):
    output={}
    suppressed=0
    for image_id,item in data.items():
        frame_presence=float(presence.get(str(image_id),1.0))
        detections=[]
        for index,row in enumerate(item.get('detections') or []):
            seq,frame_id,_=image_key(str(image_id),index)
            raw=float(row.get('score',0.0))
            base_score=float(base.get((seq,frame_id,index),raw))
            auxiliary=base_score*(.15+.85*frame_presence)
            new_row=dict(row)
            new_row['score']=fuse_score(raw,auxiliary,alpha,'geom-mix')
            suppressed+=int(frame_presence<1.0)
            detections.append(new_row)
        output[image_id]={'labels':item.get('labels',[]),'detections':detections}
    return {'suppressed_rows':suppressed,**evaluate_data(output,tvd_root,out_dir)}


def main() -> int:
    parser=argparse.ArgumentParser(description='Strict causal true-time frame presence Action Bank.')
    parser.add_argument('--tvd-root',type=Path,required=True)
    parser.add_argument('--predictionsgt-pkl',type=Path,required=True)
    parser.add_argument('--base-jsonl',type=Path,required=True)
    parser.add_argument('--base-field',required=True)
    parser.add_argument('--sequence-fps-json',type=Path,required=True)
    parser.add_argument('--thresholds',default='.3,.4,.5,.6,.7')
    parser.add_argument('--windows',default='.25,1,3')
    parser.add_argument('--fractions',default='.1,.25,.5,.75')
    parser.add_argument('--alphas',default='.02,.04,.06,.08,.1,.14,.2,.3')
    parser.add_argument('--fixed-config-json',type=Path)
    parser.add_argument('--out-json',type=Path,required=True)
    args=parser.parse_args()
    data=load_predictionsgt(args.predictionsgt_pkl)
    base,_=load_row_scores(args.base_jsonl,args.base_field,1)
    fps_map=json.loads(args.sequence_fps_json.read_text(encoding='utf8'))
    if args.fixed_config_json:
        best=json.loads(args.fixed_config_json.read_text(encoding='utf8'))['best']
        configs=[(float(best['threshold']),float(best['window_seconds']),float(best['min_fraction']),float(best['alpha']))]
    else:
        configs=[(threshold,window,fraction,alpha) for threshold in parse_csv_floats(args.thresholds) for window in parse_csv_floats(args.windows) for fraction in parse_csv_floats(args.fractions) for alpha in parse_csv_floats(args.alphas)]
    rows=[]
    cache={}
    for threshold,window,fraction,alpha in configs:
        key=(threshold,window,fraction)
        frame_presence=cache.setdefault(key,causal_presence(data,threshold,window,fraction,fps_map))
        metrics=evaluate_config(data,base,frame_presence,alpha,args.tvd_root,args.out_json.parent)
        row={'threshold':threshold,'window_seconds':window,'min_fraction':fraction,'alpha':alpha,**metrics}
        rows.append(row)
        print(json.dumps(row),flush=True)
    best=max(rows,key=lambda row:float(row['map50']))
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps({'best':best,'rows':rows},indent=2),encoding='utf8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
