from __future__ import annotations
import argparse,gc,json,math,sys,threading,time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
    if str(candidate) not in sys.path:
        sys.path.insert(0,str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_bank_motion_token_listwise import greedy_match_qualities,write_score_jsonl
from tools.train_action_chunk_bidir_full import finite,load_aux,percentile
from tools.train_action_chunk_causal_memory import load_past_neighbor
from tools.train_action_chunk_neighbor_full import HEIGHT,WIDTH

HEARTBEAT={'stage':'startup'}


def heartbeat() -> None:
    while True:
        print(json.dumps({'kind':'action_chunk_train_only_heartbeat','stage':HEARTBEAT['stage'],'updated':datetime.now().astimezone().isoformat()}),flush=True)
        time.sleep(60)


def arrays(pred,forward,neighbor,labels):
    chunks=[]
    qualities=[]
    groups=[]
    locations=[]
    cursor=0
    for image_id,item in pred.items():
        seq,frame_id,_=image_key(str(image_id),0)
        detections=list(item.get('detections') or [])
        raw=np.asarray([finite(row.get('score')) for row in detections],np.float32)
        rank=percentile(raw)
        gap=(raw.max()-raw) if len(raw) else raw
        forward_values=forward.get_many(seq,frame_id,len(detections))
        neighbor_values=neighbor.get_many(seq,frame_id,len(detections))
        rows=[]
        boxes=[]
        gt=np.asarray([row.get('bbox') for row in item.get('labels',[]) if isinstance(row.get('bbox'),list) and len(row['bbox'])==4],np.float32)
        if not gt.size:
            gt=np.zeros((0,4),np.float32)
        for index,row in enumerate(detections):
            box=row.get('bbox')
            if not isinstance(box,list) or len(box)!=4:
                continue
            x1,y1,x2,y2=[finite(value) for value in box]
            width=max(1e-3,x2-x1)
            height=max(1e-3,y2-y1)
            center_x=.5*(x1+x2)
            center_y=.5*(y1+y2)
            clipped=np.clip(raw[index],1e-6,1-1e-6)
            border=max(0.0,min(center_x,WIDTH-center_x,center_y,HEIGHT-center_y))/min(WIDTH,HEIGHT)
            prefix=np.asarray([raw[index],math.log(clipped/(1-clipped)),rank[index],gap[index],center_x/WIDTH,center_y/HEIGHT,width/WIDTH,height/HEIGHT,width*height/(WIDTH*HEIGHT),math.log(width/height),border,math.log1p(len(detections))/6.0],np.float32)
            rows.append(np.concatenate((prefix,forward_values[index],neighbor_values[index])))
            boxes.append([x1,y1,x2,y2])
            locations.append((str(image_id),index))
        if rows:
            chunks.append(np.stack(rows).astype(np.float32))
            qualities.append(greedy_match_qualities(boxes,gt) if labels else np.zeros(len(rows),np.float32))
            groups.append((cursor,cursor+len(rows)))
            cursor+=len(rows)
    return np.concatenate(chunks),np.concatenate(qualities),groups,locations


def hard_rows(features,quality,groups,margin=.3,max_negative=24):
    keep=[]
    for start,stop in groups:
        positive=np.flatnonzero(quality[start:stop]>=.5)
        negative=np.flatnonzero(quality[start:stop]<.5)
        raw=features[start:stop,0]
        if not len(negative):
            keep.extend((start+positive).tolist())
        elif not len(positive):
            keep.extend((start+negative[np.argsort(raw[negative])[::-1][:6]]).tolist())
        else:
            hard=negative[raw[negative]>=raw[positive].max()-margin]
            hard=hard[np.argsort(raw[hard])[::-1][:max_negative]] if len(hard) else negative[np.argsort(raw[negative])[::-1][:4]]
            keep.extend((start+positive).tolist())
            keep.extend((start+hard).tolist())
    return np.asarray(sorted(set(keep)),np.int64)


def fit(features,quality,groups,seed):
    keep=hard_rows(features,quality,groups)
    binary=(quality[keep]>=.5).astype(np.int32)
    positive=max(1,int(binary.sum()))
    negative=len(binary)-positive
    model=xgb.XGBClassifier(n_estimators=950,max_depth=7,learning_rate=.03,min_child_weight=6,subsample=.82,colsample_bytree=.82,reg_lambda=12,reg_alpha=.18,gamma=.03,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,negative/positive),n_jobs=8,random_state=seed)
    model.fit(features[keep],binary,verbose=False)
    return model,len(keep),positive


def main() -> int:
    parser=argparse.ArgumentParser(description='Strict causal Action Chunk train-only ensemble; validation is never added to fitting data.')
    for split in ('train','val','test'):
        parser.add_argument(f'--{split}-pkl',type=Path,required=True)
        parser.add_argument(f'--{split}-forward',type=Path,required=True)
        parser.add_argument(f'--{split}-neighbor',type=Path,required=True)
    for name in ('out-val-scores','out-test-scores','out-model-dir','out-summary'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seeds',default='2026,2027,2028,2029')
    parser.add_argument('--score-field',default='action_chunk_causal_train_only_score')
    args=parser.parse_args()
    threading.Thread(target=heartbeat,daemon=True).start()

    def load(split,labels):
        HEARTBEAT['stage']=f'load_{split}_past_only_features'
        predictions=load_predictionsgt(getattr(args,f'{split}_pkl'))
        forward=load_aux(getattr(args,f'{split}_forward'))
        neighbor,names=load_past_neighbor(getattr(args,f'{split}_neighbor'))
        result=arrays(predictions,forward,neighbor,labels)
        del predictions,forward,neighbor
        gc.collect()
        return result,names

    (train_x,train_y,train_groups,_),names=load('train',True)
    (val_x,_,_,val_locations),val_names=load('val',False)
    (test_x,_,_,test_locations),test_names=load('test',False)
    assert names==val_names==test_names
    seeds=[int(value) for value in args.seeds.split(',') if value.strip()]
    val_predictions=[]
    test_predictions=[]
    models=[]
    args.out_model_dir.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        HEARTBEAT['stage']=f'train_seed_{seed}'
        model,count,positive=fit(train_x,train_y,train_groups,seed)
        val_predictions.append(model.predict_proba(val_x)[:,1])
        test_predictions.append(model.predict_proba(test_x)[:,1])
        model_path=args.out_model_dir/f'action_chunk_causal_train_only_seed_{seed}.ubj'
        model.save_model(model_path)
        record={'seed':seed,'hard_rows':count,'positive_rows':positive,'model':str(model_path)}
        models.append(record)
        print(json.dumps({'kind':'action_chunk_causal_train_only_model',**record}),flush=True)
        del model
        gc.collect()
    HEARTBEAT['stage']='write_scores'
    write_score_jsonl(args.out_val_scores,np.mean(np.stack(val_predictions),axis=0).astype(np.float32),val_locations,args.score_field)
    write_score_jsonl(args.out_test_scores,np.mean(np.stack(test_predictions),axis=0).astype(np.float32),test_locations,args.score_field)
    summary={'model':'strict causal train-only Action Chunk ensemble','inference_features':'raw detector + forward 1s/3s bank + past-only neighbors','training_boundary':'Clip1-36 only; validation Clip37-40 never included in model fitting','features':train_x.shape[1],'neighbor_features':names,'models':models}
    args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
