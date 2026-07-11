from __future__ import annotations
import argparse,gc,json,math,sys,threading,time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

REPO=Path(__file__).resolve().parents[1]
for candidate in (REPO,REPO/'tools'):
    if str(candidate) not in sys.path:
        sys.path.insert(0,str(candidate))

from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.train_action_bank_motion_token_listwise import greedy_match_qualities,write_score_jsonl
from tools.train_action_chunk_bidir_full import CompactAux,candidate_key,finite,load_aux,percentile
from tools.train_action_chunk_neighbor_full import HEIGHT,WIDTH

HEARTBEAT={'stage':'startup'}

def heartbeat() -> None:
    while True:
        print(json.dumps({'kind':'action_chunk_causal_memory_heartbeat','stage':HEARTBEAT['stage'],'updated':datetime.now().astimezone().isoformat()}),flush=True)
        time.sleep(60)


def load_past_neighbor(path: Path) -> tuple[CompactAux,list[str]]:
    key_chunks=[]
    value_chunks=[]
    keys=[]
    values=[]
    fields=None
    with path.open(encoding='utf8') as source:
        for line in source:
            item=json.loads(line)
            meta=item.get('meta') or {}
            for row in item.get('rows') or []:
                if fields is None:
                    fields=sorted(name for name in row if name.startswith('action_chunk_neighbor_past_'))
                seq=str(row.get('seq') or meta.get('seq') or '')
                frame_id=row.get('frame_id')
                prediction_index=row.get('prediction_index')
                if not seq or frame_id is None or prediction_index is None:
                    continue
                keys.append(candidate_key(seq,int(frame_id),int(prediction_index)))
                values.append([finite(row.get(name)) for name in fields])
                if len(keys)>=100000:
                    key_chunks.append(np.asarray(keys,np.uint64))
                    value_chunks.append(np.asarray(values,np.float16))
                    keys=[]
                    values=[]
    if keys:
        key_chunks.append(np.asarray(keys,np.uint64))
        value_chunks.append(np.asarray(values,np.float16))
    if not key_chunks or fields is None:
        raise ValueError(f'no past-neighbor rows in {path}')
    return CompactAux(np.concatenate(key_chunks),np.concatenate(value_chunks)),fields


def future_strength(backward_values: np.ndarray) -> np.ndarray:
    if not len(backward_values):
        return np.zeros((0,),np.float32)
    score=(.30*backward_values[:,0]+.20*backward_values[:,1]+.15*backward_values[:,2]+.10*backward_values[:,3]+.10*backward_values[:,4]+.15*backward_values[:,8])
    return np.clip(score,0.0,1.0).astype(np.float32)


def dataset_arrays(pred: dict[str,Any],immediate: CompactAux,persistent: CompactAux,neighbor: CompactAux,labels: bool,backward: CompactAux|None=None):
    chunks=[]
    qualities=[]
    futures=[]
    groups=[]
    locations=[]
    sequences=[]
    cursor=0
    for image_id,item in pred.items():
        seq,frame_id,_=image_key(str(image_id),0)
        detections=list(item.get('detections') or [])
        raw=np.asarray([finite(row.get('score')) for row in detections],np.float32)
        rank=percentile(raw)
        gap=(raw.max()-raw) if len(raw) else raw
        immediate_values=immediate.get_many(seq,frame_id,len(detections))
        persistent_values=persistent.get_many(seq,frame_id,len(detections))
        neighbor_values=neighbor.get_many(seq,frame_id,len(detections))
        backward_values=backward.get_many(seq,frame_id,len(detections)) if backward is not None else np.zeros_like(immediate_values)
        future=future_strength(backward_values)
        rows=[]
        boxes=[]
        row_future=[]
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
            immediate_row=immediate_values[index]
            persistent_delta=np.clip(persistent_values[index]-immediate_row,-8.0,8.0)
            persistence_flags=np.asarray([persistent_values[index,6],persistent_values[index,9],persistent_values[index,0]-immediate_row[0],persistent_values[index,1]-immediate_row[1]],np.float32)
            rows.append(np.concatenate((prefix,immediate_row,persistent_delta,np.abs(persistent_delta),persistence_flags,neighbor_values[index])))
            boxes.append([x1,y1,x2,y2])
            row_future.append(future[index] if len(future) else 0.0)
            locations.append((str(image_id),index))
            sequences.append(seq)
        if rows:
            chunks.append(np.stack(rows).astype(np.float32))
            qualities.append(greedy_match_qualities(boxes,gt) if labels else np.zeros(len(rows),np.float32))
            futures.append(np.asarray(row_future,np.float32))
            groups.append((cursor,cursor+len(rows)))
            cursor+=len(rows)
    return np.concatenate(chunks),np.concatenate(qualities),np.concatenate(futures),groups,locations,np.asarray(sequences)


def hard_rows(features: np.ndarray,quality: np.ndarray,groups: list[tuple[int,int]],margin: float=.30,max_negative: int=24) -> np.ndarray:
    keep=[]
    for start,stop in groups:
        positive=np.flatnonzero(quality[start:stop]>=.5)
        negative=np.flatnonzero(quality[start:stop]<.5)
        if not len(negative):
            keep.extend((start+positive).tolist())
            continue
        raw=features[start:stop,0]
        if not len(positive):
            keep.extend((start+negative[np.argsort(raw[negative])[::-1][:6]]).tolist())
            continue
        hard=negative[raw[negative]>=raw[positive].max()-margin]
        hard=hard[np.argsort(raw[hard])[::-1][:max_negative]] if len(hard) else negative[np.argsort(raw[negative])[::-1][:4]]
        keep.extend((start+positive).tolist())
        keep.extend((start+hard).tolist())
    return np.asarray(sorted(set(keep)),np.int64)


def fit(features: np.ndarray,quality: np.ndarray,future: np.ndarray,groups: list[tuple[int,int]]):
    keep=hard_rows(features,quality,groups)
    binary=(quality[keep]>=.5).astype(np.int32)
    strength=np.clip(future[keep],0.0,1.0)
    sample_weight=np.where(binary>0,1.0+.75*strength,1.0+1.25*strength).astype(np.float32)
    positive=max(1,int(binary.sum()))
    negative=len(binary)-positive
    model=xgb.XGBClassifier(n_estimators=1000,max_depth=7,learning_rate=.03,min_child_weight=5,subsample=.9,colsample_bytree=.88,reg_lambda=10,reg_alpha=.12,gamma=.025,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',device='cuda',max_bin=256,scale_pos_weight=min(12,negative/positive),n_jobs=8,random_state=2026)
    model.fit(features[keep],binary,sample_weight=sample_weight,verbose=False)
    return model,len(keep),positive,float(sample_weight.mean())


def main() -> int:
    parser=argparse.ArgumentParser(description='Strictly causal Action Chunk Bank: past 1s/3s inputs, future evidence only as training supervision.')
    for split in ('train','val'):
        for suffix in ('pkl','immediate','persistent','backward','neighbor'):
            parser.add_argument(f'--{split}-{suffix}',type=Path,required=True)
    for suffix in ('pkl','immediate','persistent','neighbor'):
        parser.add_argument(f'--test-{suffix}',type=Path,required=True)
    for name in ('out-val-scores','out-test-scores','out-model-dir','out-summary'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--score-field',default='action_chunk_causal_memory_score')
    args=parser.parse_args()
    threading.Thread(target=heartbeat,daemon=True).start()

    def load_labeled(split: str):
        immediate=load_aux(getattr(args,f'{split}_immediate'))
        persistent=load_aux(getattr(args,f'{split}_persistent'))
        backward=load_aux(getattr(args,f'{split}_backward'))
        neighbor,names=load_past_neighbor(getattr(args,f'{split}_neighbor'))
        pred=load_predictionsgt(getattr(args,f'{split}_pkl'))
        result=dataset_arrays(pred,immediate,persistent,neighbor,True,backward)
        del pred,immediate,persistent,backward,neighbor
        gc.collect()
        return result,names

    def load_test():
        immediate=load_aux(args.test_immediate)
        persistent=load_aux(args.test_persistent)
        neighbor,names=load_past_neighbor(args.test_neighbor)
        pred=load_predictionsgt(args.test_pkl)
        result=dataset_arrays(pred,immediate,persistent,neighbor,False,None)
        del pred,immediate,persistent,neighbor
        gc.collect()
        return result,names

    HEARTBEAT['stage']='load_train_features'
    (train_x,train_y,train_future,train_groups,_,_),names=load_labeled('train')
    HEARTBEAT['stage']='load_validation_features'
    (val_x,val_y,val_future,val_groups,val_locations,val_sequences),val_names=load_labeled('val')
    HEARTBEAT['stage']='load_test_past_only_features'
    (test_x,_,_,_,test_locations,_),test_names=load_test()
    assert names==val_names==test_names
    output_oof=np.zeros(len(val_x),np.float32)
    test_predictions=[]
    models=[]
    args.out_model_dir.mkdir(parents=True,exist_ok=True)
    for held in sorted(set(val_sequences)):
        HEARTBEAT['stage']=f'train_without_{held}'
        feature_parts=[train_x]
        label_parts=[train_y]
        future_parts=[train_future]
        groups=list(train_groups)
        cursor=len(train_x)
        for start,stop in val_groups:
            if val_sequences[start]==held:
                continue
            feature_parts.append(val_x[start:stop])
            label_parts.append(val_y[start:stop])
            future_parts.append(val_future[start:stop])
            groups.append((cursor,cursor+stop-start))
            cursor+=stop-start
        fit_x=np.concatenate(feature_parts)
        fit_y=np.concatenate(label_parts)
        fit_future=np.concatenate(future_parts)
        model,count,positive,mean_weight=fit(fit_x,fit_y,fit_future,groups)
        mask=val_sequences==held
        output_oof[mask]=model.predict_proba(val_x[mask])[:,1]
        test_predictions.append(model.predict_proba(test_x)[:,1])
        model_path=args.out_model_dir/f'action_chunk_causal_memory_without_{held}.ubj'
        model.save_model(model_path)
        record={'excluded_validation_video':held,'hard_rows':count,'positive_rows':positive,'mean_future_supervision_weight':mean_weight,'model':str(model_path)}
        models.append(record)
        print(json.dumps({'kind':'action_chunk_causal_memory_model',**record}),flush=True)
        del fit_x,fit_y,fit_future,model
        gc.collect()
    HEARTBEAT['stage']='write_scores'
    write_score_jsonl(args.out_val_scores,output_oof,val_locations,args.score_field)
    write_score_jsonl(args.out_test_scores,np.mean(np.stack(test_predictions),axis=0).astype(np.float32),test_locations,args.score_field)
    summary={'model':'strict causal pure Action Chunk 1s/3s dual-memory bank','inference_features':'raw detector + immediate past bank + persistent past delta + past-only neighbors','future_information':'training sample weights only; no test backward/future input argument or file read','neighbor_features':names,'features':train_x.shape[1],'models':models}
    args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
