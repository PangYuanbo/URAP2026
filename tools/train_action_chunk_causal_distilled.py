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
from tools.sweep_tvd_predictionsgt_score_fusion import load_row_scores
from tools.train_action_bank_motion_token_listwise import write_score_jsonl
from tools.train_action_chunk_bidir_full import finite,load_aux,percentile
from tools.train_action_chunk_causal_memory import load_past_neighbor
from tools.train_action_chunk_neighbor_full import HEIGHT,WIDTH,dataset_arrays as teacher_arrays,load_neighbor

HEARTBEAT={'stage':'startup'}


def heartbeat() -> None:
    while True:
        print(json.dumps({'kind':'action_chunk_distill_heartbeat','stage':HEARTBEAT['stage'],'updated':datetime.now().astimezone().isoformat()}),flush=True)
        time.sleep(60)


def causal_arrays(pred,forward,neighbor,labels):
    chunks=[]
    qualities=[]
    groups=[]
    locations=[]
    sequences=[]
    cursor=0
    from tools.train_action_bank_motion_token_listwise import greedy_match_qualities
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
            sequences.append(seq)
        if rows:
            chunks.append(np.stack(rows).astype(np.float32))
            qualities.append(greedy_match_qualities(boxes,gt) if labels else np.zeros(len(rows),np.float32))
            groups.append((cursor,cursor+len(rows)))
            cursor+=len(rows)
    return np.concatenate(chunks),np.concatenate(qualities),groups,locations,np.asarray(sequences)


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


def teacher_train_scores(predictions,forward_path,backward_path,neighbor_path,model_dir):
    HEARTBEAT['stage']='build_offline_teacher_train_features'
    forward=load_aux(forward_path)
    backward=load_aux(backward_path)
    neighbor,_=load_neighbor(neighbor_path)
    features,_,_,locations,_=teacher_arrays(predictions,forward,backward,neighbor,False)
    del forward,backward,neighbor
    gc.collect()
    HEARTBEAT['stage']='infer_offline_teacher_train_scores'
    scores=[]
    for model_path in sorted(model_dir.glob('*.ubj')):
        model=xgb.XGBClassifier()
        model.load_model(model_path)
        scores.append(model.predict_proba(features)[:,1])
        del model
        gc.collect()
    if not scores:
        raise ValueError(f'no teacher models in {model_dir}')
    result=np.mean(np.stack(scores),axis=0).astype(np.float32)
    del features,scores
    gc.collect()
    return result,locations


def align_score_map(path,field,locations):
    score_map,_=load_row_scores(path,field,1)
    output=np.zeros(len(locations),np.float32)
    for row_index,(image_id,prediction_index) in enumerate(locations):
        output[row_index]=float(score_map.get(image_key(str(image_id),prediction_index),0.0))
    return output


def fit(features,quality,teacher,groups,teacher_weight):
    keep=hard_rows(features,quality,groups)
    binary=(quality[keep]>=.5).astype(np.float32)
    soft=np.clip((1.0-teacher_weight)*binary+teacher_weight*teacher[keep],0.0,1.0)
    positive=max(1,int(binary.sum()))
    negative=len(binary)-positive
    sample_weight=np.where(binary>.5,min(12.0,negative/positive),1.0).astype(np.float32)
    model=xgb.XGBRegressor(n_estimators=1100,max_depth=7,learning_rate=.028,min_child_weight=5,subsample=.9,colsample_bytree=.9,reg_lambda=10,reg_alpha=.12,gamma=.02,objective='reg:logistic',eval_metric='rmse',tree_method='hist',device='cuda',max_bin=256,n_jobs=8,random_state=2026)
    model.fit(features[keep],soft,sample_weight=sample_weight,verbose=False)
    return model,len(keep),positive,float(soft.mean())


def main() -> int:
    parser=argparse.ArgumentParser(description='Distill offline bidirectional Action Chunk teacher into a strictly causal past-only student.')
    for split in ('train','val'):
        for suffix in ('pkl','forward','backward','neighbor'):
            parser.add_argument(f'--{split}-{suffix}',type=Path,required=True)
    for suffix in ('pkl','forward','neighbor'):
        parser.add_argument(f'--test-{suffix}',type=Path,required=True)
    parser.add_argument('--teacher-model-dir',type=Path,required=True)
    parser.add_argument('--val-teacher-scores',type=Path,required=True)
    parser.add_argument('--val-teacher-field',default='action_chunk_neighbor_score')
    parser.add_argument('--teacher-weight',type=float,default=.4)
    for name in ('out-val-scores','out-test-scores','out-model-dir','out-summary'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--score-field',default='action_chunk_causal_distilled_score')
    args=parser.parse_args()
    threading.Thread(target=heartbeat,daemon=True).start()

    HEARTBEAT['stage']='load_train_predictions'
    train_predictions=load_predictionsgt(args.train_pkl)
    teacher_train,teacher_locations=teacher_train_scores(train_predictions,args.train_forward,args.train_backward,args.train_neighbor,args.teacher_model_dir)
    HEARTBEAT['stage']='build_causal_train_features'
    train_forward=load_aux(args.train_forward)
    train_neighbor,names=load_past_neighbor(args.train_neighbor)
    train_x,train_y,train_groups,train_locations,_=causal_arrays(train_predictions,train_forward,train_neighbor,True)
    if train_locations!=teacher_locations:
        raise ValueError('teacher and causal train locations do not align')
    del train_predictions,train_forward,train_neighbor
    gc.collect()

    HEARTBEAT['stage']='build_causal_validation_features'
    val_predictions=load_predictionsgt(args.val_pkl)
    val_forward=load_aux(args.val_forward)
    val_neighbor,val_names=load_past_neighbor(args.val_neighbor)
    val_x,val_y,val_groups,val_locations,val_sequences=causal_arrays(val_predictions,val_forward,val_neighbor,True)
    val_teacher=align_score_map(args.val_teacher_scores,args.val_teacher_field,val_locations)
    del val_predictions,val_forward,val_neighbor
    gc.collect()

    HEARTBEAT['stage']='build_causal_test_features'
    test_predictions=load_predictionsgt(args.test_pkl)
    test_forward=load_aux(args.test_forward)
    test_neighbor,test_names=load_past_neighbor(args.test_neighbor)
    test_x,_,_,test_locations,_=causal_arrays(test_predictions,test_forward,test_neighbor,False)
    del test_predictions,test_forward,test_neighbor
    gc.collect()
    assert names==val_names==test_names

    output_oof=np.zeros(len(val_x),np.float32)
    test_predictions_list=[]
    models=[]
    args.out_model_dir.mkdir(parents=True,exist_ok=True)
    for held in sorted(set(val_sequences)):
        HEARTBEAT['stage']=f'train_distilled_without_{held}'
        feature_parts=[train_x]
        quality_parts=[train_y]
        teacher_parts=[teacher_train]
        groups=list(train_groups)
        cursor=len(train_x)
        for start,stop in val_groups:
            if val_sequences[start]==held:
                continue
            feature_parts.append(val_x[start:stop])
            quality_parts.append(val_y[start:stop])
            teacher_parts.append(val_teacher[start:stop])
            groups.append((cursor,cursor+stop-start))
            cursor+=stop-start
        fit_x=np.concatenate(feature_parts)
        fit_y=np.concatenate(quality_parts)
        fit_teacher=np.concatenate(teacher_parts)
        model,count,positive,mean_target=fit(fit_x,fit_y,fit_teacher,groups,args.teacher_weight)
        mask=val_sequences==held
        output_oof[mask]=np.clip(model.predict(val_x[mask]),0.0,1.0)
        test_predictions_list.append(np.clip(model.predict(test_x),0.0,1.0))
        model_path=args.out_model_dir/f'action_chunk_causal_distilled_without_{held}.ubj'
        model.save_model(model_path)
        record={'excluded_validation_video':held,'hard_rows':count,'positive_rows':positive,'mean_soft_target':mean_target,'model':str(model_path)}
        models.append(record)
        print(json.dumps({'kind':'action_chunk_causal_distilled_model',**record}),flush=True)
        del fit_x,fit_y,fit_teacher,model
        gc.collect()
    HEARTBEAT['stage']='write_distilled_scores'
    write_score_jsonl(args.out_val_scores,output_oof,val_locations,args.score_field)
    write_score_jsonl(args.out_test_scores,np.mean(np.stack(test_predictions_list),axis=0).astype(np.float32),test_locations,args.score_field)
    summary={'model':'strict causal Action Chunk student distilled from offline Action Chunk teacher','inference_features':'raw detector + forward 1s/3s bank + past-only neighbors','teacher_information':'training soft target only; teacher/backward absent from test arguments and deployed features','teacher_weight':args.teacher_weight,'features':train_x.shape[1],'neighbor_features':names,'models':models}
    args.out_summary.write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
