from __future__ import annotations
import json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path

REPO=Path(r'C:\Users\aaron\Desktop\URAP')
RUN=REPO/'artifacts'/'detached_action_chunk_causal_distilled_v65'
PROGRESS=RUN/'progress.json'
OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_causal_distilled_v65')
DATA=Path(r'D:\URAP_vatd_rank_results\action_chunk_full_dev_v36')
NEIGHBOR=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_v44')
TEACHER=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46')
PREREQUISITE=Path(r'D:\URAP_vatd_rank_results\action_chunk_projected_heuristic_v64\official_summary.json')


def report(stage: str,done: int,total: int=3,**extra) -> None:
    RUN.mkdir(parents=True,exist_ok=True)
    payload={'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra}
    PROGRESS.write_text(json.dumps(payload),encoding='utf8')
    print(json.dumps(payload),flush=True)


def execute(stage: str,done: int,command: list[str]) -> None:
    process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'})
    report(stage,done,child_pid=process.pid,command=command)
    code=process.wait()
    if code:
        raise RuntimeError(f'{stage} failed with {code}')


def main() -> int:
    while not PREREQUISITE.is_file():
        report('waiting_for_causal_heuristic_v64',0,prerequisite=str(PREREQUISITE))
        time.sleep(60)
    prior=json.loads(PREREQUISITE.read_text(encoding='utf8'))
    if prior.get('target_met'):
        report('skipped_target_already_met',3,prior_result=str(PREREQUISITE))
        return 0
    OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(REPO/'tools'/'train_action_chunk_causal_distilled.py')]
    for split,pkl in [('train',r'D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl'),('val',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl')]:
        command += [f'--{split}-pkl',pkl,f'--{split}-forward',str(DATA/f'{split}_forward.jsonl'),f'--{split}-backward',str(DATA/f'{split}_backward.jsonl'),f'--{split}-neighbor',str(NEIGHBOR/f'{split}_neighbor_scores.jsonl')]
    command += ['--test-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--test-forward',str(DATA/'test_forward.jsonl'),'--test-neighbor',str(NEIGHBOR/'test_neighbor_scores.jsonl'),'--teacher-model-dir',str(TEACHER/'models'),'--val-teacher-scores',str(TEACHER/'val_oof_scores.jsonl'),'--teacher-weight','.4','--out-val-scores',str(OUT/'val_oof_scores.jsonl'),'--out-test-scores',str(OUT/'test_scores.jsonl'),'--out-model-dir',str(OUT/'models'),'--out-summary',str(OUT/'train_summary.json')]
    execute('train_causal_distilled_student',0,command)
    execute('select_fusion_on_oof',1,[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--tracklet-jsonl',str(OUT/'val_oof_scores.jsonl'),'--per-row-score','--score-field','action_chunk_causal_distilled_score','--modes','geom-mix','--alphas','.02,.04,.06,.08,.1,.14,.2,.3,.4,.55','--out-json',str(OUT/'val_sweep.json')])
    validation=json.loads((OUT/'val_sweep.json').read_text(encoding='utf8'))['best']
    execute('fixed_test',2,[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--tracklet-jsonl',str(OUT/'test_scores.jsonl'),'--per-row-score','--score-field','action_chunk_causal_distilled_score','--modes','geom-mix','--alphas',str(validation['alpha']),'--out-json',str(OUT/'test_fixed.json')])
    test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best']
    summary={'protocol':'strict causal Action Chunk student; offline bidirectional Action Chunk teacher used only as train/OOF soft target; fixed test','validation_selection':validation,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97}
    (OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    report('done',3,summary=summary)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
