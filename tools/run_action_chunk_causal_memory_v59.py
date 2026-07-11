from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path

REPO=Path(r'C:\Users\aaron\Desktop\URAP')
RUN=REPO/'artifacts'/'detached_action_chunk_causal_memory_v59'
PROGRESS=RUN/'progress.json'
OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_causal_memory_v59')
IMMEDIATE=Path(r'D:\URAP_vatd_rank_results\action_chunk_full_dev_v36')
PERSISTENT=Path(r'D:\URAP_vatd_rank_results\action_chunk_persistent_forward_v54')
NEIGHBOR=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_v44')


def report(stage: str,done: int,total: int=3,**extra) -> None:
    RUN.mkdir(parents=True,exist_ok=True)
    PROGRESS.write_text(json.dumps({'stage':stage,'done':done,'total':total,'updated':datetime.now().astimezone().isoformat(),**extra}),encoding='utf8')


def execute(stage: str,done: int,command: list[str]) -> None:
    process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO)+os.pathsep+str(REPO/'tools'),'PYTHONUNBUFFERED':'1'})
    report(stage,done,child_pid=process.pid,command=command)
    code=process.wait()
    if code:
        raise RuntimeError(f'{stage} failed with {code}')


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(REPO/'tools'/'train_action_chunk_causal_memory.py')]
    for split,pkl in [('train',r'D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl'),('val',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl')]:
        command += [f'--{split}-pkl',pkl,f'--{split}-immediate',str(IMMEDIATE/f'{split}_forward.jsonl'),f'--{split}-persistent',str(PERSISTENT/f'{split}_forward.jsonl'),f'--{split}-backward',str(IMMEDIATE/f'{split}_backward.jsonl'),f'--{split}-neighbor',str(NEIGHBOR/f'{split}_neighbor_scores.jsonl')]
    command += ['--test-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--test-immediate',str(IMMEDIATE/'test_forward.jsonl'),'--test-persistent',str(PERSISTENT/'test_forward.jsonl'),'--test-neighbor',str(NEIGHBOR/'test_neighbor_scores.jsonl'),'--out-val-scores',str(OUT/'val_oof_scores.jsonl'),'--out-test-scores',str(OUT/'test_scores.jsonl'),'--out-model-dir',str(OUT/'models'),'--out-summary',str(OUT/'train_summary.json')]
    execute('train_strict_causal_memory',0,command)
    execute('select_fusion_on_oof',1,[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--tracklet-jsonl',str(OUT/'val_oof_scores.jsonl'),'--per-row-score','--score-field','action_chunk_causal_memory_score','--modes','geom-mix','--alphas','.02,.04,.06,.08,.1,.14,.2,.3,.4,.55','--out-json',str(OUT/'val_sweep.json')])
    validation=json.loads((OUT/'val_sweep.json').read_text(encoding='utf8'))['best']
    execute('fixed_test',2,[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--tracklet-jsonl',str(OUT/'test_scores.jsonl'),'--per-row-score','--score-field','action_chunk_causal_memory_score','--modes','geom-mix','--alphas',str(validation['alpha']),'--out-json',str(OUT/'test_fixed.json')])
    test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best']
    summary={'protocol':'strict causal pure Action Chunk 1s/3s memory; future evidence used only as training supervision; OOF selection; fixed test','validation_selection':validation,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97}
    (OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    report('done',3,summary=summary)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
