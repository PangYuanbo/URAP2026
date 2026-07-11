from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path

REPO=Path(r'C:\Users\aaron\Desktop\URAP')
RUN=REPO/'artifacts'/'detached_action_chunk_rank_transplant_v66'
PROGRESS=RUN/'progress.json'
OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_rank_transplant_v66')


def report(stage: str,done: int,total: int=2,**extra) -> None:
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
    OUT.mkdir(parents=True,exist_ok=True)
    roots={
        'causal_v38':(Path(r'D:\URAP_vatd_rank_results\action_chunk_causal_v38'),'action_chunk_causal_score'),
        'causal_memory_v59':(Path(r'D:\URAP_vatd_rank_results\action_chunk_causal_memory_v59'),'action_chunk_causal_memory_score'),
        'projected_memory_v62':(Path(r'D:\URAP_vatd_rank_results\action_chunk_projected_causal_memory_v62'),'action_chunk_causal_memory_score'),
        'distilled_v65':(Path(r'D:\URAP_vatd_rank_results\action_chunk_causal_distilled_v65'),'action_chunk_causal_distilled_score'),
    }
    val_sources={name:{'path':str(root/'val_oof_scores.jsonl'),'field':field} for name,(root,field) in roots.items()}
    test_sources={name:{'path':str(root/'test_scores.jsonl'),'field':field} for name,(root,field) in roots.items()}
    (OUT/'val_sources.json').write_text(json.dumps(val_sources,indent=2),encoding='utf8')
    (OUT/'test_sources.json').write_text(json.dumps(test_sources,indent=2),encoding='utf8')
    common=[sys.executable,str(REPO/'tools'/'sweep_action_chunk_rank_transplant.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone']
    execute('select_rank_transplant_on_oof',0,common+['--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--sources-json',str(OUT/'val_sources.json'),'--out-json',str(OUT/'val_sweep.json')])
    execute('fixed_test',1,common+['--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--sources-json',str(OUT/'test_sources.json'),'--fixed-config-json',str(OUT/'val_sweep.json'),'--out-json',str(OUT/'test_fixed.json')])
    validation=json.loads((OUT/'val_sweep.json').read_text(encoding='utf8'))['best']
    test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best']
    summary={'protocol':'strict causal frame-rank transplant; every frame preserves the exact original detector score multiset; OOF selection; fixed test','validation_selection':validation,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97}
    (OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    report('done',2,summary=summary)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
