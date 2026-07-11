from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime
from pathlib import Path

REPO=Path(r'C:\Users\aaron\Desktop\URAP')
RUN=REPO/'artifacts'/'detached_action_chunk_conservative_graph_v68'
PROGRESS=RUN/'progress.json'
OUT=Path(r'D:\URAP_vatd_rank_results\action_chunk_conservative_graph_v68')
BASE=Path(r'D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46')
HOM=Path(r'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies')


def report(stage: str,done: int,total: int=4,**extra) -> None:
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
    graph=REPO/'tools'/'score_action_chunk_graph.py'
    common=[sys.executable,str(graph),'--unary-field','action_chunk_neighbor_score','--decays','.02,.04,.08,.12,.2','--transition-weights','.05,.1,.2,.4']
    execute('score_validation_conservative_graph',0,common+['--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--unary-scores',str(BASE/'val_oof_scores.jsonl'),'--homography-cache',str(HOM/'val.pkl'),'--out-jsonl',str(OUT/'val_graph_scores.jsonl'),'--out-summary',str(OUT/'val_graph_summary.json')])
    fields=json.loads((OUT/'val_graph_summary.json').read_text(encoding='utf8'))['fields']
    best=None
    for field in fields:
        path=OUT/f'val_sweep_{field}.json'
        execute(f'validate_{field}',1,[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--tracklet-jsonl',str(OUT/'val_graph_scores.jsonl'),'--per-row-score','--score-field',field,'--modes','geom-mix','--alphas','.01,.02,.04,.06,.08,.1,.14,.2,.3','--out-json',str(path)])
        candidate={**json.loads(path.read_text(encoding='utf8'))['best'],'field':field}
        if best is None or candidate['map50']>best['map50']:
            best=candidate
    (OUT/'validation_selection.json').write_text(json.dumps(best,indent=2),encoding='utf8')
    execute('score_test_conservative_graph',2,common+['--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--unary-scores',str(BASE/'test_scores.jsonl'),'--homography-cache',str(HOM/'test.pkl'),'--out-jsonl',str(OUT/'test_graph_scores.jsonl'),'--out-summary',str(OUT/'test_graph_summary.json')])
    execute('fixed_test',3,[sys.executable,str(REPO/'tools'/'sweep_tvd_predictionsgt_score_fusion.py'),'--tvd-root',r'D:\urap_modal_stage\TransVisDrone','--predictionsgt-pkl',r'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--tracklet-jsonl',str(OUT/'test_graph_scores.jsonl'),'--per-row-score','--score-field',best['field'],'--modes','geom-mix','--alphas',str(best['alpha']),'--out-json',str(OUT/'test_fixed.json')])
    test=json.loads((OUT/'test_fixed.json').read_text(encoding='utf8'))['best']
    summary={'protocol':'pure offline bidirectional Action Chunk conservative path consistency on V46 unary; validation selection; fixed test','validation_selection':best,'test_fixed':test,'target_map50':.97,'target_met':test['map50']>=.97}
    (OUT/'official_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    report('done',4,summary=summary)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
