#!/usr/bin/env python3
"""Merge disjoint SAMURAI evaluation shards into one canonical run."""
from __future__ import annotations
import argparse, csv, json, math, shutil
from pathlib import Path
import numpy as np


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--shard-root',type=Path,action='append',required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--expected-sequences',type=int,required=True)
    return p.parse_args()


def read_rows(path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))


def summarize(name,rows):
    vis=[r for r in rows if int(r['visible'])]
    i=np.asarray([float(r['iou']) for r in vis],dtype=float)
    e=np.asarray([float(r['center_error']) for r in vis],dtype=float)
    t=np.linspace(0,1,21)
    return {'sequence':name,'frames':len(rows),'visible_frames':len(vis),'mean_iou':float(i.mean()) if len(i) else 0.0,'success_auc':float(np.mean([(i>=x).mean() for x in t])) if len(i) else 0.0,'success_50':float((i>=.5).mean()) if len(i) else 0.0,'precision_5':float((e<=5).mean()) if len(e) else 0.0,'precision_10':float((e<=10).mean()) if len(e) else 0.0,'precision_20':float((e<=20).mean()) if len(e) else 0.0}


def main():
    a=parse_args(); files={}
    for root in a.shard_root:
        for p in (root/'predictions').glob('*.csv'):
            if p.stem in files: raise ValueError(f'duplicate sequence {p.stem}')
            files[p.stem]=p
    if len(files)!=a.expected_sequences: raise ValueError(f'incomplete shards: {len(files)}/{a.expected_sequences}')
    pred=a.output_root/'predictions'; pred.mkdir(parents=True,exist_ok=True)
    all_rows=[]; seq=[]
    for name in sorted(files):
        dest=pred/files[name].name
        if not dest.exists(): shutil.copy2(files[name],dest)
        rows=read_rows(files[name]); all_rows.extend(rows); seq.append(summarize(name,rows))
    vis=[r for r in all_rows if int(r['visible'])]
    i=np.asarray([float(r['iou']) for r in vis]); e=np.asarray([float(r['center_error']) for r in vis]); t=np.linspace(0,1,21)
    first = json.loads((a.shard_root[0] / 'metrics.json').read_text(encoding='utf-8-sig'))
    report={k:first[k] for k in ('model_config','checkpoint','device','dtype','propagation_mode')}
    report.update(sequences=len(seq),frames=len(all_rows),visible_frames=len(vis),mean_iou=float(i.mean()),success_auc=float(np.mean([(i>=x).mean() for x in t])),success_50=float((i>=.5).mean()),precision_5=float((e<=5).mean()),precision_10=float((e<=10).mean()),precision_20=float((e<=20).mean()),sequence_results=seq)
    (a.output_root / 'metrics.json').write_text(json.dumps(report, indent=2) + chr(10), encoding='utf-8')
    (a.output_root / 'progress.json').write_text(json.dumps({'status': 'completed', 'done_sequences': len(seq), 'total_sequences': len(seq), 'done_frames': len(all_rows), 'last_completed_sequence': seq[-1]['sequence']}, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(report,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
