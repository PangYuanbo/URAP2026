import argparse
import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--progress', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--frames-root', type=Path, default=Path(r'D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test'))
    parser.add_argument('--tracklets', type=Path)
    parser.add_argument('--appearance-search-fraction', type=float, default=0.0)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    prediction_dir = out_dir / 'aotpredictions'
    shard_root = out_dir / 'shards'
    logs = out_dir / 'logs'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    python = repo / 'papers/TransVisDrone/.venv/Scripts/python.exe'
    source = args.source.resolve() if args.source else repo / 'papers/TransVisDrone/runs/val/AOT_URAP/fulltest_conf0p2_wport_baseline/aotpredictions'
    frames = args.frames_root.resolve()
    tracklets = args.tracklets.resolve() if args.tracklets else repo / 'artifacts/route_b_official/aot_action_chunk_transfer_v1/tracklets_with_action_chunk_scores.jsonl'
    parts = sorted(source.glob('predictions_split_*.pkl'), key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    lock = threading.Lock()
    state = {'completed': set(), 'running': {}, 'failed': {}}
    summaries = {}

    for part in parts:
        split_id = int(part.stem.rsplit('_', 1)[1])
        final_output = prediction_dir / f'predictions_split_{split_id}.pkl'
        shard_summary = shard_root / f'part{split_id}' / 'flow_recovery_summary.json'
        if final_output.exists() and shard_summary.exists():
            state['completed'].add(split_id)
            summaries[split_id] = json.loads(shard_summary.read_text(encoding='utf-8-sig'))

    def run_part(part):
        split_id = int(part.stem.rsplit('_', 1)[1])
        output_name = f'predictions_split_{split_id}.pkl'
        final_output = prediction_dir / output_name
        shard_dir = shard_root / f'part{split_id}'
        shard_summary = shard_dir / 'flow_recovery_summary.json'
        child_progress = shard_dir / 'progress.json'
        if final_output.exists() and shard_summary.exists():
            return split_id, json.loads(shard_summary.read_text(encoding='utf-8-sig'))
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / f'part{split_id}.out.txt'
        stderr_path = logs / f'part{split_id}.err.txt'
        command = [
            str(python), str(repo / 'tools/aot_action_bank_flow_recovery.py'),
            '--results-folder', str(source), '--prediction-part', str(part), '--output-name', output_name,
            '--tracklets', str(tracklets), '--frames-root', str(frames / f'part{split_id}' / 'frames'),
            '--out-dir', str(shard_dir), '--progress', str(child_progress),
            '--min-action-score', '0.6', '--max-gap', '30', '--promotion-iou', '0.3',
            '--appearance-search-fraction', str(args.appearance_search_fraction)
        ]
        with lock:
            state['running'][split_id] = {'pid': None, 'group_done': 0, 'group_total': 0, 'last_completed_unit': 'starting'}
        with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
            process = subprocess.Popen(command, cwd=repo, stdout=stdout, stderr=stderr)
            with lock:
                state['running'][split_id]['pid'] = process.pid
            while process.poll() is None:
                if child_progress.exists():
                    child = json.loads(child_progress.read_text(encoding='utf-8-sig'))
                    with lock:
                        state['running'][split_id].update({'group_done': child.get('done', 0), 'group_total': child.get('total', 0), 'last_completed_unit': child.get('last_completed_unit', 'running')})
                time.sleep(5)
        with lock:
            state['running'].pop(split_id, None)
        if process.returncode:
            with lock:
                state['failed'][split_id] = {'exit_code': process.returncode, 'stderr': str(stderr_path)}
            raise RuntimeError(f'part{split_id} failed exit={process.returncode}; stderr={stderr_path}')
        shard_output = shard_dir / 'aotpredictions' / output_name
        if not shard_output.exists() or not shard_summary.exists():
            raise RuntimeError(f'part{split_id} output missing')
        shutil.copy2(shard_output, final_output)
        summary = json.loads(shard_summary.read_text(encoding='utf-8-sig'))
        with lock:
            state['completed'].add(split_id)
            summaries[split_id] = summary
        return split_id, summary

    pending = [part for part in parts if int(part.stem.rsplit('_', 1)[1]) not in state['completed']]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_part, part) for part in pending]
        while any(not future.done() for future in futures):
            with lock:
                payload = {
                    'status': 'running', 'done': len(state['completed']), 'total': len(parts),
                    'completed_parts': sorted(state['completed']), 'running_parts': state['running'],
                    'failed_parts': state['failed'], 'last_completed_unit': f'part{max(state["completed"])}' if state['completed'] else 'none',
                    'workers': args.workers, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            write_json(args.progress, payload)
            time.sleep(10)
        for future in futures:
            future.result()

    combined = {'protocol': 'sharded camera-motion-compensated action-bank gap recovery', 'source': str(source), 'frames_root': str(frames), 'parts': len(parts), 'records': sum(int(item.get('records', 0)) for item in summaries.values()), 'frame_paths': sum(int(item.get('frame_paths', 0)) for item in summaries.values()), 'counters': {}, 'uses_labels': False}
    for item in summaries.values():
        for key, value in (item.get('counters') or {}).items():
            combined['counters'][key] = combined['counters'].get(key, 0) + int(value)
    write_json(out_dir / 'flow_recovery_summary.json', combined)
    write_json(args.progress, {'status': 'complete', 'done': len(parts), 'total': len(parts), 'completed_parts': sorted(state['completed']), 'running_parts': {}, 'last_completed_unit': 'flow_recovery_summary', 'summary': combined, 'workers': args.workers, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')})


if __name__ == '__main__':
    main()

