import argparse
import json
import subprocess

import time
from pathlib import Path


def write_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)



def run_logged(command, cwd, stdout_path, stderr_path):
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        result = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=False)
    if result.returncode:
        raise RuntimeError(f'Command failed exit={result.returncode}; stderr={stderr_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    logs = output_root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / 'progress.json'
    flow_root = repo / 'artifacts/route_b_official/aot_action_bank_flow_full_v30'
    flow_summary = flow_root / 'flow_recovery_summary.json'
    flow_runner = repo / 'artifacts/route_b_official/aot_action_bank_flow_full_v30_runner'
    flow_pid_file = flow_runner / 'aot_flow_full_v30_pid.txt'
    flow_progress = flow_runner / 'aot_flow_full_v30_progress.json'
    while not flow_summary.exists():
        pid = int(flow_pid_file.read_text(encoding='ascii').strip())
        if not flow_progress.exists() or time.time() - flow_progress.stat().st_mtime > 180:
            raise RuntimeError('Flow v30 progress is stale and summary is missing')
        flow = {'done': 0, 'total': 440, 'last_completed_unit': 'none'}
        if flow_progress.exists():
            flow = json.loads(flow_progress.read_text(encoding='utf-8-sig'))
        write_progress(progress_path, {'phase': 'wait_flow_v30', 'done': flow['done'], 'total': flow['total'], 'flow_pid': pid, 'last_completed_unit': flow.get('last_completed_unit'), 'updated': time.strftime('%Y-%m-%d %H:%M:%S')})
        time.sleep(30)
    python = repo / 'papers/TransVisDrone/.venv/Scripts/python.exe'
    gate_out = repo / 'artifacts/route_b_official/aot_action_bank_flow_full_v31_fixed'
    write_progress(progress_path, {'phase': 'apply_fixed_gate', 'done': 1, 'total': 3, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')})
    run_logged([
        str(python), str(repo / 'tools/aot_action_bank_train_apply_fixed_gate.py'),
        '--train-predictions', str(repo / 'artifacts/route_b_official/aot_action_bank_flow_part0_v27/aotpredictions/predictions_split_0.pkl'),
        '--train-candidate-match-folder', str(repo / 'artifacts/route_b_official/aot_action_bank_flow_part0_v27/official_eval/result/result_metrics_min_track_len_0'),
        '--train-baseline-match-folder', str(repo / 'artifacts/route_b_official/aot_action_chunk_transfer_v1/val_baseline_eval/result/result_metrics_min_track_len_0'),
        '--target-predictions', str(flow_root / 'aotpredictions/predictions_split_0.pkl'),
        '--out-dir', str(gate_out), '--candidate-threshold', '0.05', '--base-threshold', '0.1'
    ], repo, logs / 'apply_fixed_gate.out.txt', logs / 'apply_fixed_gate.err.txt')
    write_progress(progress_path, {'phase': 'official_eval', 'done': 2, 'total': 3, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')})
    eval_out = gate_out / 'official_eval'
    run_logged([
        str(python), './evaluate_aot.py', '--results_folder', str(gate_out / 'aotpredictions'),
        '--evaluation_folder', str(eval_out), '--detection_threshold', '0.2', '--dataset-path', r'U:\URAP_datasets\AOT\part1'
    ], repo / 'papers/TransVisDrone', logs / 'official_eval.out.txt', logs / 'official_eval.err.txt')
    summaries = sorted((eval_out / 'summaries').glob('result_metrics*_summary*.json'), key=lambda path: path.stat().st_mtime, reverse=True)
    if not summaries:
        raise RuntimeError('Official summary missing')
    metrics = json.loads(summaries[0].read_text(encoding='utf-8-sig'))
    baseline = {'afdr': 0.8685312193818473, 'fppi': 0.262303510022747, 'edr300': 0.9257142857142857}
    gain = 100 * (float(metrics['fl_dr_in_range']) - baseline['afdr'])
    result = {'protocol': 'part0-selected fixed full AOT camera-compensated Action Bank', 'baseline': baseline, 'full': {'afdr': float(metrics['fl_dr_in_range']), 'fppi': float(metrics['fppi']), 'far': float(metrics['far']), 'summary': str(summaries[0])}, 'afdr_gain_points': gain, 'fppi_change': float(metrics['fppi']) - baseline['fppi'], 'target_met': 3 <= gain <= 5 and float(metrics['fppi']) <= baseline['fppi']}
    (gate_out / 'official_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_progress(progress_path, {'phase': 'done', 'done': 3, 'total': 3, 'result': result, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')})


if __name__ == '__main__':
    main()



