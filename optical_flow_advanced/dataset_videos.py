from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from video_comparison import digest, process_sequence, read_json, replace_with_retry, timestamp, write_json


REPO = Path(__file__).resolve().parents[1]
FFMPEG = Path('C:/Users/aaron/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0-full_build/bin/ffmpeg.exe')


def indexed_frames(root):
    grouped = defaultdict(list)
    for path in Path(root).iterdir():
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        match = re.fullmatch(r'(?:phantom|Clip_)(\d+)_(\d+)', path.stem)
        if not match:
            raise ValueError(f'Unrecognized frame name: {path}')
        grouped[int(match[1])].append((int(match[2]), str(path)))
    for frames in grouped.values():
        frames.sort()
        if len({index for index, path in frames}) != len(frames):
            raise ValueError('Duplicate frame indices')
    return grouped


def round_robin(sequences):
    grouped = defaultdict(list)
    for sequence in sequences:
        grouped[sequence['dataset']].append(sequence)
    queues = {dataset: deque(sorted(items, key=lambda item: (item['cached_frame_count'], item['name'])))
              for dataset, items in grouped.items()}
    result = []
    while any(queues.values()):
        for dataset in ('NPS', 'AOT', 'ARD'):
            if queues.get(dataset):
                result.append(queues[dataset].popleft())
    return result


def inventory(run_dir, output_root, workers, opencv_threads):
    run_dir, output_root = Path(run_dir).resolve(), Path(output_root).resolve()
    if workers < 1 or opencv_threads < 1:
        raise ValueError('Worker and thread counts must be positive')
    if (run_dir / 'manifest.json').exists():
        raise ValueError('Inventory already exists; use its pinned manifest rather than overwriting it')
    if shutil.disk_usage(output_root.anchor).free < 100 * 1024 ** 3:
        raise ValueError('Output drive needs at least 100 GiB free')
    if not FFMPEG.exists():
        raise FileNotFoundError(FFMPEG)
    sequences = []
    for split, root in [('train', 'D:/URAP_nps_train_tvd/AllFrames/train'),
                        ('val', 'D:/URAP_nps_val_tvd/AllFrames/val'),
                        ('test', 'D:/URAP_nps_test_tvd/AllFrames/test')]:
        for clip, frames in indexed_frames(root).items():
            if any(second[0] != first[0] + 1 for first, second in zip(frames, frames[1:])):
                raise ValueError(f'NPS frame gap: {split}/{clip}')
            name = f'Clip_{clip:03d}'
            video = Path(f'D:/URAP_nps_train_raw/Videos/Clip_{clip}.mov')
            sequence = {'id': f'NPS/{split}/{name}', 'dataset': 'NPS', 'split': split, 'name': name,
                        'cached_frame_count': len(frames), 'original_reference': 'Recomputed author YOLOMG FD5 on shared 1080p canvas'}
            if split == 'train' and video.exists():
                sequence.update(kind='video', source=str(video), stage_source=False)
            else:
                sequence.update(kind='frames', frames=[path for index, path in frames], fps=29.97,
                                fps_provenance='Nominal NPS playback rate, based on local original MOV; frames have no embedded timing',
                                source_frame_indices=[index for index, path in frames])
            sequences.append(sequence)
    aot_root = Path('D:/URAP_local_datasets/AOT_PART0_V88/AOT_part1')
    annotations = json.loads((aot_root / 'ImageSets/groundtruth_part0.json').read_text(encoding='utf-8'))
    for directory in sorted((aot_root / 'Images').iterdir()):
        if not directory.is_dir():
            continue
        frames = sorted(directory.glob('*.png'))
        metadata = annotations['samples'][directory.name]['metadata']
        if len(frames) != int(metadata['number_of_frames']):
            raise ValueError(f'AOT cached frame count differs from metadata: {directory.name}')
        sequences.append({'id': f'AOT/cached_part0/{directory.name}', 'dataset': 'AOT', 'split': 'cached_part0',
                          'name': directory.name, 'kind': 'frames', 'frames': [str(path) for path in frames],
                          'fps': float(metadata['fps']), 'fps_provenance': 'AOT flight metadata',
                          'cached_frame_count': len(frames), 'declared_frame_count': metadata['number_of_frames'],
                          'original_reference': 'Shared YOLOMG FD5 reference; NOT native AOT/TVD preprocessing (TVD uses RGB)',
                          'annotation_source': str(aot_root / 'ImageSets/groundtruth_part0.json')})
    ard_raw = {}
    for split in ('train_videos', 'test_videos'):
        for path in (Path('G:/My Drive/URAP/ARD100') / split).glob('*.mp4'):
            match = re.fullmatch(r'phantom(\d+)', path.stem)
            if match:
                clip = int(match[1])
                if clip in ard_raw:
                    raise ValueError(f'Duplicate raw ARD clip: {clip}')
                ard_raw[clip] = path
    for split, root in [('train', 'D:/URAP_modal_datasets/ARD100_YOLOMG_ORIGINAL/images/train'),
                        ('val', 'D:/URAP_modal_datasets/ARD100_YOLOMG_ORIGINAL/images/val'),
                        ('test', 'D:/URAP_local_datasets/ARD100_YOLOMG_TEST_IMAGES_20260815_1402/test')]:
        for clip, frames in indexed_frames(root).items():
            video = ard_raw.pop(clip)
            name = f'Clip_{clip:03d}'
            gaps = sum(second[0] - first[0] - 1 for first, second in zip(frames, frames[1:]))
            sequences.append({'id': f'ARD/{split}/{name}', 'dataset': 'ARD', 'split': split, 'name': name,
                              'kind': 'video', 'source': str(video), 'stage_source': True,
                              'source_bytes': video.stat().st_size, 'cached_frame_count': len(frames),
                              'cached_frame_gaps_avoided_by_raw_video': gaps,
                              'original_reference': 'Recomputed author YOLOMG FD5 on full raw video; NOT reused historical cached masks'})
    if ard_raw:
        raise ValueError(f'Unassigned raw ARD clips: {list(ard_raw)}')
    counts = {dataset: sum(sequence['dataset'] == dataset for sequence in sequences) for dataset in ('AOT', 'NPS', 'ARD')}
    if counts != {'AOT': 10, 'NPS': 50, 'ARD': 100}:
        raise ValueError(f'Unexpected inventory: {counts}')
    run_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    snapshots = run_dir / 'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    new_source = REPO / 'tools/yolomg_parallax_robust_difference.py'
    author_source = REPO / 'URAP-UAV-to-UAV-Detection-and-Tracking/papers/YOLOMG/test_code/MOD_Functions.py'
    method_snapshot, author_snapshot = snapshots / new_source.name, snapshots / author_source.name
    shutil.copy2(new_source, method_snapshot)
    shutil.copy2(author_source, author_snapshot)
    source_paths = [method_snapshot, author_snapshot, Path(__file__), Path(__file__).with_name('video_comparison.py')]
    manifest = {'created_at': timestamp(), 'run_dir': str(run_dir), 'output_root': str(output_root),
                'workers': workers, 'opencv_threads': opencv_threads, 'ffmpeg': str(FFMPEG),
                'ffprobe': str(FFMPEG.with_name('ffprobe.exe')), 'method_snapshot': str(method_snapshot),
                'author_snapshot': str(author_snapshot), 'source_hashes': {str(path): digest(path) for path in source_paths},
                'parameters': json.loads((REPO / 'optical_flow_advanced/configs/parallax_robust_default.json').read_text()),
                'metric_stride': 30, 'display_gain': 3.0, 'sequence_counts': counts,
                'cached_frame_counts': {dataset: sum(item['cached_frame_count'] for item in sequences if item['dataset'] == dataset)
                                        for dataset in counts},
                'protocol': {'scope': 'All 100 ARD raw videos, all 50 NPS clips, and the 10 locally cached AOT flights only.',
                             'aot_coverage': 'PARTIAL: not the complete public AOT dataset',
                             'new': 'Pinned parallax-robust method, t-1/t/t+1, default parameters, no training',
                             'original': 'Author motion_compensate + Gaussian11 + FD5 t-2/t/t+2 + author JPEG conversion',
                             'original_arithmetic': 'Preserves author uint8 addition before division, including wraparound; no silent bug fixes',
                             'control': 'Same new tracking and t-1/t/t+1, local=False, without adaptive suppression',
                             'raw_control': 'Parallax-aligned residual before adaptive suppression; recorded in metrics',
                             'geometry': 'Aspect-preserving letterbox to 1920x1080 before all methods; shared content mask excludes padding',
                             'boundaries': 'Clamp temporal neighbors to the same source sequence; first/last two source frames excluded from sampled metrics',
                             'playback': 'Source video FPS or declared image-sequence FPS; no frame dropping, no audio',
                             'display': 'Standalone motion videos use unscaled grayscale; comparison uses identical fixed linear x3 for all motion panels',
                             'metrics': 'Every 30th frame, common valid-region intensity; author failure excludes all methods for that sample; not AP, tracking, or GT-based target preservation',
                             'limitations': ['AOT native TVD has no original flow; author FD5 is a shared reference only',
                                             'Full-pipeline old/new comparison differs in temporal spacing, blur, arithmetic and suppression',
                                             'Current local field uses all reliable tracks, not only homography inliers',
                                             'Compressed videos are viewing artifacts; quantitative metrics are computed before encoding']},
                'sequences': round_robin(sequences)}
    write_json(run_dir / 'manifest.json', manifest)
    write_json(output_root / 'manifest.json', manifest)
    render_catalog(manifest, [])
    print(json.dumps({'manifest': str(run_dir / 'manifest.json'), 'output_root': str(output_root),
                      'sequence_counts': counts, 'cached_frame_counts': manifest['cached_frame_counts']}, indent=2))
    return manifest


def sequence_directory(manifest, sequence):
    return Path(manifest['output_root']) / sequence['dataset'] / sequence['split'] / sequence['name']


def render_catalog(manifest, reports):
    root = Path(manifest['output_root'])
    rows = []
    summaries = {}
    for dataset in ('AOT', 'NPS', 'ARD'):
        selected = [report for report in reports if report.get('dataset') == dataset and report.get('status') == 'completed']
        summaries[dataset] = {'completed_sequences': len(selected), 'total_sequences': manifest['sequence_counts'][dataset],
                              'completed_frames': sum(report['frames'] for report in selected), 'metrics': {},
                              'author_reference_failed_frames': sum(report.get('author_reference_failed_frames', 0) for report in selected)}
        for method in ('author_fd5', 'global_only', 'parallax_raw', 'parallax'):
            metrics = [report['metrics'][method] for report in selected if method in report['metrics']]
            pixels = sum(item['valid_pixels'] for item in metrics)
            summaries[dataset]['metrics'][method] = {'valid_pixels': pixels,
                'sampled_frames': sum(item['sampled_frames'] for item in metrics),
                'mean_intensity': sum(item['mean_intensity'] * item['valid_pixels'] for item in metrics if item['mean_intensity'] is not None) / pixels if pixels else None,
                'active_fraction_gt8': sum(item['active_fraction_gt8'] * item['valid_pixels'] for item in metrics if item['active_fraction_gt8'] is not None) / pixels if pixels else None}
    for report in reports:
        if report.get('status') != 'completed':
            continue
        relative = report['sequence']
        rows.append('<tr><td>' + html.escape(relative) + '</td><td>' + str(report['frames']) + '</td><td>' +
                    ' | '.join(f'<a href="{html.escape(relative)}/{name}">{label}</a>' for name, label in
                               [('comparison.mp4', '4-panel comparison'), ('new_motion.mp4', 'New motion'),
                                ('original_fd5.mp4', 'Author FD5'), ('preview.jpg', 'Preview'), ('report.json', 'Metrics')]) + '</td></tr>')
    status_text = ' / '.join(f'{dataset}: {value["completed_sequences"]}/{value["total_sequences"]}' for dataset, value in summaries.items())
    metric_rows = []
    table_lines = ['| Dataset | Completed sequences | Sampled frames | Author FD5 mean | Global-only mean | Parallax raw mean | New final mean |',
                   '|---|---:|---:|---:|---:|---:|---:|']
    for dataset, summary in summaries.items():
        values = [summary['metrics'][method]['mean_intensity'] for method in ('author_fd5', 'global_only', 'parallax_raw', 'parallax')]
        formatted = [f'{value:.4f}' if value is not None else 'pending' for value in values]
        cells = [dataset, f'{summary["completed_sequences"]}/{summary["total_sequences"]}',
                 str(summary['metrics']['parallax']['sampled_frames'])] + formatted
        metric_rows.append('<tr>' + ''.join('<td>' + html.escape(cell) + '</td>' for cell in cells) + '</tr>')
        table_lines.append('| ' + ' | '.join(cells) + ' |')
    report_text = '# Optical-flow compensated-difference comparison\n\n'
    report_text += f'Updated: {timestamp()}\n\n{status_text}\n\n'
    report_text += '**Coverage:** 100 ARD raw videos, 50 NPS clips, only 10 cached AOT flights (partial AOT). Values below cover completed sequences only.\n\n'
    report_text += '\n'.join(table_lines) + '\n\n'
    report_text += 'Intensity is in 0-255 units, measured before video encoding on identical valid pixels sampled every 30th frame. These are not detection or tracking scores. AOT FD5 is a shared YOLOMG reference, not native TVD preprocessing.\n\n'
    report_text += '## How to interpret the comparison\n\n'
    report_text += '- Compare global-only against parallax raw for the effect of local alignment, keeping the tracker and temporal spacing matched.\n'
    report_text += '- Compare parallax raw against new final for adaptive suppression; a large drop here is not automatically an optical-flow improvement.\n'
    report_text += '- Author FD5 versus new final also changes temporal spacing, blur, arithmetic, and suppression, so it is a complete-pipeline comparison.\n'
    report_text += '- Lower image-wide residual activity is not proof of better background cancellation or target preservation. Object-region metrics and downstream AP/recall have not been measured here.\n'
    report_text += '- Inspect four-panel videos for registration errors, target disappearance, border artifacts, and temporal consistency. Standalone MP4s have no display gain; the four-panel display uses a shared x3 gain.\n\n'
    report_text += 'Author-reference failed frames: ' + str(sum(report.get('author_reference_failed_frames', 0) for report in reports)) + '. Failed samples are excluded from all shared metrics.\n'
    (root / 'comparison_report.md').write_text(report_text, encoding='utf-8')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>URAP motion comparisons</title>'
    page += '<style>body{font:16px system-ui;background:#111827;color:#edf2f7;max-width:1400px;margin:40px auto;padding:20px}a{color:#71c9ff}td,th{padding:12px;text-align:left;border-bottom:1px solid #344156}p{line-height:1.6}table{width:100%}</style>'
    page += '<h1>Parallax-robust motion comparisons</h1><p>' + html.escape(status_text) + '</p>'
    page += '<p><strong>AOT is partial: 10 cached flights only.</strong> NPS: 50 clips. ARD: 100 full raw videos. Refresh this local page for completed outputs.</p>'
    page += '<p>Panels: RGB / author FD5 reference / matched-temporal global-only control / new parallax + suppression. AOT has no native TVD optical-flow input; FD5 is a shared reference. All motion panels use identical fixed linear display gain. Standalone motion files are unscaled grayscale.</p>'
    page += '<p>Metrics measure image-wide valid-region residual intensity, not detection accuracy or ground-truth target preservation. Darker does not automatically mean better.</p><p><a href="manifest.json">Full protocol</a> | <a href="summary.json">Aggregate metrics</a> | <a href="comparison_report.md">Comparison report</a></p>'
    page += '<table><tr>' + ''.join('<th>' + name + '</th>' for name in ['Dataset', 'Completed sequences', 'Sampled frames', 'Author FD5 mean', 'Global-only mean', 'Parallax raw mean', 'New final mean']) + '</tr>' + ''.join(metric_rows) + '</table>'
    page += '<table><tr><th>Sequence</th><th>Frames</th><th>Saved files</th></tr>' + ''.join(rows) + '</table></html>'
    temporary = root / 'index.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    replace_with_retry(temporary, root / 'index.html')
    write_json(root / 'summary.json', {'updated_at': timestamp(), 'datasets': summaries,
                                     'author_reference_failed_frames': sum(report.get('author_reference_failed_frames', 0) for report in reports),
                                     'interpretation': manifest['protocol']['metrics'], 'aot_coverage': 'partial'})


def run_manifest(manifest_path, resume=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    run_dir = Path(manifest['run_dir'])
    state_path = run_dir / 'progress.json'
    if state_path.exists() and not resume:
        raise ValueError('Run state exists: explicit --resume is required')
    if (run_dir / 'STOP_REQUESTED').exists():
        raise ValueError('STOP_REQUESTED exists; explicitly clear it before resuming')
    completed, pending, failures, active = [], deque(), [], {}
    for sequence in manifest['sequences']:
        report_path = sequence_directory(manifest, sequence) / 'report.json'
        report = read_json(report_path) if report_path.exists() else None
        if report and report.get('status') == 'completed' and not report.get('preview_only'):
            if not all(Path(output['path']).exists() and Path(output['path']).stat().st_size == output['bytes'] for output in report['outputs']):
                raise ValueError(f'Completed output missing/changed: {sequence["id"]}')
            completed.append(report)
        else:
            pending.append(sequence)
    started = timestamp()
    logs = run_dir / 'sequence_logs'
    logs.mkdir(exist_ok=True)
    stopping = False
    while True:
        stopping = stopping or (run_dir / 'STOP_REQUESTED').exists()
        if shutil.disk_usage(manifest['output_root']).free < 40 * 1024 ** 3:
            stopping = True
            print(f'{timestamp()} LOW_DISK: stopping new sequences; active workers may finish', flush=True)
        while pending and len(active) < manifest['workers'] and not stopping:
            sequence = pending.popleft()
            name = sequence['id'].replace('/', '_') + '_' + str(int(time.time()))
            stdout_path, stderr_path = logs / (name + '.stdout.log'), logs / (name + '.stderr.log')
            stdout, stderr = stdout_path.open('wb'), stderr_path.open('wb')
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'process', '--manifest', str(manifest_path), '--sequence', sequence['id']]
            process = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active[process.pid] = {'process': process, 'sequence': sequence, 'stdout': stdout, 'stderr': stderr,
                                   'started_at': timestamp(), 'stdout_path': str(stdout_path), 'stderr_path': str(stderr_path)}
            print(f'{timestamp()} START pid={process.pid} sequence={sequence["id"]}', flush=True)
        for pid, entry in list(active.items()):
            result = entry['process'].poll()
            if result is None:
                continue
            entry['stdout'].close()
            entry['stderr'].close()
            report_path = sequence_directory(manifest, entry['sequence']) / 'report.json'
            report = read_json(report_path) if report_path.exists() else {'error': 'Worker exited without report'}
            if result == 0 and report.get('status') == 'completed':
                completed.append(report)
            else:
                failures.append({'sequence': entry['sequence']['id'], 'exit_code': result,
                                 'error': report.get('error'), 'stderr': entry['stderr_path']})
            print(f'{timestamp()} EXIT pid={pid} sequence={entry["sequence"]["id"]} code={result}', flush=True)
            del active[pid]
            render_catalog(manifest, completed)
        live = []
        for pid, entry in active.items():
            progress_path = sequence_directory(manifest, entry['sequence']) / 'progress.json'
            progress = read_json(progress_path) if progress_path.exists() else {}
            live.append({'pid': pid, 'sequence': entry['sequence']['id'], 'started_at': entry['started_at'],
                         'stdout': entry['stdout_path'], 'stderr': entry['stderr_path'], 'progress': progress})
        final = not active and (not pending or stopping)
        phase = ('stopped' if stopping else 'completed_with_failures' if failures else 'completed') if final else 'running'
        write_json(state_path, {'phase': phase, 'pid': os.getpid(), 'started_at': started, 'updated_at': timestamp(),
                                'done': len(completed), 'total': len(manifest['sequences']), 'unit': 'sequences',
                                'completed_frames': sum(report['frames'] for report in completed),
                                'last_completed': completed[-1]['sequence'] if completed else None,
                                'last_completed_at': completed[-1]['completed_at'] if completed else None,
                                'pending_sequences': len(pending), 'active': live, 'failures': failures,
                                'output_root': manifest['output_root']})
        if final:
            break
        time.sleep(3)
    render_catalog(manifest, completed)
    print(f'{timestamp()} {phase.upper()} done={len(completed)}/{len(manifest["sequences"])} failures={len(failures)}', flush=True)
    return 1 if failures else 0


def smoke_manifest(manifest_path, max_frames, resume=False):
    manifest = read_json(manifest_path)
    root = Path(manifest['output_root']) / '_smoke'
    state_path = Path(manifest['run_dir']) / 'smoke_progress.json'
    if state_path.exists() and not resume:
        raise ValueError('Smoke state exists: explicit --resume is required')
    selected = [next(item for item in manifest['sequences'] if item['dataset'] == dataset)
                for dataset in ('NPS', 'AOT', 'ARD')]
    started = timestamp()
    reports = []
    for sequence in selected:
        progress_path = root / sequence['id'] / 'progress.json'
        state = {'phase': 'running', 'pid': os.getpid(), 'started_at': started, 'updated_at': timestamp(),
                 'done': len(reports), 'total': len(selected), 'unit': 'sample_sequences',
                 'active_sequence': sequence['id'], 'worker_progress_path': str(progress_path)}
        write_json(state_path, state)
        report_path = root / sequence['id'] / 'report.json'
        existing = read_json(report_path) if report_path.exists() else None
        if resume and existing and existing.get('status') == 'completed':
            if existing['frames'] != min(existing['source_frames'], max_frames):
                raise ValueError('Smoke frame limit changed; use a fresh validation inventory')
            reports.append(existing)
        else:
            reports.append(process_sequence(manifest_path, sequence['id'], max_frames, root))
    write_json(state_path, dict(state, phase='completed', done=len(reports), updated_at=timestamp(),
                               active_sequence=None, worker_progress_path=None))
    write_json(root / 'smoke_summary.json', {'preview_only': True, 'reports': reports})
    print(f'{timestamp()} SMOKE_COMPLETED {len(reports)}/{len(selected)}', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Generate pinned, resumable AOT/NPS/ARD motion-comparison videos')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('inventory')
    prepare.add_argument('--run-dir', type=Path, required=True)
    prepare.add_argument('--output-root', type=Path, required=True)
    prepare.add_argument('--workers', type=int, default=2)
    prepare.add_argument('--opencv-threads', type=int, default=4)
    run = commands.add_parser('run')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--resume', action='store_true')
    worker = commands.add_parser('process')
    worker.add_argument('--manifest', type=Path, required=True)
    worker.add_argument('--sequence', required=True)
    worker.add_argument('--max-frames', type=int, default=0)
    worker.add_argument('--preview-root', type=Path)
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--manifest', type=Path, required=True)
    smoke.add_argument('--max-frames', type=int, default=90)
    smoke.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args.run_dir, args.output_root, args.workers, args.opencv_threads)
    elif args.command == 'process':
        process_sequence(args.manifest, args.sequence, args.max_frames, args.preview_root)
    elif args.command == 'smoke':
        if args.max_frames < 31:
            raise ValueError('Smoke test needs at least 31 frames to sample a nonboundary frame')
        return smoke_manifest(args.manifest, args.max_frames, args.resume)
    else:
        return run_manifest(args.manifest, args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
