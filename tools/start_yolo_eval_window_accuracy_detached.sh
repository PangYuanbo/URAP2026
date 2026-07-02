#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exe="python3"
method="yolomg"
paper_repo=""
data=""
weights=""
gt=""
gt_format="yolo-dir"
out="runs/window_accuracy/detached_yolo_eval"
fps="30"
window_seconds="3"
match_iou="0.5"
score_threshold="0.25"
gt_frame_offset="0"
pred_frame_offset="0"
frame_manifest=""
frame_manifest_format=""
frame_manifest_offset="0"
img_width=""
img_height=""
name="window_accuracy_eval"
project=""
task="val"
img="1280"
batch_size="1"
device="cpu"
num_frames="5"
half="0"
augment="0"
skip_eval="0"
pred_labels_dir=""
run_id="paper_window_accuracy"
extra_eval_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="$2"; shift 2 ;;
    --python) python_exe="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --paper-repo) paper_repo="$2"; shift 2 ;;
    --data) data="$2"; shift 2 ;;
    --weights) weights="$2"; shift 2 ;;
    --gt) gt="$2"; shift 2 ;;
    --gt-format) gt_format="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --window-seconds) window_seconds="$2"; shift 2 ;;
    --match-iou) match_iou="$2"; shift 2 ;;
    --score-threshold) score_threshold="$2"; shift 2 ;;
    --gt-frame-offset) gt_frame_offset="$2"; shift 2 ;;
    --pred-frame-offset) pred_frame_offset="$2"; shift 2 ;;
    --frame-manifest) frame_manifest="$2"; shift 2 ;;
    --frame-manifest-format) frame_manifest_format="$2"; shift 2 ;;
    --frame-manifest-offset) frame_manifest_offset="$2"; shift 2 ;;
    --img-width) img_width="$2"; shift 2 ;;
    --img-height) img_height="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --project) project="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --img) img="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --num-frames) num_frames="$2"; shift 2 ;;
    --half) half="1"; shift ;;
    --augment) augment="1"; shift ;;
    --skip-eval) skip_eval="1"; shift ;;
    --pred-labels-dir) pred_labels_dir="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --extra-eval-arg) extra_eval_args+=("$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$gt" ]]; then
  echo "--gt is required" >&2
  exit 2
fi
if [[ "$skip_eval" != "1" && ( -z "$data" || -z "$weights" ) ]]; then
  echo "--data and --weights are required unless --skip-eval is set" >&2
  exit 2
fi
if [[ "$skip_eval" == "1" && -z "$pred_labels_dir" ]]; then
  echo "--pred-labels-dir is required when --skip-eval is set" >&2
  exit 2
fi
if [[ -n "$img_width" && -z "$img_height" || -z "$img_width" && -n "$img_height" ]]; then
  echo "--img-width and --img-height must be provided together" >&2
  exit 2
fi

runner="$repo_root/tools/run_yolo_eval_window_accuracy.py"
if [[ ! -f "$runner" ]]; then
  echo "Runner not found: $runner" >&2
  exit 2
fi

if [[ "$python_exe" == */* ]]; then
  if [[ "$python_exe" = /* ]]; then
    python_launch="$python_exe"
  else
    python_launch="$repo_root/$python_exe"
  fi
else
  python_launch="$(command -v "$python_exe" || true)"
fi
if [[ -z "$python_launch" || ! -x "$python_launch" ]]; then
  echo "Python executable not found or not executable: $python_exe" >&2
  exit 2
fi

if [[ "$out" = /* ]]; then out_abs="$out"; else out_abs="$repo_root/$out"; fi
if [[ -n "$project" ]]; then
  if [[ "$project" = /* ]]; then project_abs="$project"; else project_abs="$repo_root/$project"; fi
else
  project_abs="$out_abs/eval"
fi
if [[ -n "$pred_labels_dir" ]]; then
  if [[ "$pred_labels_dir" = /* ]]; then labels_dir="$pred_labels_dir"; else labels_dir="$repo_root/$pred_labels_dir"; fi
else
  labels_dir="$project_abs/$name/labels"
fi

mkdir -p "$out_abs/logs"
pid_file="$out_abs/${run_id}_pid.txt"
meta_file="$out_abs/${run_id}_meta.json"

if [[ -f "$pid_file" ]]; then
  existing_pid="$(head -n 1 "$pid_file" || true)"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "RUNNING pid=$existing_pid"
    [[ -f "$meta_file" ]] && sed -n '1,120p' "$meta_file"
    exit 0
  fi
fi

ts="$(date +%Y%m%d_%H%M%S)"
stdout="$out_abs/logs/${run_id}_${ts}.out.log"
stderr="$out_abs/logs/${run_id}_${ts}.err.log"
launch_script="$out_abs/logs/${run_id}_${ts}.launch.sh"
exit_file="$out_abs/logs/${run_id}_${ts}.exit"
eval_log="$out_abs/eval.log"
safe_run_id="$(printf '%s' "$run_id" | tr -c '[:alnum:]_.-' '-')"
launch_label="com.urap.window-accuracy.${safe_run_id}.${ts}"

args=(
  "$runner"
  --method "$method"
  --gt "$gt"
  --gt-format "$gt_format"
  --out "$out_abs"
  --project "$project_abs"
  --name "$name"
  --task "$task"
  --img "$img"
  --batch-size "$batch_size"
  --device "$device"
  --num-frames "$num_frames"
  --fps "$fps"
  --window-seconds "$window_seconds"
  --match-iou "$match_iou"
  --score-threshold "$score_threshold"
  --gt-frame-offset "$gt_frame_offset"
  --pred-frame-offset "$pred_frame_offset"
  --frame-manifest-offset "$frame_manifest_offset"
)
[[ -n "$paper_repo" ]] && args+=(--repo "$paper_repo")
[[ -n "$data" ]] && args+=(--data "$data")
[[ -n "$weights" ]] && args+=(--weights "$weights")
[[ "$skip_eval" == "1" ]] && args+=(--skip-eval --pred-labels-dir "$labels_dir")
[[ -n "$frame_manifest" ]] && args+=(--frame-manifest "$frame_manifest")
[[ -n "$frame_manifest_format" ]] && args+=(--frame-manifest-format "$frame_manifest_format")
[[ "$half" == "1" ]] && args+=(--half)
[[ "$augment" == "1" ]] && args+=(--augment)
[[ -n "$img_width" ]] && args+=(--img-width "$img_width" --img-height "$img_height")
if ((${#extra_eval_args[@]})); then
  for extra in "${extra_eval_args[@]}"; do
    [[ -n "$extra" ]] && args+=(--extra-eval-arg "$extra")
  done
fi

total_frames=0
if [[ -n "$frame_manifest" ]]; then
  manifest_path="$frame_manifest"
  [[ "$manifest_path" != /* ]] && manifest_path="$repo_root/$manifest_path"
  if [[ -d "$manifest_path" ]]; then
    total_frames="$(find "$manifest_path" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l | tr -d ' ')"
  fi
fi

command_line="$(printf '%q ' "$python_launch" "${args[@]}")"

rm -f "$pid_file" "$exit_file" "$stdout" "$stderr"

launch_code="$(
  printf '#!/usr/bin/env bash\n'
  printf 'set +e\n'
  printf 'cd %q || exit 127\n' "$repo_root"
  printf 'echo "$$" > %q\n' "$pid_file"
  printf 'echo "wrapper_start: $(date '\''+%%Y-%%m-%%d %%H:%%M:%%S'\'')" >> %q\n' "$stderr"
  printf 'echo "command: %s" >> %q\n' "$command_line" "$stderr"
  printf 'cmd=( '
  for value in "$python_launch" "${args[@]}"; do
    printf '%q ' "$value"
  done
  printf ')\n'
  printf '"${cmd[@]}" >>%q 2>>%q\n' "$stdout" "$stderr"
  printf 'status=$?\n'
  printf 'echo "$status" > %q\n' "$exit_file"
  printf 'echo "wrapper_exit: $status $(date '\''+%%Y-%%m-%%d %%H:%%M:%%S'\'')" >> %q\n' "$stderr"
  printf 'exit "$status"\n'
)"
printf '%s\n' "$launch_code" >"$launch_script"
chmod +x "$launch_script"

launch_method="nohup"
if [[ "$(uname -s)" == "Darwin" ]] && command -v launchctl >/dev/null 2>&1; then
  launch_method="launchctl"
  launchctl submit -l "$launch_label" -- /bin/bash -c "$launch_code"
else
  nohup bash "$launch_script" >/dev/null 2>&1 < /dev/null &
fi

pid=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  [[ -s "$pid_file" ]] && pid="$(head -n 1 "$pid_file" || true)" && break
  sleep 0.2
done
if [[ -z "$pid" ]]; then
  echo "NOT RUNNING"
  echo "done/total: 0/${total_frames:-0}"
  echo "pid: none"
  echo "start_time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "last_completed_unit: launch failed before pid file was written"
  echo "stdout: $stdout"
  echo "stderr: $stderr"
  echo "eval_log: $eval_log"
  echo "launch_method: $launch_method"
  echo "launch_label: $launch_label"
  echo "launch_script: $launch_script"
  [[ -f "$stderr" ]] && tail -n 40 "$stderr"
  exit 1
fi

META_START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
META_PID="$pid" \
META_METHOD="$method" \
META_RUN_ID="$run_id" \
META_STDOUT="$stdout" \
META_STDERR="$stderr" \
META_EVAL_LOG="$eval_log" \
META_OUTPUT_ROOT="$out_abs" \
META_PROJECT="$project_abs" \
META_LABELS="$labels_dir" \
META_TOTAL_FRAMES="${total_frames:-0}" \
META_RUNNER="$runner" \
META_PYTHON="$python_launch" \
META_LAUNCH_METHOD="$launch_method" \
META_LAUNCH_LABEL="$launch_label" \
META_LAUNCH_SCRIPT="$launch_script" \
META_EXIT_FILE="$exit_file" \
META_COMMAND="$command_line" \
python3 - "$meta_file" <<'PY'
import json
import os
from pathlib import Path
meta = {
    "start_time": os.environ["META_START_TIME"],
    "pid": int(os.environ["META_PID"]),
    "method": os.environ["META_METHOD"],
    "run_id": os.environ["META_RUN_ID"],
    "stdout_log": os.environ["META_STDOUT"],
    "stderr_log": os.environ["META_STDERR"],
    "eval_log": os.environ["META_EVAL_LOG"],
    "output_root": os.environ["META_OUTPUT_ROOT"],
    "project": os.environ["META_PROJECT"],
    "prediction_labels_dir": os.environ["META_LABELS"],
    "total_frames": int(os.environ["META_TOTAL_FRAMES"] or 0),
    "runner": os.environ["META_RUNNER"],
    "python": os.environ["META_PYTHON"],
    "launch_method": os.environ["META_LAUNCH_METHOD"],
    "launch_label": os.environ["META_LAUNCH_LABEL"],
    "launch_script": os.environ["META_LAUNCH_SCRIPT"],
    "exit_status_file": os.environ["META_EXIT_FILE"],
    "command": os.environ["META_COMMAND"],
}
Path(__import__("sys").argv[1]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
PY

sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  status="unknown"
  [[ -f "$exit_file" ]] && status="$(head -n 1 "$exit_file" || true)"
  echo "NOT RUNNING"
  echo "done/total: 0/${total_frames:-0}"
  echo "pid: $pid"
  echo "start_time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "last_completed_unit: launch exited immediately status=$status"
  echo "stdout: $stdout"
  echo "stderr: $stderr"
  echo "eval_log: $eval_log"
  echo "launch_method: $launch_method"
  echo "launch_label: $launch_label"
  echo "launch_script: $launch_script"
  echo "exit_status_file: $exit_file"
  if [[ -f "$stdout" ]]; then
    echo
    echo "== stdout tail =="
    tail -n 40 "$stdout"
  fi
  if [[ -f "$stderr" ]]; then
    echo
    echo "== stderr tail =="
    tail -n 40 "$stderr"
  fi
  exit 1
fi

echo "RUNNING"
echo "done/total: 0/${total_frames:-0}"
echo "pid: $pid"
echo "start_time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "last_completed_unit: launched paper eval/window-accuracy runner"
echo "stdout: $stdout"
echo "stderr: $stderr"
echo "eval_log: $eval_log"
echo "launch_method: $launch_method"
echo "launch_label: $launch_label"
echo "launch_script: $launch_script"
echo "exit_status_file: $exit_file"
echo "output_root: $out_abs"
echo "prediction_labels_dir: $labels_dir"
