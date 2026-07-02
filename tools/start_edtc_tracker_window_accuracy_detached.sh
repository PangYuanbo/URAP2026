#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exe="python3"
dataset_root=""
tracker_model=""
yolo_weights=""
yolo_data=""
out="runs/window_accuracy/papers/edtc_antiuav600"
results_dir=""
skip_track="0"
config_name="urap_window_accuracy"
fps="30"
window_seconds="3"
iou="0.5"
threads="0"
num_gpus="1"
device="cpu"
search_area_scale="4.55"
sequence=""
run_id="edtc_antiuav600"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="$2"; shift 2 ;;
    --python) python_exe="$2"; shift 2 ;;
    --dataset-root) dataset_root="$2"; shift 2 ;;
    --tracker-model) tracker_model="$2"; shift 2 ;;
    --yolo-weights) yolo_weights="$2"; shift 2 ;;
    --yolo-data) yolo_data="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --skip-track) skip_track="1"; shift ;;
    --config-name) config_name="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --window-seconds) window_seconds="$2"; shift 2 ;;
    --iou) iou="$2"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --num-gpus) num_gpus="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --search-area-scale) search_area_scale="$2"; shift 2 ;;
    --sequence) sequence="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$dataset_root" ]]; then
  echo "--dataset-root is required" >&2
  exit 2
fi
if [[ "$skip_track" == "1" && -z "$results_dir" ]]; then
  echo "--results-dir is required with --skip-track" >&2
  exit 2
fi
if [[ "$skip_track" != "1" && ( -z "$tracker_model" || -z "$yolo_weights" || -z "$yolo_data" ) ]]; then
  echo "--tracker-model, --yolo-weights, and --yolo-data are required unless --skip-track is set" >&2
  exit 2
fi

runner="$repo_root/tools/run_edtc_tracker_window_accuracy.py"
if [[ ! -f "$runner" ]]; then
  echo "Runner not found: $runner" >&2
  exit 2
fi

if [[ "$python_exe" == */* ]]; then
  if [[ "$python_exe" = /* ]]; then python_launch="$python_exe"; else python_launch="$repo_root/$python_exe"; fi
else
  python_launch="$(command -v "$python_exe" || true)"
fi
if [[ -z "$python_launch" || ! -x "$python_launch" ]]; then
  echo "Python executable not found or not executable: $python_exe" >&2
  exit 2
fi

if [[ "$out" = /* ]]; then out_abs="$out"; else out_abs="$repo_root/$out"; fi
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

dataset_abs="$dataset_root"
[[ "$dataset_abs" != /* ]] && dataset_abs="$repo_root/$dataset_abs"
if [[ ! -f "$dataset_abs/list.txt" ]]; then
  echo "AntiUAV list.txt not found: $dataset_abs/list.txt" >&2
  exit 2
fi

if [[ -n "$results_dir" ]]; then
  if [[ "$results_dir" = /* ]]; then results_abs="$results_dir"; else results_abs="$repo_root/$results_dir"; fi
else
  results_abs="$out_abs/tracking_results/uavtrack_eh/$config_name"
fi

total_sequences="$(python3 - "$dataset_abs/list.txt" "$sequence" <<'PY'
from pathlib import Path
import sys
names = [line.strip() for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
print(1 if sys.argv[2] else len(names))
PY
)"

ts="$(date +%Y%m%d_%H%M%S)"
stdout="$out_abs/logs/${run_id}_${ts}.out.log"
stderr="$out_abs/logs/${run_id}_${ts}.err.log"
launch_script="$out_abs/logs/${run_id}_${ts}.launch.sh"
exit_file="$out_abs/logs/${run_id}_${ts}.exit"
safe_run_id="$(printf '%s' "$run_id" | tr -c '[:alnum:]_.-' '-')"
launch_label="com.urap.edtc-window-accuracy.${safe_run_id}.${ts}"

args=(
  "$runner"
  --dataset-root "$dataset_abs"
  --out "$out_abs"
  --config-name "$config_name"
  --fps "$fps"
  --window-seconds "$window_seconds"
  --iou "$iou"
  --threads "$threads"
  --num-gpus "$num_gpus"
  --device "$device"
  --search-area-scale "$search_area_scale"
)
if [[ "$skip_track" == "1" ]]; then
  args+=(--skip-track --results-dir "$results_abs")
else
  args+=(--tracker-model "$tracker_model" --yolo-weights "$yolo_weights" --yolo-data "$yolo_data")
fi
[[ -n "$sequence" ]] && args+=(--sequence "$sequence")

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
  echo "done/total: 0/${total_sequences:-0}"
  echo "pid: none"
  echo "start_time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "last_completed_unit: launch failed before pid file was written"
  echo "stdout: $stdout"
  echo "stderr: $stderr"
  echo "launch_method: $launch_method"
  echo "launch_label: $launch_label"
  echo "launch_script: $launch_script"
  [[ -f "$stderr" ]] && tail -n 40 "$stderr"
  exit 1
fi

META_START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
META_PID="$pid" \
META_RUN_ID="$run_id" \
META_STDOUT="$stdout" \
META_STDERR="$stderr" \
META_OUTPUT_ROOT="$out_abs" \
META_RESULTS_DIR="$results_abs" \
META_TOTAL_SEQUENCES="${total_sequences:-0}" \
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
    "run_id": os.environ["META_RUN_ID"],
    "stdout_log": os.environ["META_STDOUT"],
    "stderr_log": os.environ["META_STDERR"],
    "output_root": os.environ["META_OUTPUT_ROOT"],
    "results_dir": os.environ["META_RESULTS_DIR"],
    "total_sequences": int(os.environ["META_TOTAL_SEQUENCES"] or 0),
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
  echo "done/total: 0/${total_sequences:-0}"
  echo "pid: $pid"
  echo "start_time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "last_completed_unit: launch exited immediately status=$status"
  echo "stdout: $stdout"
  echo "stderr: $stderr"
  echo "launch_method: $launch_method"
  echo "launch_label: $launch_label"
  echo "launch_script: $launch_script"
  echo "exit_status_file: $exit_file"
  [[ -f "$stdout" ]] && tail -n 40 "$stdout"
  [[ -f "$stderr" ]] && tail -n 40 "$stderr"
  exit 1
fi

echo "RUNNING"
echo "done/total: 0/${total_sequences:-0}"
echo "pid: $pid"
echo "start_time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "last_completed_unit: launched EDTC tracker/window-accuracy runner"
echo "stdout: $stdout"
echo "stderr: $stderr"
echo "launch_method: $launch_method"
echo "launch_label: $launch_label"
echo "launch_script: $launch_script"
echo "exit_status_file: $exit_file"
echo "output_root: $out_abs"
echo "results_dir: $results_abs"
