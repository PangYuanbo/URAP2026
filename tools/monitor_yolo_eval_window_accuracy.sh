#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="runs/window_accuracy/detached_yolo_eval"
run_id="paper_window_accuracy"
tail_lines="80"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --tail-lines) tail_lines="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$out" = /* ]]; then out_abs="$out"; else out_abs="$repo_root/$out"; fi
pid_file="$out_abs/${run_id}_pid.txt"
meta_file="$out_abs/${run_id}_meta.json"

if [[ ! -f "$meta_file" ]]; then
  echo "NOT RUNNING"
  echo "meta_not_found: $meta_file"
  exit 0
fi

meta_values=()
while IFS= read -r line; do
  meta_values+=("$line")
done < <(python3 - "$meta_file" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
for key in ("pid", "start_time", "stdout_log", "stderr_log", "eval_log", "output_root", "prediction_labels_dir", "total_frames", "command", "launch_method", "launch_label", "launch_script", "exit_status_file"):
    print(data.get(key, ""))
PY
)
pid="${meta_values[0]}"
start_time="${meta_values[1]}"
stdout="${meta_values[2]}"
stderr="${meta_values[3]}"
eval_log="${meta_values[4]}"
output_root="${meta_values[5]}"
labels_dir="${meta_values[6]}"
total_frames="${meta_values[7]:-0}"
command="${meta_values[8]}"
launch_method="${meta_values[9]:-}"
launch_label="${meta_values[10]:-}"
launch_script="${meta_values[11]:-}"
exit_status_file="${meta_values[12]:-}"

pid_value=""
[[ -f "$pid_file" ]] && pid_value="$(head -n 1 "$pid_file" || true)"
if [[ -z "$pid_value" ]]; then pid_value="$pid"; fi

running="0"
if [[ "$pid_value" =~ ^[0-9]+$ ]] && kill -0 "$pid_value" 2>/dev/null; then
  running="1"
  echo "RUNNING"
else
  echo "NOT RUNNING"
fi

label_count=0
latest_label=""
if [[ -d "$labels_dir" ]]; then
  label_count="$(find "$labels_dir" -type f -name '*.txt' | wc -l | tr -d ' ')"
  latest_label="$(find "$labels_dir" -type f -name '*.txt' -print0 | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
fi

eval_progress_images=""
eval_progress_total=""
if [[ -f "$eval_log" ]]; then
  read -r eval_progress_images eval_progress_total < <(python3 - "$eval_log" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", text)
if matches:
    print(*matches[-1])
PY
)
fi
if [[ -n "$eval_progress_total" && ( -z "${total_frames:-}" || "$total_frames" == "0" ) ]]; then
  total_frames="$eval_progress_total"
fi

summary_file="$out_abs/summary.json"
metrics_file="$out_abs/per_frame_window_metrics.csv"
worst_file="$out_abs/worst_windows.csv"
segments_file="$out_abs/low_accuracy_segments.csv"

done="$label_count"
last_unit="prediction_labels=$label_count"
if [[ -n "$eval_progress_images" ]]; then
  done="$eval_progress_images"
  last_unit="eval_progress_images=$eval_progress_images prediction_labels=$label_count"
fi
if [[ -f "$summary_file" ]]; then
  read -r videos frames < <(python3 - "$summary_file" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(data.get("videos", ""), data.get("frames", ""))
PY
)
  done="${total_frames:-0}"
  [[ "$done" == "0" ]] && done="$frames"
  last_unit="curves_complete videos=$videos frames=$frames"
elif [[ "$label_count" == "0" && -f "$eval_log" ]]; then
  last_unit="$(tail -n 1 "$eval_log" || true)"
fi

echo "done/total: $done/${total_frames:-0}"
echo "pid: $pid_value"
if [[ "$running" == "1" ]]; then
  echo "pid_start: $(ps -p "$pid_value" -o lstart= | sed 's/^ *//')"
  echo "pid_command_line: $(ps -p "$pid_value" -o command= | sed 's/^ *//')"
  descendant_processes="$(python3 - "$pid_value" <<'PY'
import subprocess
import sys
root = sys.argv[1]
rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True).splitlines()
children = {}
commands = {}
for row in rows:
    parts = row.strip().split(None, 2)
    if len(parts) < 3:
        continue
    pid, ppid, command = parts
    children.setdefault(ppid, []).append(pid)
    commands[pid] = command
todo = list(children.get(root, []))
seen = set()
while todo:
    pid = todo.pop(0)
    if pid in seen:
        continue
    seen.add(pid)
    print(f"{pid} {commands.get(pid, '')}")
    todo.extend(children.get(pid, []))
PY
)"
  if [[ -n "$descendant_processes" ]]; then
    echo "descendant_processes:"
    echo "$descendant_processes"
  fi
fi
echo "start_time: $start_time"
echo "last_completed_unit: $last_unit"
echo "output_root: $out_abs"
echo "prediction_labels_dir: $labels_dir"
echo "prediction_label_count: $label_count"
if [[ -n "$eval_progress_images" ]]; then
  echo "eval_progress_images: $eval_progress_images"
  echo "eval_progress_total: ${eval_progress_total:-0}"
fi

latest_output="$(python3 - "$stdout" "$stderr" "$eval_log" "$summary_file" "$metrics_file" "$worst_file" "$segments_file" "$latest_label" <<'PY'
import os, sys
paths = [p for p in sys.argv[1:] if p]
paths = [p for p in paths if os.path.isfile(p)]
if paths:
    print(max(paths, key=lambda p: os.path.getmtime(p)))
PY
)"
if [[ -n "$latest_output" ]]; then
  echo "last_output_timestamp: $(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$latest_output")"
  echo "last_output_path: $latest_output"
else
  echo "last_output_timestamp: none"
fi

if [[ "$running" == "1" && "$(uname -s)" == "Darwin" ]] && command -v ioreg >/dev/null 2>&1; then
  gpu_signal=""
  gpu_raw="$(mktemp "${TMPDIR:-/tmp}/urap_agx.XXXXXX")"
  if ioreg -r -c AGXAccelerator -d 1 >"$gpu_raw" 2>/dev/null; then
    gpu_signal="$(python3 - "$gpu_raw" <<'PY'
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
keys = [
    "Device Utilization %",
    "Renderer Utilization %",
    "Tiler Utilization %",
    "In use system memory",
    "Alloc system memory",
]
values = []
for key in keys:
    match = re.search(rf'"{re.escape(key)}"=([0-9]+)', text)
    if match:
        value = int(match.group(1))
        if "memory" in key.lower():
            value = round(value / (1024 ** 3), 2)
            values.append(f"{key}={value}GiB")
        else:
            values.append(f"{key}={value}%")
pid_match = re.search(r'"fLastSubmissionPID"=([0-9]+)', text)
if pid_match:
    values.append(f"last_submission_pid={pid_match.group(1)}")
if values:
    print(", ".join(values))
PY
)"
  fi
  rm -f "$gpu_raw"
  [[ -n "$gpu_signal" ]] && echo "gpu_signal: $gpu_signal"
fi

echo "stdout: $stdout"
echo "stderr: $stderr"
echo "eval_log: $eval_log"
[[ -n "$launch_method" ]] && echo "launch_method: $launch_method"
[[ -n "$launch_label" ]] && echo "launch_label: $launch_label"
[[ -n "$launch_script" ]] && echo "launch_script: $launch_script"
[[ -n "$exit_status_file" ]] && echo "exit_status_file: $exit_status_file"
if [[ -n "$exit_status_file" && -f "$exit_status_file" ]]; then
  echo "exit_status: $(head -n 1 "$exit_status_file" || true)"
fi
echo "command: $command"
[[ -f "$summary_file" ]] && echo "summary: $summary_file"
[[ -f "$metrics_file" ]] && echo "per_frame_csv: $metrics_file"
[[ -f "$worst_file" ]] && echo "worst_windows_csv: $worst_file"
[[ -f "$segments_file" ]] && echo "low_accuracy_segments_csv: $segments_file"

if [[ -f "$eval_log" ]]; then
  echo
  echo "== eval log tail =="
  tail -n "$tail_lines" "$eval_log"
fi
if [[ -f "$stdout" ]]; then
  echo
  echo "== stdout tail =="
  tail -n "$tail_lines" "$stdout"
fi
if [[ -f "$stderr" ]]; then
  echo
  echo "== stderr tail =="
  tail -n "$tail_lines" "$stderr"
fi
