#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="runs/window_accuracy/papers/edtc_antiuav600"
run_id="edtc_antiuav600"
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
summary_file="$out_abs/summary.json"
metrics_file="$out_abs/per_frame_window_metrics.csv"
worst_file="$out_abs/worst_windows.csv"
segments_file="$out_abs/low_accuracy_segments.csv"
tracker_log="$out_abs/edtc_tracker.log"

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
for key in ("pid", "start_time", "stdout_log", "stderr_log", "output_root", "results_dir", "total_sequences", "command", "launch_method", "launch_label", "launch_script", "exit_status_file"):
    print(data.get(key, ""))
PY
)

pid="${meta_values[0]}"
start_time="${meta_values[1]}"
stdout="${meta_values[2]}"
stderr="${meta_values[3]}"
output_root="${meta_values[4]}"
results_dir="${meta_values[5]}"
total_sequences="${meta_values[6]:-0}"
command="${meta_values[7]}"
launch_method="${meta_values[8]:-}"
launch_label="${meta_values[9]:-}"
launch_script="${meta_values[10]:-}"
exit_status_file="${meta_values[11]:-}"

pid_value=""
[[ -f "$pid_file" ]] && pid_value="$(head -n 1 "$pid_file" || true)"
[[ -z "$pid_value" ]] && pid_value="$pid"

running="0"
if [[ "$pid_value" =~ ^[0-9]+$ ]] && kill -0 "$pid_value" 2>/dev/null; then
  running="1"
  echo "RUNNING"
else
  echo "NOT RUNNING"
fi

result_count=0
latest_result=""
if [[ -d "$results_dir" ]]; then
  result_count="$(find "$results_dir" -type f -name '*.txt' ! -name '*_time.txt' ! -name '*_all_*' | wc -l | tr -d ' ')"
  latest_result="$(find "$results_dir" -type f -name '*.txt' ! -name '*_time.txt' ! -name '*_all_*' -print0 | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"
fi

done="$result_count"
last_unit="tracker_results=$result_count"
if [[ -f "$summary_file" ]]; then
  read -r videos frames < <(python3 - "$summary_file" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(data.get("videos", ""), data.get("frames", ""))
PY
)
  done="${total_sequences:-0}"
  last_unit="curves_complete videos=$videos frames=$frames"
elif [[ -n "$latest_result" ]]; then
  last_unit="tracker_result=$(basename "$latest_result")"
elif [[ -f "$tracker_log" ]]; then
  last_unit="$(awk 'NF {line=$0} END {print line}' "$tracker_log" || true)"
fi

echo "done/total: $done/${total_sequences:-0}"
echo "pid: $pid_value"
if [[ "$running" == "1" ]]; then
  echo "pid_start: $(ps -p "$pid_value" -o lstart= | sed 's/^ *//')"
  echo "pid_command_line: $(ps -p "$pid_value" -o command= | sed 's/^ *//')"
  descendant_processes="$(python3 - "$pid_value" <<'PY'
import subprocess
import sys
root = sys.argv[1]
rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pcpu=,pmem=,etime=,command="], text=True).splitlines()
children = {}
commands = {}
for row in rows:
    parts = row.strip().split(None, 5)
    if len(parts) < 6:
        continue
    pid, ppid, pcpu, pmem, etime, command = parts
    children.setdefault(ppid, []).append(pid)
    commands[pid] = (ppid, pcpu, pmem, etime, command)
todo = list(children.get(root, []))
seen = set()
while todo:
    pid = todo.pop(0)
    if pid in seen:
        continue
    seen.add(pid)
    ppid, pcpu, pmem, etime, command = commands.get(pid, ("", "", "", "", ""))
    print(f"{pid} ppid={ppid} cpu={pcpu}% mem={pmem}% elapsed={etime} {command}")
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
echo "results_dir: $results_dir"
next_sequence="$(python3 - "$command" "$results_dir" <<'PY'
import shlex
import sys
from pathlib import Path

command = sys.argv[1]
results_dir = Path(sys.argv[2])
try:
    parts = shlex.split(command)
except ValueError:
    parts = command.split()

dataset_root = None
sequence_filter = None
for index, part in enumerate(parts):
    if part == "--dataset-root" and index + 1 < len(parts):
        dataset_root = Path(parts[index + 1])
    elif part == "--sequence" and index + 1 < len(parts):
        sequence_filter = parts[index + 1]

if dataset_root is None or not (dataset_root / "list.txt").is_file():
    raise SystemExit(0)

names = [line.strip() for line in (dataset_root / "list.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
if sequence_filter:
    try:
        sequence_index = int(sequence_filter)
        candidates = [names[sequence_index]] if 0 <= sequence_index < len(names) else []
    except ValueError:
        candidates = [sequence_filter]
else:
    candidates = names

completed = set()
if results_dir.is_dir():
    for path in results_dir.glob("*.txt"):
        name = path.name
        if name.endswith("_time.txt") or "_all_" in name:
            continue
        completed.add(path.stem)

for idx, name in enumerate(candidates):
    if name in completed:
        continue
    seq_dir = dataset_root / name
    frame_count = 0
    if seq_dir.is_dir():
        frame_count = sum(1 for path in seq_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    print(f"index={idx} name={name} frames={frame_count}")
    break
PY
)"
[[ -n "$next_sequence" ]] && echo "next_expected_sequence: $next_sequence"

latest_output="$(python3 - "$stdout" "$stderr" "$tracker_log" "$summary_file" "$metrics_file" "$worst_file" "$segments_file" "$latest_result" <<'PY'
import os, sys
paths = [p for p in sys.argv[1:] if p]
paths = [p for p in paths if os.path.isfile(p)]
if paths:
    print(max(paths, key=lambda p: os.path.getmtime(p)))
PY
)"
if [[ -n "$latest_output" ]]; then
  latest_output_epoch="$(python3 - "$latest_output" <<'PY'
import os
import sys
print(int(os.path.getmtime(sys.argv[1])))
PY
)"
  if stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$latest_output" >/dev/null 2>&1; then
    echo "last_output_timestamp: $(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$latest_output")"
  else
    echo "last_output_timestamp: $(stat -c '%y' "$latest_output" | cut -d. -f1)"
  fi
  if [[ "$running" == "1" && "$latest_output_epoch" =~ ^[0-9]+$ ]]; then
    echo "last_output_age_seconds: $(( $(date +%s) - latest_output_epoch ))"
  fi
  echo "last_output_path: $latest_output"
else
  echo "last_output_timestamp: none"
fi

if [[ "$running" == "1" && "$command" == *"--device mps"* && "$(uname -s)" == "Darwin" ]] && command -v ioreg >/dev/null 2>&1; then
  gpu_signal=""
  gpu_raw="$(mktemp "${TMPDIR:-/tmp}/urap_agx.XXXXXX")"
  if ioreg -r -c AGXAccelerator -d 1 >"$gpu_raw" 2>/dev/null; then
    gpu_signal="$(python3 - "$gpu_raw" <<'PY'
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
values = []
for key in ("Device Utilization %", "Renderer Utilization %", "Tiler Utilization %", "In use system memory", "Alloc system memory"):
    match = re.search(rf'"{re.escape(key)}"=([0-9]+)', text)
    if match:
        value = int(match.group(1))
        if "memory" in key.lower():
            values.append(f"{key}={round(value / (1024 ** 3), 2)}GiB")
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
[[ -f "$tracker_log" ]] && echo "tracker_log: $tracker_log"
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

if [[ -f "$stdout" ]]; then
  echo
  echo "== stdout tail =="
  tail -n "$tail_lines" "$stdout"
fi
if [[ -f "$tracker_log" ]]; then
  echo
  echo "== tracker log tail =="
  tail -n "$tail_lines" "$tracker_log"
fi
if [[ -f "$stderr" ]]; then
  echo
  echo "== stderr tail =="
  tail -n "$tail_lines" "$stderr"
fi
