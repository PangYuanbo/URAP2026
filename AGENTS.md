# URAP Agent Rules (Non-Negotiable)

This file records **hard constraints** for the coding agent working in this repo.

## 1) No Cheating / No Misrepresentation

- **Never claim a long-running job is “still running” unless verified**.
- Verification must include at least:
  - OS-level process exists (PID + command line matches), and
  - progress evidence (new output files / updated timestamps / log tail moving), and
  - if GPU-related, a quick GPU signal (utilization or memory use) consistent with the job.
- **Never silently restart** a stopped job. If a job was not running, say explicitly:
  - it was stopped,
  - when the stop was observed,
  - what progress is already done,
  - when the restart happened,
  - new PID / log file path.

## 2) Isolate Long-Running Jobs (Detachable Runs)

Problem: interactive tool/PTY sessions may terminate, which can kill child processes and pause the pipeline.

Rules:
- Any job expected to run > 5 minutes must be launched in an **isolated / detached** way on Windows:
  - `Start-Process` with `-RedirectStandardOutput/-RedirectStandardError`
  - write a **PID file** and **log paths** into the repo (so status can be checked without the agent session)
- Do **not** rely on “tool session id” as the source of truth. Always track by **PID**.

Repo convention:
- Long runs must have:
  - `tools/start_*_detached.ps1` (start/resume; writes PID + logs)
  - `tools/monitor_*_*.ps1` (reports PID alive + progress counters)
  - optionally `tools/stop_*_*.ps1` (clean stop/kill)

## 3) Status Reporting Template

When reporting a job status, include:
- `done/total`
- PID(s) and start time
- last output timestamp / last completed unit (flight/clip/epoch)
- log file path(s)
- if the job is **not running**, say **NOT RUNNING** (no “probably running” language)

