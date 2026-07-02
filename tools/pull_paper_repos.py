from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PaperRepo:
    name: str
    path: Path
    url: str
    branch: str | None = None
    requires_auth: bool = False
    gitlab_project_id: int | None = None
    gitlab_ref: str | None = None
    snapshot_skip_suffixes: tuple[str, ...] = ()


PAPER_REPOS = [
    PaperRepo("YOLOMG", Path("papers/YOLOMG"), "https://github.com/Irisky123/YOLOMG.git"),
    PaperRepo("TransVisDrone", Path("papers/TransVisDrone"), "https://github.com/tusharsangam/TransVisDrone.git"),
    PaperRepo("ESOD", Path("papers/ESOD"), "https://github.com/alibaba/esod.git"),
    PaperRepo("EDTC", Path("papers/EDTC"), "https://github.com/xuefeng-zhu5/EDTC.git"),
    PaperRepo(
        "Li_TETC_NPS",
        Path("papers/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking"),
        "https://github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking.git",
    ),
    PaperRepo(
        "Dogfight_Drone_Detection",
        Path("datasets/Drone-Detection"),
        "https://github.com/mwaseema/Drone-Detection.git",
    ),
    PaperRepo(
        "AICrowd_Winner_v022",
        Path("papers/AICrowd_AOT_Challenge_Winner/submission-v022/airborne-detection-starter-kit-submission-v022"),
        "https://gitlab.aicrowd.com/dmytro_poplavskiy/airborne-detection-starter-kit.git",
        branch="submission-v022",
        gitlab_project_id=4284,
        gitlab_ref="submission-v022",
        snapshot_skip_suffixes=(".pt", ".pth", ".pyc"),
    ),
]


def _run_git(args: list[str], cwd: Path, dry_run: bool) -> tuple[int, str]:
    cmd = ["git", *args]
    if dry_run:
        return 0, "dry-run: " + " ".join(cmd)
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return proc.returncode, proc.stdout.strip()


def _head(path: Path) -> str | None:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _remote(path: Path) -> str | None:
    proc = subprocess.run(["git", "remote", "get-url", "origin"], cwd=path, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _api_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "URAP2026-paper-pull/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _api_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "URAP2026-paper-pull/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def _gitlab_api_url(repo: PaperRepo, endpoint: str, params: dict[str, str | int] | None = None) -> str:
    if repo.gitlab_project_id is None:
        raise ValueError("gitlab_project_id is required for GitLab API snapshots")
    query = urllib.parse.urlencode(params or {})
    url = f"https://gitlab.aicrowd.com/api/v4/projects/{repo.gitlab_project_id}/{endpoint}"
    return f"{url}?{query}" if query else url


def _gitlab_tree(repo: PaperRepo) -> list[dict[str, Any]]:
    if not repo.gitlab_ref:
        raise ValueError("gitlab_ref is required for GitLab API snapshots")
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = _api_json(
            _gitlab_api_url(
                repo,
                "repository/tree",
                {"ref": repo.gitlab_ref, "recursive": "true", "per_page": 100, "page": page},
            )
        )
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out


def _gitlab_tag(repo: PaperRepo) -> dict[str, Any] | None:
    if not repo.gitlab_ref:
        return None
    try:
        return _api_json(_gitlab_api_url(repo, f"repository/tags/{urllib.parse.quote(repo.gitlab_ref, safe='')}"))
    except Exception:
        return None


def _sync_gitlab_api_snapshot(repo: PaperRepo, target: Path, dry_run: bool) -> dict[str, Any]:
    marker = target / ".urap_snapshot.json"
    if target.exists() and any(target.iterdir()) and not marker.is_file() and not (target / ".git").is_dir():
        return {"status": "nonempty_without_git"}
    if dry_run:
        return {
            "status": "would_snapshot",
            "head": None,
            "output": "would download source snapshot via GitLab API and skip configured binary weights",
        }

    tree = _gitlab_tree(repo)
    blobs = [item for item in tree if item.get("type") == "blob"]
    skipped = [
        item["path"]
        for item in blobs
        if any(str(item["path"]).endswith(suffix) for suffix in repo.snapshot_skip_suffixes)
    ]
    download = [item for item in blobs if item["path"] not in set(skipped)]
    tag = _gitlab_tag(repo)
    commit = (tag or {}).get("commit") or {}
    head = commit.get("short_id") or (str(commit.get("id", ""))[:8] if commit.get("id") else None)

    target.mkdir(parents=True, exist_ok=True)
    for item in download:
        rel = str(item["path"])
        raw_url = _gitlab_api_url(
            repo,
            f"repository/files/{urllib.parse.quote(rel, safe='')}/raw",
            {"ref": repo.gitlab_ref or "master"},
        )
        out_path = target / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(_api_bytes(raw_url))

    marker_payload = {
        "name": repo.name,
        "url": repo.url,
        "gitlab_project_id": repo.gitlab_project_id,
        "ref": repo.gitlab_ref,
        "commit": commit.get("id"),
        "short_id": head,
        "downloaded_files": len(download),
        "skipped_files": skipped,
        "note": "Code snapshot downloaded via GitLab API because unauthenticated git clone/archive is not available in this environment.",
    }
    marker.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")
    return {
        "status": "snapshotted",
        "head": head,
        "output": f"downloaded {len(download)} source files via GitLab API; skipped {len(skipped)} binary files",
        "download_files": len(download),
        "skipped_files": skipped,
    }


def sync_repo(repo: PaperRepo, root: Path, dry_run: bool, include_auth_required: bool) -> dict[str, Any]:
    target = root / repo.path
    report: dict[str, Any] = {
        "name": repo.name,
        "path": str(target),
        "url": repo.url,
        "branch": repo.branch,
        "requires_auth": repo.requires_auth,
    }
    if repo.requires_auth and not include_auth_required:
        report["status"] = "skipped_auth_required"
        return report

    if repo.gitlab_project_id is not None:
        report.update(_sync_gitlab_api_snapshot(repo, target, dry_run=dry_run))
        return report

    if (target / ".git").is_dir():
        remote = _remote(target)
        if remote and remote != repo.url:
            report.update({"status": "remote_mismatch", "remote": remote})
            return report
        code, output = _run_git(["pull", "--ff-only"], cwd=target, dry_run=dry_run)
        status = "would_update" if dry_run and code == 0 else "updated" if code == 0 else "failed"
        report.update({"status": status, "output": output, "head": _head(target) if not dry_run else None})
        return report

    if target.exists() and any(target.iterdir()):
        report["status"] = "nonempty_without_git"
        return report

    target.parent.mkdir(parents=True, exist_ok=True)
    clone_args = ["clone"]
    if repo.branch:
        clone_args.extend(["--branch", repo.branch, "--single-branch"])
    clone_args.extend([repo.url, str(target)])
    code, output = _run_git(clone_args, cwd=root, dry_run=dry_run)
    status = "would_clone" if dry_run and code == 0 else "cloned" if code == 0 else "failed"
    report.update({"status": status, "output": output, "head": _head(target) if code == 0 and not dry_run else None})
    if code != 0 and repo.requires_auth and "Username" in output:
        report["status"] = "auth_required"
    return report


def sync_all(root: Path, dry_run: bool = False, include_auth_required: bool = False, only: set[str] | None = None) -> dict[str, Any]:
    reports = []
    for repo in PAPER_REPOS:
        if only and repo.name not in only:
            continue
        try:
            reports.append(sync_repo(repo, root=root, dry_run=dry_run, include_auth_required=include_auth_required))
        except Exception as exc:
            reports.append(
                {
                    "name": repo.name,
                    "path": str(root / repo.path),
                    "url": repo.url,
                    "branch": repo.branch,
                    "requires_auth": repo.requires_auth,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "head": _head(root / repo.path) if (root / repo.path).exists() else None,
                }
            )
    return {
        "root": str(root),
        "dry_run": dry_run,
        "include_auth_required": include_auth_required,
        "repos": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone or fast-forward paper repositories used by URAP2026.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-auth-required", action="store_true", help="Attempt repos known to need credentials, such as AICrowd GitLab.")
    parser.add_argument("--only", action="append", default=[], help="Sync only this repo name. Can be repeated.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = sync_all(
        root=args.root.resolve(),
        dry_run=args.dry_run,
        include_auth_required=args.include_auth_required,
        only=set(args.only) if args.only else None,
    )
    for repo in report["repos"]:
        print(f"{repo['status']}: {repo['name']} -> {repo['path']}")
        if repo.get("head"):
            print(f"  head: {repo['head']}")
        if repo.get("output"):
            print(f"  {repo['output'].splitlines()[-1]}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if any(repo["status"] in {"failed", "remote_mismatch", "nonempty_without_git"} for repo in report["repos"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
