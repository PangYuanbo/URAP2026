from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


REPO = Path(r"C:\Users\aaron\Desktop\URAP")
DEFAULT_LOCAL_SITE = REPO / r"artifacts\yolomg_motion_process_site"
DEFAULT_OUT = REPO / r"artifacts\yolomg_motion_process_drive_external_site"
DEFAULT_DB = Path(r"C:\Users\aaron\AppData\Local\Google\DriveFS\114123953747902308024\metadata_sqlite_db")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an external-link YOLOMG gallery from Google DriveFS metadata.")
    p.add_argument("--local-site", default=str(DEFAULT_LOCAL_SITE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--drivefs-db", default=str(DEFAULT_DB))
    p.add_argument("--drive-root-title", default="yolomg_motion_process_site_syncfix")
    return p.parse_args()


def connect_readonly(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)


def find_root(con: sqlite3.Connection, title: str) -> tuple[int, str]:
    rows = con.execute(
        """
        select i.stable_id, s.cloud_id
        from items i
        join stable_ids s on i.stable_id = s.stable_id
        where i.local_title = ? and i.is_folder = 1 and i.trashed = 0
        order by i.stable_id desc
        """,
        (title,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"Google Drive folder not found in DriveFS metadata: {title}")
    return int(rows[0][0]), str(rows[0][1])


def descendants(con: sqlite3.Connection, root_stable_id: int) -> dict[str, dict[str, object]]:
    rows = con.execute(
        """
        with recursive tree(stable_id, path) as (
          select i.stable_id, i.local_title
          from items i
          where i.stable_id = ?
          union all
          select c.stable_id, tree.path || '/' || c.local_title
          from stable_parents sp
          join tree on sp.parent_stable_id = tree.stable_id
          join items c on c.stable_id = sp.item_stable_id
          where c.trashed = 0
        )
        select tree.path, i.stable_id, s.cloud_id, i.local_title, i.mime_type, i.file_size, i.is_folder
        from tree
        join items i on i.stable_id = tree.stable_id
        left join stable_ids s on s.stable_id = i.stable_id
        """,
        (root_stable_id,),
    ).fetchall()
    out: dict[str, dict[str, object]] = {}
    for path, stable_id, cloud_id, title, mime_type, file_size, is_folder in rows:
        rel = str(path).split("/", 1)[1] if "/" in str(path) else ""
        out[rel.replace("\\", "/")] = {
            "stable_id": stable_id,
            "id": cloud_id,
            "title": title,
            "mime_type": mime_type,
            "file_size": file_size,
            "is_folder": bool(is_folder),
        }
    return out


def drive_file_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def drive_preview_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/preview"


def drive_thumbnail_url(file_id: str) -> str:
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w640"


def main() -> None:
    args = parse_args()
    local_site = Path(args.local_site)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_items = json.loads((local_site / "videos.json").read_text(encoding="utf-8"))
    with connect_readonly(Path(args.drivefs_db)) as con:
        root_stable_id, root_cloud_id = find_root(con, args.drive_root_title)
        metadata = descendants(con, root_stable_id)

    external_items = []
    missing: list[str] = []
    for item in local_items:
        media_meta = metadata.get(str(item["src"]))
        poster_meta = metadata.get(str(item["poster"])) if item.get("poster") else None
        if not media_meta or not media_meta.get("id"):
            missing.append(str(item["src"]))
            continue
        file_id = str(media_meta["id"])
        poster_id = str(poster_meta["id"]) if poster_meta and poster_meta.get("id") else ""
        external_items.append(
            {
                **item,
                "drive_id": file_id,
                "drive_url": drive_file_url(file_id),
                "preview_url": drive_preview_url(file_id),
                "embed_url": drive_preview_url(file_id),
                "poster_drive_id": poster_id,
                "poster": drive_thumbnail_url(poster_id) if poster_id else "",
                "src": drive_preview_url(file_id),
                "ready": True,
            }
        )

    if missing:
        raise RuntimeError(f"Missing Drive IDs for {len(missing)} media files, first={missing[:5]}")

    folder_url = f"https://drive.google.com/drive/folders/{root_cloud_id}"
    (out_dir / "videos_external.json").write_text(json.dumps(external_items, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(HTML.replace("__FOLDER_URL__", folder_url), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "YOLOMG external-link gallery",
                f"Drive folder: {folder_url}",
                "Open index.html from any static host. Video playback uses Google Drive preview URLs.",
                "Access depends on the Google Drive sharing settings of the uploaded files/folder.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"folder_url={folder_url}")
    print(f"videos={len(external_items)}")
    print(f"site={out_dir / 'index.html'}")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLOMG Motion Process Gallery - Drive Links</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #191d20;
      --panel-2: #20262a;
      --text: #edf1f4;
      --muted: #9aa6ad;
      --line: #343c42;
      --accent: #7dd3fc;
      --good: #86efac;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
    }
    .app { display: grid; grid-template-columns: 360px 1fr; height: 100vh; min-height: 620px; }
    aside { border-right: 1px solid var(--line); background: var(--panel); display: flex; flex-direction: column; min-width: 0; }
    header { padding: 18px; border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 8px; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 12px; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
    .stat { background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .stat strong { display: block; font-size: 16px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .controls { display: grid; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--line); }
    input, select { width: 100%; background: #0f1214; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 10px 11px; outline: none; }
    .list { overflow: auto; padding: 10px; display: grid; gap: 8px; align-content: start; }
    .item { display: grid; grid-template-columns: 96px 1fr; gap: 10px; padding: 8px; background: transparent; border: 1px solid transparent; border-radius: 6px; color: inherit; text-align: left; cursor: pointer; }
    .item:hover, .item.active { background: var(--panel-2); border-color: var(--line); }
    .thumb { width: 96px; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 4px; background: #0b0d0e; }
    .meta { min-width: 0; }
    .name { font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .mini { display: block; color: var(--muted); font-size: 12px; margin-top: 3px; }
    main { display: grid; grid-template-rows: auto 1fr auto; min-width: 0; height: 100vh; }
    .topbar { padding: 16px 20px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    .title h2 { margin: 0; font-size: 19px; font-weight: 650; }
    .title p { margin: 4px 0 0; color: var(--muted); }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    button, a.button { background: var(--panel-2); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 9px 12px; text-decoration: none; cursor: pointer; }
    button:hover, a.button:hover { border-color: var(--accent); }
    .stage { min-height: 0; display: grid; place-items: center; padding: 16px 20px; }
    iframe { width: min(100%, 1480px); height: min(76vh, 840px); background: black; border: 1px solid var(--line); border-radius: 6px; }
    .footer { padding: 12px 20px; color: var(--muted); border-top: 1px solid var(--line); display: flex; justify-content: space-between; gap: 12px; font-size: 12px; }
    .ready { color: var(--good); }
    @media (max-width: 900px) {
      body { overflow: auto; }
      .app { grid-template-columns: 1fr; height: auto; }
      aside, main { height: auto; }
      .list { max-height: 420px; }
      .topbar, .footer { flex-direction: column; align-items: flex-start; }
      iframe { height: 56vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>YOLOMG Motion Process Gallery</h1>
        <div class="sub">Google Drive external preview links</div>
        <div class="stats">
          <div class="stat"><strong id="totalCount">0</strong><span>videos</span></div>
          <div class="stat"><strong id="trainCount">0</strong><span>train</span></div>
          <div class="stat"><strong id="testCount">0</strong><span>test</span></div>
        </div>
      </header>
      <div class="controls">
        <input id="search" placeholder="Search phantom ID">
        <select id="split">
          <option value="all">All splits</option>
          <option value="test">Test</option>
          <option value="train">Train</option>
        </select>
      </div>
      <div id="list" class="list"></div>
    </aside>
    <main>
      <div class="topbar">
        <div class="title">
          <h2 id="videoTitle">Loading...</h2>
          <p id="videoMeta"></p>
        </div>
        <div class="actions">
          <button id="prev">Previous</button>
          <button id="next">Next</button>
          <a id="openFile" class="button" href="#" target="_blank" rel="noreferrer">Open Drive Video</a>
          <a class="button" href="__FOLDER_URL__" target="_blank" rel="noreferrer">Drive Folder</a>
        </div>
      </div>
      <div class="stage">
        <iframe id="player" allow="autoplay; fullscreen" allowfullscreen></iframe>
      </div>
      <div class="footer">
        <span id="path"></span>
        <span>Playback requires access to the linked Google Drive files.</span>
      </div>
    </main>
  </div>
  <script>
    const state = { videos: [], filtered: [], current: 0 };
    const els = {
      list: document.getElementById('list'),
      search: document.getElementById('search'),
      split: document.getElementById('split'),
      player: document.getElementById('player'),
      title: document.getElementById('videoTitle'),
      meta: document.getElementById('videoMeta'),
      path: document.getElementById('path'),
      openFile: document.getElementById('openFile'),
      prev: document.getElementById('prev'),
      next: document.getElementById('next'),
      total: document.getElementById('totalCount'),
      train: document.getElementById('trainCount'),
      test: document.getElementById('testCount')
    };
    function applyFilters() {
      const q = els.search.value.trim().toLowerCase();
      const split = els.split.value;
      state.filtered = state.videos.filter(v => (split === 'all' || v.split === split) && (!q || v.video.toLowerCase().includes(q)));
      state.current = Math.min(state.current, Math.max(0, state.filtered.length - 1));
      renderList();
      selectVideo(state.current);
    }
    function renderList() {
      els.list.innerHTML = '';
      state.filtered.forEach((v, idx) => {
        const btn = document.createElement('button');
        btn.className = `item ${idx === state.current ? 'active' : ''}`;
        btn.innerHTML = `
          <img class="thumb" src="${v.poster || ''}" alt="">
          <span class="meta">
            <span class="name">${v.video}</span>
            <span class="mini">${v.split} | ${v.frames || '?'} frames</span>
            <span class="mini ready">Drive linked</span>
          </span>`;
        btn.addEventListener('click', () => selectVideo(idx));
        els.list.appendChild(btn);
      });
    }
    function selectVideo(idx) {
      if (!state.filtered.length) {
        els.title.textContent = 'No videos';
        els.meta.textContent = '';
        els.player.removeAttribute('src');
        return;
      }
      state.current = Math.max(0, Math.min(idx, state.filtered.length - 1));
      const v = state.filtered[state.current];
      els.title.textContent = `${v.split.toUpperCase()} / ${v.video}`;
      els.meta.textContent = `${v.frames || '?'} frames | ${state.current + 1}/${state.filtered.length}`;
      els.path.textContent = v.drive_url;
      els.player.src = v.embed_url;
      els.openFile.href = v.drive_url;
      [...els.list.children].forEach((el, i) => el.classList.toggle('active', i === state.current));
      const active = els.list.children[state.current];
      if (active) active.scrollIntoView({ block: 'nearest' });
    }
    els.search.addEventListener('input', applyFilters);
    els.split.addEventListener('change', applyFilters);
    els.prev.addEventListener('click', () => selectVideo(state.current - 1));
    els.next.addEventListener('click', () => selectVideo(state.current + 1));
    window.addEventListener('keydown', (e) => {
      if (e.target.matches('input, select')) return;
      if (e.key === 'ArrowUp') selectVideo(state.current - 1);
      if (e.key === 'ArrowDown') selectVideo(state.current + 1);
    });
    fetch('videos_external.json')
      .then(r => r.json())
      .then(videos => {
        state.videos = videos;
        els.total.textContent = videos.length;
        els.train.textContent = videos.filter(v => v.split === 'train').length;
        els.test.textContent = videos.filter(v => v.split === 'test').length;
        state.filtered = videos;
        renderList();
        selectVideo(0);
      })
      .catch(err => {
        els.title.textContent = 'Failed to load videos_external.json';
        els.meta.textContent = String(err);
      });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
