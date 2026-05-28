from __future__ import annotations

import argparse
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class PlayerHandler(BaseHTTPRequestHandler):
    player_html: Path
    video_root: Path

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), fmt % args), flush=True)

    def do_GET(self) -> None:
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self._handle_request(send_body=False)

    def _handle_request(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html", "/player.html"}:
            self._send_file(self.player_html, inline_type="text/html; charset=utf-8", send_body=send_body)
            return
        if parsed.path.startswith("/video/"):
            name = unquote(parsed.path.removeprefix("/video/")).strip("/")
            if "." not in name:
                candidates = [
                    self.video_root / f"{name}.mp4",
                    self.video_root / f"{name}.avi",
                    self.video_root / f"{name}.mov",
                ]
                path = next((p for p in candidates if p.exists()), None)
            else:
                path = self.video_root / name
            if not path or not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "video not found")
                return
            self._send_file(path, send_body=send_body)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def _send_file(self, path: Path, inline_type: str | None = None, send_body: bool = True) -> None:
        size = path.stat().st_size
        mime = inline_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start_s, _, end_s = range_header.removeprefix("bytes=").partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start >= size or end < start:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if not send_body:
                return
            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                        break
                    remaining -= len(chunk)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if not send_body:
            return
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--player-html", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8777)
    args = p.parse_args()

    PlayerHandler.player_html = args.player_html.resolve()
    PlayerHandler.video_root = args.video_root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), PlayerHandler)
    print(f"serving http://{args.host}:{args.port}/", flush=True)
    print(f"player={PlayerHandler.player_html}", flush=True)
    print(f"video_root={PlayerHandler.video_root}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
