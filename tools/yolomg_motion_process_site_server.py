from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the YOLOMG motion-process website.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8782)
    p.add_argument("--directory", default=r"C:\Users\aaron\Desktop\URAP\artifacts\yolomg_motion_process_site")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    directory = Path(args.directory)
    handler = lambda *h_args, **h_kwargs: SimpleHTTPRequestHandler(*h_args, directory=str(directory), **h_kwargs)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving=http://{args.host}:{args.port}/", flush=True)
    print(f"directory={directory}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
