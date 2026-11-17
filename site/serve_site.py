"""
Serve the built student site from this computer (for testing, or for a classroom without internet).

    python serve_site.py                 # http://localhost:8000  (this computer only)
    python serve_site.py --lan           # also reachable by the laptops on the same network or hotspot
    python serve_site.py --port 8080

Why not plain ``python -m http.server``: on Windows it labels ``.mjs`` files with the wrong type and
the browser then refuses to start Python. This script sends the types a web host sends.
Only the standard library is needed. Stop it with Ctrl+C (and stop it before rebuilding the site).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import socket

SITE = pathlib.Path(__file__).resolve().parent / "_output"
DEFAULT_PORT = 8000
TYPES = {                         # what browsers need for the Python runtime
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".whl": "application/octet-stream",
    ".ipynb": "application/json",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **TYPES}

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")      # a rebuilt site is seen at once
        super().end_headers()


def lan_address() -> str:
    """The address of this computer on the local network (no data is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the MATH 331 student site.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--lan", action="store_true", help="let other computers on this network connect")
    parser.add_argument("--directory", type=pathlib.Path, default=SITE)
    args = parser.parse_args()
    if not (args.directory / "index.html").exists():
        raise SystemExit(f"no site in {args.directory}: run build_site.py first")
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    handler = functools.partial(Handler, directory=str(args.directory))
    with http.server.ThreadingHTTPServer((host, args.port), handler) as server:
        print(f"serving {args.directory}")
        print(f"  on this computer:  http://localhost:{args.port}")
        if args.lan:
            print(f"  for the class:     http://{lan_address()}:{args.port}")
        print("Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    main()
