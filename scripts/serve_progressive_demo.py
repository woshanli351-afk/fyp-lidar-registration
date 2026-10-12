#!/usr/bin/env python3
"""Validate and serve the progressive replay from a fixed local origin."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser


REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo" / "progressive_replay_v2"


def validate_demo() -> dict:
    required = [DEMO / name for name in ("index.html", "styles.css", "app.js", "data/demo.json")]
    missing = [str(path.relative_to(REPO)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing demo files: " + ", ".join(missing))
    bundle = json.loads((DEMO / "data/demo.json").read_text(encoding="utf-8"))
    if len(bundle.get("metrics", [])) != 10 or len(bundle.get("events", [])) != 75:
        raise SystemExit("Demo data failed validation: expected 10 metric rows and 75 events")
    if bundle.get("metadata", {}).get("transform_direction") != "T_query_from_candidate":
        raise SystemExit("Demo data failed validation: transform direction changed")
    geospatial = bundle.get("geospatial", {})
    tracks = geospatial.get("trajectories", {})
    if set(tracks) != {"robot1", "robot3"} or any(not tracks[robot] for robot in tracks):
        raise SystemExit("Demo data failed validation: incomplete geospatial tracks")
    if "not_used_by_scan_context_gicp_or_pose_graph" not in geospatial.get("usage", ""):
        raise SystemExit("Demo data failed validation: geospatial claim boundary missing")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    bundle = validate_demo()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        partial(SimpleHTTPRequestHandler, directory=str(DEMO)),
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Progressive replay ready: {url}")
    print(
        f"Validated {len(bundle['events'])} events and {len(bundle['metrics'])} policy rows; "
        "press Ctrl-C to stop."
    )
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
