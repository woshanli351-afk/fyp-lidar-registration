#!/usr/bin/env python3
"""Validate and serve the progressive replay from a fixed local origin."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading
import webbrowser


REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo" / "progressive_replay_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_demo() -> dict:
    required = [
        DEMO / name
        for name in (
            "index.html",
            "styles.css",
            "app.js",
            "data/demo.json",
            "data/gravity_map.json",
        )
    ]
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
    gravity = bundle.get("gravity_diagnostic", {})
    if gravity.get("status") != "SOFT_GRAVITY_REMAINS_OFFLINE_ONLY":
        raise SystemExit("Demo data failed validation: Gravity must remain offline-only")
    if gravity.get("graph", {}).get("nodes") != 5796 or gravity.get("graph", {}).get("loop_factors") != 75:
        raise SystemExit("Demo data failed validation: Gravity graph identity changed")
    if gravity.get("selection", {}).get("used_gt") is not False:
        raise SystemExit("Demo data failed validation: Gravity selection must remain GT-free")
    decision = gravity.get("decision", {})
    if decision.get("map_improved") is not False or decision.get("final_backend_changed") is not False:
        raise SystemExit("Demo data failed validation: Gravity decision boundary changed")
    gravity_map_path = DEMO / gravity.get("map_export", {}).get("file", "")
    if not gravity_map_path.is_file() or sha256(gravity_map_path) != gravity["map_export"].get("sha256"):
        raise SystemExit("Demo data failed validation: Gravity map hash mismatch")
    gravity_map = json.loads(gravity_map_path.read_text(encoding="utf-8"))
    if gravity_map.get("schema") != "FYP_GRAVITY_DEMO_MAP_V2" or gravity_map.get("mode") != "GT_FREE_PRESENTATION_EXPORT":
        raise SystemExit("Demo data failed validation: Gravity map schema changed")
    if set(gravity_map.get("variants", {})) != {"free", "soft_x8"}:
        raise SystemExit("Demo data failed validation: incomplete Gravity map variants")
    for variant in ("free", "soft_x8"):
        robots = gravity_map["variants"][variant]
        if set(robots) != {"robot1", "robot3"} or any(not robots[robot] for robot in robots):
            raise SystemExit(f"Demo data failed validation: incomplete Gravity map {variant}")
        if any(len(point) != 4 for robot in robots for point in robots[robot]):
            raise SystemExit(f"Demo data failed validation: Gravity map {variant} lacks keyframe IDs")
    if set(gravity_map.get("keyframe_ranges", {})) != {"robot1", "robot3"}:
        raise SystemExit("Demo data failed validation: Gravity keyframe ranges missing")
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
        "Gravity remains offline-only; press Ctrl-C to stop."
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
