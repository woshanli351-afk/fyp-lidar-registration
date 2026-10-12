#!/usr/bin/env python3
"""Add a presentation-only GNSS overlay to an existing replay bundle.

The GNSS samples are joined by robot/keyframe ID and are never used to select
Scan Context candidates, accept GICP registrations, or optimise the pose graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROBOTS = ("robot1", "robot3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_robot_track(path: Path, keyframes: list[int]) -> list[list[float]]:
    frame = pd.read_csv(path).set_index("keyframe_id", verify_integrity=True)
    required = {
        "gnss1_latitude",
        "gnss1_longitude",
        "gnss2_latitude",
        "gnss2_longitude",
        "gnss1_match_status",
        "gnss2_match_status",
        "gnss_both_valid",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing GNSS columns: {sorted(missing)}")
    absent = sorted(set(keyframes).difference(frame.index))
    if absent:
        raise ValueError(f"{path} missing replay keyframes: {absent[:8]}")

    selected = frame.loc[keyframes]
    bad = selected[
        (selected.gnss1_match_status != "MATCHED")
        | (selected.gnss2_match_status != "MATCHED")
        | (~selected.gnss_both_valid.astype(bool))
    ]
    if not bad.empty:
        raise ValueError(f"{path} has {len(bad)} invalid replay GNSS rows")

    latitude = (selected.gnss1_latitude + selected.gnss2_latitude) / 2.0
    longitude = (selected.gnss1_longitude + selected.gnss2_longitude) / 2.0
    return [
        [int(keyframe), round(float(lat), 7), round(float(lon), 7)]
        for keyframe, lat, lon in zip(keyframes, latitude, longitude, strict=True)
    ]


def enrich(bundle_path: Path, gnss_root: Path) -> None:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    local = bundle["trajectories"]["local"]
    tracks: dict[str, list[list[float]]] = {}
    sources = []
    for robot in ROBOTS:
        source = gnss_root / f"{robot}_full_ekf_stability.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        keyframes = [int(point[0]) for point in local[robot]]
        tracks[robot] = load_robot_track(source, keyframes)
        sources.append(source)

    all_points = [point for robot in ROBOTS for point in tracks[robot]]
    latitudes = [point[1] for point in all_points]
    longitudes = [point[2] for point in all_points]
    bundle["geospatial"] = {
        "source": "dual_gnss_midpoint_joined_by_keyframe_id",
        "usage": "presentation_only_not_used_by_scan_context_gicp_or_pose_graph",
        "attribution": "Esri World Imagery",
        "bounds": {
            "south": min(latitudes),
            "west": min(longitudes),
            "north": max(latitudes),
            "east": max(longitudes),
        },
        "trajectories": tracks,
    }
    bundle["metadata"]["geospatial_overlay"] = (
        "Dual-GNSS midpoint reference overlay; visualization only and excluded "
        "from Scan Context, GICP, and pose-graph decisions."
    )
    bundle["provenance"] = [
        row for row in bundle["provenance"] if not row["file"].startswith("external_gnss_audit/")
    ]
    bundle["provenance"].extend(
        {"file": f"external_gnss_audit/{path.name}", "sha256": sha256(path)}
        for path in sources
    )
    bundle_path.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    print(
        f"Updated {bundle_path} with "
        f"{sum(len(points) for points in tracks.values())} validated GNSS samples"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gnss-root",
        type=Path,
        required=True,
        help="Directory containing robot1/robot3 full EKF stability CSV files",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "demo"
        / "progressive_replay_v2"
        / "data"
        / "demo.json",
    )
    args = parser.parse_args()
    enrich(args.bundle.resolve(), args.gnss_root.resolve())


if __name__ == "__main__":
    main()
