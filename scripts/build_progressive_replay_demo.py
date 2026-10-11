#!/usr/bin/env python3
"""Build the source-backed data bundle for the progressive offline replay demo.

The builder deliberately exports only presentation-safe fields. Stage 11 ground
truth labels remain aggregated metrics; the event stream contains no GT fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "demo" / "progressive_replay_v2"
POLICY_LABELS = {
    "sc_rank1": "SC Rank-1",
    "sc_score_gate": "SC score gate",
    "gicp_rank1": "Rank-1 + GICP",
    "gicp_top3": "Top-3 + GICP",
    "selective_gicp": "Selective GICP",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_name(path: Path, team_root: Path) -> str:
    for root, prefix in ((team_root, "team/"), (REPO, "companion/")):
        try:
            return prefix + path.relative_to(root).as_posix()
        except ValueError:
            pass
    return "external_map_export/" + path.name


def compact_trajectory(path: Path, stride: int = 8) -> dict[str, list[list[float]]]:
    frame = pd.read_csv(path)
    required = {"robot_id", "keyframe_id", "tx", "ty"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing trajectory columns: {sorted(missing)}")
    result: dict[str, list[list[float]]] = {}
    for robot, group in frame.groupby("robot_id", sort=True):
        sampled = group.iloc[::stride].copy()
        if sampled.index[-1] != group.index[-1]:
            sampled = pd.concat([sampled, group.iloc[[-1]]])
        result[str(robot)] = [
            [int(row.keyframe_id), round(float(row.tx), 5), round(float(row.ty), 5)]
            for row in sampled.itertuples()
        ]
    return result


def compact_cloud(points: list[list[float]], limit: int = 110) -> list[list[float]]:
    if len(points) <= limit:
        chosen = points
    else:
        step = max(1, len(points) // limit)
        chosen = points[::step][:limit]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in chosen]


def metric_rows(stage11_root: Path) -> list[dict]:
    source = stage11_root / "combined_policy_tradeoff.csv"
    frame = pd.read_csv(source)
    fields = [
        "direction",
        "policy",
        "queries",
        "valid_overlap_queries",
        "no_overlap_queries",
        "precision",
        "correct_pair_recall",
        "no_overlap_rejection_rate",
        "mapping_hazard_event_count",
        "mean_gicp_calls",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "queries_per_second",
        "median_translation_error_m",
        "p95_translation_error_m",
        "median_rotation_error_deg",
        "p95_rotation_error_deg",
    ]
    rows = []
    for record in frame[fields].to_dict("records"):
        clean = {}
        for key, value in record.items():
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, float):
                clean[key] = round(value, 6)
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        clean["label"] = POLICY_LABELS[clean["policy"]]
        rows.append(clean)
    return rows


def event_rows(team_root: Path) -> list[dict]:
    clean_root = team_root / "outputs" / "cumulti_v1" / "12d_clean_backend"
    loops = pd.read_csv(clean_root / "ordered_rank1_clean_loops.csv")
    historical = json.loads(
        (team_root / "demo" / "final_replay" / "data" / "replay.json").read_text()
    )
    lookup = {
        (
            event["source_robot"],
            int(event["source_keyframe"]),
            event["target_robot"],
            int(event["target_keyframe"]),
        ): event
        for event in historical["events"]
    }
    events = []
    for row in loops.itertuples():
        key = (
            row.query_robot,
            int(row.query_keyframe_id),
            row.candidate_robot,
            int(row.candidate_keyframe_id),
        )
        if key not in lookup:
            raise KeyError(f"Missing frozen visualization sample for loop {key}")
        visual = lookup[key]
        events.append(
            {
                "index": int(row.arrival_index),
                "loop_id": int(row.frozen_loop_id),
                "query_robot": row.query_robot,
                "query_keyframe": int(row.query_keyframe_id),
                "candidate_robot": row.candidate_robot,
                "candidate_keyframe": int(row.candidate_keyframe_id),
                "sc_score": round(float(row.SC_score), 5),
                "best_shift": int(row.best_shift),
                "sc_yaw_deg": round(float(row.SC_yaw_initialization_deg), 2),
                "gicp_quality": round(float(row.GICP_quality), 5),
                "transform_direction": "T_query_from_candidate",
                "query_cloud_xy": compact_cloud(visual["query_cloud_xy"]),
                "candidate_raw_xy": compact_cloud(visual["candidate_raw_xy"]),
                "candidate_gicp_aligned_xy": compact_cloud(
                    visual["candidate_gicp_aligned_xy"]
                ),
            }
        )
    return events


def read_binary_xyz_ply(path: Path, limit: int = 12_000) -> list[list[float]]:
    with path.open("rb") as handle:
        vertex_count = None
        properties = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property "):
                properties.append(text)
            elif text == "end_header":
                break
        expected = ["property float x", "property float y", "property float z"]
        if vertex_count is None or properties != expected:
            raise ValueError(f"Expected binary XYZ-only PLY: {path}")
        points = np.fromfile(handle, dtype="<f4", count=vertex_count * 3).reshape(-1, 3)
    points = points[np.isfinite(points[:, :2]).all(axis=1)]
    step = max(1, math.ceil(len(points) / limit))
    sampled = points[::step, :2][:limit]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in sampled]


def map_points(map_root: Path) -> dict[str, dict[str, list[list[float]]]]:
    return {
        name: {
            robot: read_binary_xyz_ply(map_root / f"{name}_{robot}.ply")
            for robot in ("robot1", "robot3")
        }
        for name in ("K1", "K10", "K75")
    }


def build(team_root: Path, map_root: Path, output: Path) -> None:
    stage11_root = REPO / "results" / "cumulti_stage11"
    clean_root = team_root / "outputs" / "cumulti_v1" / "12d_clean_backend"
    sparse_root = team_root / "outputs" / "cumulti_v1" / "13a_sparse_loops"
    required = [
        stage11_root / "combined_policy_tradeoff.csv",
        clean_root / "clean_local_trajectory.csv",
        clean_root / "clean_first_loop_trajectory.csv",
        clean_root / "clean_all75_gtsam_trajectory.csv",
        clean_root / "ordered_rank1_clean_loops.csv",
        sparse_root / "trajectories" / "K10_optimized.csv",
        team_root / "demo" / "final_replay" / "data" / "replay.json",
    ]
    required.extend(
        map_root / f"{name}_{robot}.ply"
        for name in ("K1", "K10", "K75")
        for robot in ("robot1", "robot3")
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing demo inputs:\n" + "\n".join(missing))

    (output / "data").mkdir(parents=True, exist_ok=True)
    bundle = {
        "metadata": {
            "mode": "FROZEN_OFFLINE_PROGRESSIVE_REPLAY",
            "backend": "CENTRALIZED_BATCH_GTSAM_SNAPSHOTS",
            "frontend_policy": "M1000_SCAN_CONTEXT_RANK1_PLUS_GICP",
            "gicp_threshold": 0.6091,
            "transform_direction": "T_query_from_candidate",
            "graph_nodes": 5796,
            "odometry_factors": 5794,
            "clean_loop_factors": 75,
            "source_stage11_commit": "2ff22b8e5aa661d36f25414cdcac5de8e4ed87b1",
            "source_clean_backend_commit": "8946e29eb74658bfb0a5bf85635ff1aeac05a417",
            "claim_boundary": (
                "Recorded offline replay with frozen candidates and batch snapshots; "
                "not live sensing, causal online SC/GICP, or validated incremental iSAM2."
            ),
        },
        "metrics": metric_rows(stage11_root),
        "events": event_rows(team_root),
        "trajectories": {
            "local": compact_trajectory(clean_root / "clean_local_trajectory.csv"),
            "K1": compact_trajectory(clean_root / "clean_first_loop_trajectory.csv"),
            "K10": compact_trajectory(sparse_root / "trajectories" / "K10_optimized.csv"),
            "K75": compact_trajectory(clean_root / "clean_all75_gtsam_trajectory.csv"),
        },
        "map_points": map_points(map_root),
        "provenance": [
            {
                "file": provenance_name(path, team_root),
                "sha256": sha256(path),
            }
            for path in required
        ],
    }
    target = output / "data" / "demo.json"
    target.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--team-root",
        type=Path,
        required=True,
        help="Checkout of James-5757/fyp-place-recognition containing Stage 12D/13A assets",
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
        help="Directory containing K1/K10/K75 Robot1/Robot3 binary XYZ PLY files",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.team_root.resolve(), args.map_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
