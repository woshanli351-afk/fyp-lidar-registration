#!/usr/bin/env python3
"""Export presentation-safe Free/Soft Gravity map samples.

The exporter deliberately reads only pose trajectories plus the keyframe ID to
LiDAR-file mapping.  Ground-truth pose columns in the keyframe tables are never
loaded.  The result is a compact XYZ sample for the progressive replay's
offline Gravity diagnostic view; it is not an evaluation artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_pgo_map_tail import (
    ROBOTS,
    build_labeled_map,
    load_trajectory,
    voxel_with_labels,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_sample(points: np.ndarray, limit: int) -> np.ndarray:
    """Return a stable, coverage-preserving sample including both endpoints."""
    if limit <= 0:
        raise ValueError("point limit must be positive")
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def compact_points(points: np.ndarray) -> list[list[float]]:
    return np.round(points.astype(float), 3).tolist()


def build_variant(
    trajectory_path: Path,
    keyframe_tables: dict[str, Path],
    lidar_roots: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[dict[str, list[list[float]]], dict[str, int]]:
    trajectory = load_trajectory(trajectory_path)
    maps = build_labeled_map(
        trajectory,
        keyframe_tables,
        lidar_roots,
        args.frame_stride,
        args.point_stride,
        args.range_min_m,
        args.range_max_m,
    )
    sampled: dict[str, list[list[float]]] = {}
    counts: dict[str, int] = {}
    for robot in ROBOTS:
        points, labels = maps[robot]
        voxelized, _ = voxel_with_labels(points, labels, args.voxel_m)
        counts[robot] = int(len(voxelized))
        sampled[robot] = compact_points(
            deterministic_sample(voxelized, args.max_points_per_robot)
        )
    return sampled, counts


def run(args: argparse.Namespace) -> None:
    keyframe_tables = {
        "robot1": args.robot1_keyframes.resolve(),
        "robot3": args.robot3_keyframes.resolve(),
    }
    lidar_roots = {
        "robot1": args.robot1_lidar_root.resolve(),
        "robot3": args.robot3_lidar_root.resolve(),
    }
    free_points, free_counts = build_variant(
        args.free_trajectory.resolve(), keyframe_tables, lidar_roots, args
    )
    soft_points, soft_counts = build_variant(
        args.soft_trajectory.resolve(), keyframe_tables, lidar_roots, args
    )
    bundle = {
        "schema": "FYP_GRAVITY_DEMO_MAP_V1",
        "mode": "GT_FREE_PRESENTATION_EXPORT",
        "candidate": "ekf_gravity_soft_cov_x8",
        "configuration": {
            "frame_stride": args.frame_stride,
            "point_stride": args.point_stride,
            "range_m": [args.range_min_m, args.range_max_m],
            "voxel_m": args.voxel_m,
            "max_points_per_robot": args.max_points_per_robot,
        },
        "source_sha256": {
            "free_trajectory": sha256(args.free_trajectory),
            "soft_trajectory": sha256(args.soft_trajectory),
            "robot1_keyframes": sha256(args.robot1_keyframes),
            "robot3_keyframes": sha256(args.robot3_keyframes),
        },
        "voxelized_point_counts": {
            "free": free_counts,
            "soft_x8": soft_counts,
        },
        "variants": {
            "free": free_points,
            "soft_x8": soft_points,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free-trajectory", type=Path, required=True)
    parser.add_argument("--soft-trajectory", type=Path, required=True)
    parser.add_argument("--robot1-keyframes", type=Path, required=True)
    parser.add_argument("--robot3-keyframes", type=Path, required=True)
    parser.add_argument("--robot1-lidar-root", type=Path, required=True)
    parser.add_argument("--robot3-lidar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--point-stride", type=int, default=16)
    parser.add_argument("--range-min-m", type=float, default=1.0)
    parser.add_argument("--range-max-m", type=float, default=60.0)
    parser.add_argument("--voxel-m", type=float, default=0.5)
    parser.add_argument("--max-points-per-robot", type=int, default=6000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
