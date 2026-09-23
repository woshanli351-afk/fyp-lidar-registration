#!/usr/bin/env python3
"""Estimate sequential keyframe odometry edges with coarse-to-fine GICP."""

from __future__ import annotations

import argparse
import csv
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np


def agent_frames() -> dict[str, list[int]]:
    return {
        "robot_a": list(range(0, 2401, 5)),
        "robot_b": list(range(2600, 4541, 5)),
    }


def sequential_edges(frames_by_agent: dict[str, list[int]]) -> list[tuple[str, int, int]]:
    return [
        (agent, previous, current)
        for agent, frames in frames_by_agent.items()
        for previous, current in zip(frames, frames[1:])
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velodyne-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.75)
    parser.add_argument("--min-range", type=float, default=2.0)
    parser.add_argument("--max-range", type=float, default=50.0)
    parser.add_argument("--min-height", type=float, default=-3.0)
    parser.add_argument("--max-height", type=float, default=5.0)
    parser.add_argument("--coarse-distance", type=float, default=5.0)
    parser.add_argument("--fine-distance", type=float, default=1.0)
    parser.add_argument("--coarse-iterations", type=int, default=40)
    parser.add_argument("--fine-iterations", type=int, default=30)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required: pip install open3d-cpu==0.19.0") from exc

    edges = sequential_edges(agent_frames())
    if args.limit is not None:
        edges = edges[: args.limit]

    @lru_cache(maxsize=args.cache_size)
    def load_cloud(frame: int):
        path = args.velodyne_dir / f"{frame:06d}.bin"
        values = np.fromfile(path, dtype=np.float32)
        if values.size % 4:
            raise ValueError(f"Invalid Velodyne scan: {path}")
        points = values.reshape(-1, 4)[:, :3]
        radius = np.linalg.norm(points[:, :2], axis=1)
        keep = (
            np.isfinite(points).all(axis=1)
            & (radius >= args.min_range) & (radius <= args.max_range)
            & (points[:, 2] >= args.min_height) & (points[:, 2] <= args.max_height)
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[keep].astype(np.float64, copy=False))
        return cloud.voxel_down_sample(args.voxel_size)

    coarse_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=args.coarse_iterations
    )
    fine_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=args.fine_iterations
    )
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    fields = [
        "agent", "target_frame", "source_frame", "measurement_convention",
        "gicp_fitness", "gicp_inlier_rmse", "gicp_quality",
        "gicp_transform", "registration_seconds",
    ]
    completed: set[tuple[str, int, int]] = set()
    previous_by_agent: dict[str, np.ndarray] = {}
    mode = "w"
    if args.resume and args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
        for row in old_rows:
            key = (row["agent"], int(row["target_frame"]), int(row["source_frame"]))
            completed.add(key)
            previous_by_agent[row["agent"]] = np.asarray(
                json.loads(row["gicp_transform"]), dtype=float
            ).reshape(4, 4)
        mode = "a"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for index, (agent, target_frame, source_frame) in enumerate(edges, start=1):
            if (agent, target_frame, source_frame) in completed:
                continue
            started = time.perf_counter()
            initialization = previous_by_agent.get(agent, np.eye(4))
            coarse = o3d.pipelines.registration.registration_generalized_icp(
                load_cloud(source_frame), load_cloud(target_frame),
                args.coarse_distance, initialization, estimation, coarse_criteria,
            )
            fine = o3d.pipelines.registration.registration_generalized_icp(
                load_cloud(source_frame), load_cloud(target_frame),
                args.fine_distance, coarse.transformation, estimation, fine_criteria,
            )
            transform = np.asarray(fine.transformation)
            previous_by_agent[agent] = transform
            writer.writerow({
                "agent": agent,
                "target_frame": target_frame,
                "source_frame": source_frame,
                "measurement_convention": "source_current_to_target_previous_velodyne",
                "gicp_fitness": float(fine.fitness),
                "gicp_inlier_rmse": float(fine.inlier_rmse),
                "gicp_quality": float(fine.fitness) / (1.0 + float(fine.inlier_rmse)),
                "gicp_transform": json.dumps(transform.reshape(-1).tolist()),
                "registration_seconds": time.perf_counter() - started,
            })
            handle.flush()
            if index % 10 == 0 or index == len(edges):
                print(f"Odometry {index}/{len(edges)} edges", flush=True)
    print(f"Keyframe odometry: {args.output}; requested_edges={len(edges)}")


if __name__ == "__main__":
    main()
