#!/usr/bin/env python3
"""Run coarse-to-fine ICP and GICP on reranked Top-K candidate pairs.

Registration never reads ground-truth fields.  They are copied to the output
only so a separate evaluator can measure verification quality afterwards.
The estimated transform maps the candidate (source) cloud into the query
(target) cloud.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np


def yaw_initialization(best_shift: int, num_sectors: int = 60) -> np.ndarray:
    angle = 2.0 * math.pi * best_shift / num_sectors
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = [[cosine, -sine], [sine, cosine]]
    return result


def select_rows(path: Path, top_k: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_frame", "candidate_frame", "rank", "best_shift"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    selected = [row for row in rows if int(row["rank"]) <= top_k]
    selected.sort(key=lambda row: (int(row["query_frame"]), int(row["rank"])))
    return selected


def transform_summary(transform: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(transform[:3, 3]))
    yaw_degrees = float(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))
    return translation, yaw_degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--velodyne-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-sectors", type=int, default=60)
    parser.add_argument("--voxel-size", type=float, default=0.75)
    parser.add_argument("--min-range", type=float, default=2.0)
    parser.add_argument("--max-range", type=float, default=50.0)
    parser.add_argument("--min-height", type=float, default=-3.0)
    parser.add_argument("--max-height", type=float, default=5.0)
    parser.add_argument("--coarse-distance", type=float, default=3.0)
    parser.add_argument("--fine-distance", type=float, default=1.0)
    parser.add_argument("--coarse-iterations", type=int, default=30)
    parser.add_argument("--fine-iterations", type=int, default=30)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Run only the first N pairs for a smoke test")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = [
        args.top_k, args.num_sectors, args.voxel_size, args.max_range,
        args.coarse_distance, args.fine_distance, args.coarse_iterations,
        args.fine_iterations, args.cache_size,
    ]
    if any(value <= 0 for value in positive_values):
        raise ValueError("Top-K, geometry scales, iteration counts, and cache-size must be positive")
    if args.min_range >= args.max_range or args.min_height >= args.max_height:
        raise ValueError("Invalid point-cloud crop bounds")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required: pip install open3d==0.19.0") from exc

    rows = select_rows(args.candidate_csv, args.top_k)
    if args.limit is not None:
        rows = rows[:args.limit]

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

    criteria_coarse = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=args.coarse_iterations,
    )
    criteria_fine = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=args.fine_iterations,
    )

    output_fields = list(rows[0]) + [
        "source_is_candidate", "init_yaw_degrees",
        "icp_fitness", "icp_inlier_rmse", "icp_quality", "icp_translation_norm", "icp_yaw_degrees", "icp_transform",
        "gicp_fitness", "gicp_inlier_rmse", "gicp_quality", "gicp_translation_norm", "gicp_yaw_degrees", "gicp_transform",
        "registration_seconds",
    ]
    completed: set[tuple[int, int]] = set()
    mode = "w"
    if args.resume and args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as handle:
            completed = {
                (int(row["query_frame"]), int(row["candidate_frame"]))
                for row in csv.DictReader(handle)
            }
        mode = "a"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(rows, start=1):
            query = int(row["query_frame"])
            candidate = int(row["candidate_frame"])
            if (query, candidate) in completed:
                continue
            started = time.perf_counter()
            source, target = load_cloud(candidate), load_cloud(query)
            initialization = yaw_initialization(int(row["best_shift"]), args.num_sectors)

            icp_coarse = o3d.pipelines.registration.registration_icp(
                source, target, args.coarse_distance, initialization,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), criteria_coarse,
            )
            icp = o3d.pipelines.registration.registration_icp(
                source, target, args.fine_distance, icp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), criteria_fine,
            )
            gicp_coarse = o3d.pipelines.registration.registration_generalized_icp(
                source, target, args.coarse_distance, initialization,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(), criteria_coarse,
            )
            gicp = o3d.pipelines.registration.registration_generalized_icp(
                source, target, args.fine_distance, gicp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(), criteria_fine,
            )
            icp_translation, icp_yaw = transform_summary(np.asarray(icp.transformation))
            gicp_translation, gicp_yaw = transform_summary(np.asarray(gicp.transformation))
            updated = dict(row)
            updated.update({
                "source_is_candidate": 1,
                "init_yaw_degrees": 360.0 * int(row["best_shift"]) / args.num_sectors,
                "icp_fitness": float(icp.fitness),
                "icp_inlier_rmse": float(icp.inlier_rmse),
                "icp_quality": float(icp.fitness) / (1.0 + float(icp.inlier_rmse)),
                "icp_translation_norm": icp_translation,
                "icp_yaw_degrees": icp_yaw,
                "icp_transform": json.dumps(np.asarray(icp.transformation).reshape(-1).tolist()),
                "gicp_fitness": float(gicp.fitness),
                "gicp_inlier_rmse": float(gicp.inlier_rmse),
                "gicp_quality": float(gicp.fitness) / (1.0 + float(gicp.inlier_rmse)),
                "gicp_translation_norm": gicp_translation,
                "gicp_yaw_degrees": gicp_yaw,
                "gicp_transform": json.dumps(np.asarray(gicp.transformation).reshape(-1).tolist()),
                "registration_seconds": time.perf_counter() - started,
            })
            writer.writerow(updated)
            handle.flush()
            if index % 10 == 0 or index == len(rows):
                print(f"Verified {index}/{len(rows)} candidate pairs", flush=True)
    print(f"Geometric verification: {args.output}; requested_pairs={len(rows)}")


if __name__ == "__main__":
    main()
