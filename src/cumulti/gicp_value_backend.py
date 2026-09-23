#!/usr/bin/env python3
"""GT-free reusable GICP backend for the CU-Multi Stage-11 value study.

The database candidate is always the Open3D source and the query is always the
target.  Every returned transform is therefore ``T_query_from_candidate``.
"""
from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import argparse
import csv
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


ROBOTS = {"robot1", "robot2", "robot3"}
RESULT_FIELDS = [
    "transform_direction", "init_yaw_deg", "gicp_fitness",
    "gicp_inlier_rmse", "gicp_quality", "gicp_transform",
    "source_load_s", "source_preprocess_s", "source_cache_hit",
    "target_load_s", "target_preprocess_s", "target_cache_hit",
    "coarse_registration_s", "fine_registration_s", "registration_s",
    "pair_wall_s",
]
FORBIDDEN_FIELD_MARKERS = ("gt_", "ground_truth", "offline_", "is_positive", "positive_count")


def parse_direction(direction: str) -> tuple[str, str]:
    parts = direction.split("_to_")
    if len(parts) != 2 or parts[0] not in ROBOTS or parts[1] not in ROBOTS or parts[0] == parts[1]:
        raise ValueError(f"Unsupported cross-robot direction: {direction}")
    return parts[0], parts[1]


def signed_shift(shift: int, sectors: int = 60) -> int:
    if shift < 0 or shift >= sectors:
        raise ValueError(f"Scan Context shift must be in [0,{sectors}): {shift}")
    return shift if shift <= sectors // 2 else shift - sectors


def yaw_initialization(best_shift: int, sectors: int = 60) -> tuple[np.ndarray, float]:
    yaw_deg = signed_shift(best_shift, sectors) * (360.0 / sectors)
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = [[cosine, -sine], [sine, cosine]]
    return transform, yaw_deg


def assert_gt_free_fields(fields: list[str]) -> None:
    bad = [field for field in fields if any(marker in field.lower() for marker in FORBIDDEN_FIELD_MARKERS)]
    if bad:
        raise ValueError(f"GT/evaluation fields are forbidden in the online manifest: {bad}")


def read_cloud_index(processed_root: Path, robot: str) -> dict[int, Path]:
    run_root = (processed_root / "full_2hz" / robot).resolve()
    index = pd.read_csv(run_root / "keyframes.csv")
    result: dict[int, Path] = {}
    for row in index.itertuples(index=False):
        relative = Path(str(row.lidar_file))
        if relative.is_absolute():
            raise ValueError(f"Absolute LiDAR path is not allowed in frozen cache: {relative}")
        if len(relative.parts) == 1:
            relative = Path("lidar") / relative
        target = (run_root / relative).resolve()
        if not target.is_relative_to(run_root) or not target.is_file():
            raise ValueError(f"Missing or escaping LiDAR path: {target}")
        result[int(row.keyframe_id)] = target
    return result


class GICPBackend:
    """Persistent GICP backend with an explicit point-cloud LRU cache."""

    def __init__(
        self,
        processed_root: Path,
        direction: str,
        *,
        cache_size: int = 64,
        voxel_size: float = 0.75,
        min_range: float = 2.0,
        max_range: float = 50.0,
        min_height: float = -3.0,
        max_height: float = 5.0,
        coarse_distance: float = 3.0,
        fine_distance: float = 1.0,
        iterations: int = 30,
    ) -> None:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("Open3D is required for Stage-11 GICP") from exc
        self.o3d = o3d
        self.query_robot, self.database_robot = parse_direction(direction)
        self.cloud_paths = {
            robot: read_cloud_index(processed_root, robot)
            for robot in (self.query_robot, self.database_robot)
        }
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[tuple[str, int], object] = OrderedDict()
        self.voxel_size = float(voxel_size)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.coarse_distance = float(coarse_distance)
        self.fine_distance = float(fine_distance)
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations))
        self.estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

    def cloud(self, robot: str, keyframe_id: int) -> tuple[object, float, float, bool]:
        key = (robot, int(keyframe_id))
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value, 0.0, 0.0, True

        load_start = time.perf_counter()
        values = np.load(self.cloud_paths[robot][int(keyframe_id)], allow_pickle=False)
        load_s = time.perf_counter() - load_start
        if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values[:, :3]).all():
            raise ValueError(f"Invalid cached point cloud: {robot}:{keyframe_id}")

        preprocess_start = time.perf_counter()
        xyz = values[:, :3]
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        keep = (
            (radius >= self.min_range) & (radius <= self.max_range)
            & (xyz[:, 2] >= self.min_height) & (xyz[:, 2] <= self.max_height)
        )
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(xyz[keep].astype(np.float64))
        cloud = cloud.voxel_down_sample(self.voxel_size)
        preprocess_s = time.perf_counter() - preprocess_start
        if len(cloud.points) < 50:
            raise ValueError(f"Too few cropped points: {robot}:{keyframe_id}")
        self.cache[key] = cloud
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return cloud, load_s, preprocess_s, False

    def register(self, row: dict[str, object]) -> dict[str, object]:
        if str(row["query_robot"]) != self.query_robot or str(row["candidate_robot"]) != self.database_robot:
            raise ValueError("Manifest direction does not match backend direction")
        wall_start = time.perf_counter()
        source, source_load, source_pre, source_hit = self.cloud(
            self.database_robot, int(row["candidate_keyframe_id"])
        )
        target, target_load, target_pre, target_hit = self.cloud(
            self.query_robot, int(row["query_keyframe_id"])
        )
        initialization, yaw_deg = yaw_initialization(int(row["best_shift"]))
        coarse_start = time.perf_counter()
        coarse = self.o3d.pipelines.registration.registration_generalized_icp(
            source, target, self.coarse_distance, initialization, self.estimation, self.criteria
        )
        coarse_s = time.perf_counter() - coarse_start
        fine_start = time.perf_counter()
        fine = self.o3d.pipelines.registration.registration_generalized_icp(
            source, target, self.fine_distance, coarse.transformation, self.estimation, self.criteria
        )
        fine_s = time.perf_counter() - fine_start
        fitness = float(fine.fitness)
        rmse = float(fine.inlier_rmse)
        return {
            **row,
            "transform_direction": "T_query_from_candidate",
            "init_yaw_deg": yaw_deg,
            "gicp_fitness": fitness,
            "gicp_inlier_rmse": rmse,
            "gicp_quality": fitness / (1.0 + rmse),
            "gicp_transform": json.dumps(np.asarray(fine.transformation).reshape(-1).tolist()),
            "source_load_s": source_load,
            "source_preprocess_s": source_pre,
            "source_cache_hit": source_hit,
            "target_load_s": target_load,
            "target_preprocess_s": target_pre,
            "target_cache_hit": target_hit,
            "coarse_registration_s": coarse_s,
            "fine_registration_s": fine_s,
            "registration_s": coarse_s + fine_s,
            "pair_wall_s": time.perf_counter() - wall_start,
        }


def load_manifest(path: Path, direction: str, top_k: int) -> tuple[list[str], list[dict[str, object]]]:
    query_robot, database_robot = parse_direction(direction)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        assert_gt_free_fields(fields)
        required = {
            "query_robot", "query_keyframe_id", "candidate_robot",
            "candidate_keyframe_id", "SC_rank", "SC_score", "best_shift",
        }
        if not required.issubset(fields):
            raise ValueError(f"Manifest lacks fields: {sorted(required - set(fields))}")
        rows = [row for row in reader if int(row["SC_rank"]) <= top_k]
    if not rows or any(
        row["query_robot"] != query_robot or row["candidate_robot"] != database_robot
        for row in rows
    ):
        raise ValueError("Manifest contains an unexpected robot direction")
    rows.sort(key=lambda row: (int(row["query_keyframe_id"]), int(row["SC_rank"])))
    return fields, rows


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    fields, rows = load_manifest(args.manifest, args.direction, args.top_k)
    if args.limit is not None:
        rows = rows[: args.limit]
    output_fields = fields + [field for field in RESULT_FIELDS if field not in fields]
    completed: set[tuple[int, int]] = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        with args.output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or []) != output_fields:
                raise ValueError("Existing output schema differs from this run")
            completed = {
                (int(row["query_keyframe_id"]), int(row["candidate_keyframe_id"]))
                for row in reader
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = GICPBackend(
        args.processed_root, args.direction, cache_size=args.cache_size,
        voxel_size=args.voxel_size, coarse_distance=args.coarse_distance,
        fine_distance=args.fine_distance, iterations=args.iterations,
    )
    mode = "a" if args.output.exists() else "w"
    written = 0
    timings: list[float] = []
    with args.output.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            pair = (int(row["query_keyframe_id"]), int(row["candidate_keyframe_id"]))
            if pair in completed:
                continue
            result = backend.register(row)
            writer.writerow(result)
            handle.flush()
            written += 1
            timings.append(float(result["pair_wall_s"]))
            if written % 50 == 0:
                print(f"GICP pairs written: {written}; query={pair[0]}", flush=True)
    return {
        "direction": args.direction,
        "requested_rows": len(rows),
        "completed_before_run": len(completed),
        "new_rows": written,
        "mean_new_pair_wall_s": float(np.mean(timings)) if timings else None,
        "output": str(args.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, choices=(1, 3), default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--voxel-size", type=float, default=0.75)
    parser.add_argument("--coarse-distance", type=float, default=3.0)
    parser.add_argument("--fine-distance", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    parse_direction(args.direction)
    print(json.dumps(run_cli(args), indent=2))


if __name__ == "__main__":
    main()
