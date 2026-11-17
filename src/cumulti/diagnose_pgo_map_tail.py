#!/usr/bin/env python3
"""Trace cross-robot map NN tails to frames, regions and nearby loop edges.

This is a GT-free diagnostic.  It loads only keyframe identity and LiDAR
filenames from processed keyframe tables, then reproduces the frozen symmetric
nearest-neighbour metric for two already-frozen trajectories.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROBOTS = ("robot1", "robot3")
POSE_COLUMNS = [
    "robot_id", "keyframe_id", "timestamp", "tx", "ty", "tz",
    "qx", "qy", "qz", "qw",
]
LOOP_COLUMNS = [
    "arrival_index", "frozen_loop_id", "query_robot", "query_keyframe_id",
    "candidate_robot", "candidate_keyframe_id",
]
RESIDUAL_COLUMNS = [
    "arrival_index", "query_keyframe_id", "candidate_keyframe_id",
    "translation_residual_m", "rotation_residual_deg",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("statistics require finite non-empty values")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def load_trajectory(path: Path) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(POSE_COLUMNS) - header
    if missing:
        raise ValueError(f"trajectory missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=POSE_COLUMNS)
    if frame.duplicated(["robot_id", "keyframe_id"]).any():
        raise ValueError("trajectory has duplicate semantic nodes")
    if set(frame.robot_id.unique()) != set(ROBOTS):
        raise ValueError("trajectory must contain exactly robot1 and robot3")
    numeric = frame[["keyframe_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("trajectory contains non-finite pose values")
    return frame.sort_values(["robot_id", "keyframe_id"]).reset_index(drop=True)


def load_lidar_index(path: Path, required_ids: list[int]) -> pd.DataFrame:
    required = {"keyframe_id", "lidar_file"}
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"keyframe table missing columns: {sorted(missing)}")
    # Position, orientation and all other offline labels are intentionally not loaded.
    frame = pd.read_csv(path, usecols=sorted(required))
    frame["keyframe_id"] = pd.to_numeric(frame.keyframe_id, errors="raise").astype(int)
    frame = frame[frame.keyframe_id.isin(required_ids)].copy()
    if frame.keyframe_id.duplicated().any():
        raise ValueError("keyframe LiDAR index has duplicate IDs")
    missing_ids = sorted(set(required_ids) - set(frame.keyframe_id))
    if missing_ids:
        raise ValueError(f"keyframe LiDAR index missing IDs: {missing_ids[:10]}")
    return frame.set_index("keyframe_id", drop=False)


def matrix_from_row(row: object) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    matrix[:3, 3] = [row.tx, row.ty, row.tz]
    return matrix


def build_labeled_map(
    trajectory: pd.DataFrame,
    keyframe_tables: dict[str, Path],
    lidar_roots: dict[str, Path],
    frame_stride: int,
    point_stride: int,
    range_min_m: float,
    range_max_m: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for robot in ROBOTS:
        subset = trajectory[trajectory.robot_id.eq(robot)].sort_values("keyframe_id")
        rows = list(subset.itertuples())[::frame_stride]
        ids = [int(row.keyframe_id) for row in rows]
        lidar_index = load_lidar_index(keyframe_tables[robot], ids)
        clouds: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for row in rows:
            keyframe_id = int(row.keyframe_id)
            lidar_path = lidar_roots[robot] / str(lidar_index.loc[keyframe_id].lidar_file)
            points = np.load(lidar_path)[::point_stride, :3]
            ranges = np.linalg.norm(points, axis=1)
            valid = (
                np.isfinite(points).all(axis=1)
                & (ranges > range_min_m)
                & (ranges < range_max_m)
            )
            points = points[valid]
            pose = matrix_from_row(row)
            clouds.append(points @ pose[:3, :3].T + pose[:3, 3])
            labels.append(np.full(len(points), keyframe_id, dtype=np.int64))
        result[robot] = (np.vstack(clouds), np.concatenate(labels))
    return result


def voxel_with_labels(
    points: np.ndarray,
    keyframe_ids: np.ndarray,
    size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / size_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices = np.sort(indices)
    return points[indices], keyframe_ids[indices]


def directed_nearest_neighbours(
    source_robot: str,
    target_robot: str,
    source_points: np.ndarray,
    source_frames: np.ndarray,
    target_points: np.ndarray,
    target_frames: np.ndarray,
) -> pd.DataFrame:
    distances, indices = cKDTree(target_points).query(source_points, k=1)
    return pd.DataFrame({
        "source_robot": source_robot,
        "source_keyframe_id": source_frames.astype(int),
        "target_robot": target_robot,
        "target_keyframe_id": target_frames[indices].astype(int),
        "distance_m": distances.astype(float),
        "source_x": source_points[:, 0],
        "source_y": source_points[:, 1],
        "source_z": source_points[:, 2],
    })


def symmetric_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.r_[
        cKDTree(second).query(first, k=1)[0],
        cKDTree(first).query(second, k=1)[0],
    ]


def threshold_summary(values: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    return {
        f"fraction_at_or_below_{value:g}m": float(np.mean(values <= value))
        for value in thresholds
    }


def grouped_distance_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    free_p90: float,
    free_p95: float,
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, sort=False).distance_m
    summary = grouped.agg(
        point_count="size",
        mean_m="mean",
        median_m="median",
        p90_m=lambda values: values.quantile(0.90),
        p95_m=lambda values: values.quantile(0.95),
        p99_m=lambda values: values.quantile(0.99),
        max_m="max",
    ).reset_index()
    tail90 = grouped.apply(lambda values: int((values > free_p90).sum())).rename(
        "points_above_free_global_p90"
    )
    tail95 = grouped.apply(lambda values: int((values > free_p95).sum())).rename(
        "points_above_free_global_p95"
    )
    summary = summary.merge(tail90.reset_index(), on=group_columns)
    summary = summary.merge(tail95.reset_index(), on=group_columns)
    summary["fraction_above_free_global_p90"] = (
        summary.points_above_free_global_p90 / summary.point_count
    )
    summary["fraction_above_free_global_p95"] = (
        summary.points_above_free_global_p95 / summary.point_count
    )
    return summary


def add_reference_regions(
    frame: pd.DataFrame,
    reference_trajectory: pd.DataFrame,
    region_size_m: float,
) -> pd.DataFrame:
    reference = reference_trajectory[["robot_id", "keyframe_id", "tx", "ty"]].rename(
        columns={"robot_id": "source_robot", "keyframe_id": "source_keyframe_id",
                 "tx": "reference_pose_x", "ty": "reference_pose_y"}
    )
    result = frame.merge(
        reference,
        on=["source_robot", "source_keyframe_id"],
        how="left",
        validate="many_to_one",
    )
    if result[["reference_pose_x", "reference_pose_y"]].isna().any(axis=None):
        raise ValueError("reference pose missing for a map point")
    result["region_x_index"] = np.floor(result.reference_pose_x / region_size_m).astype(int)
    result["region_y_index"] = np.floor(result.reference_pose_y / region_size_m).astype(int)
    return result


def distance_histogram(
    frame: pd.DataFrame,
    variant: str,
    bin_m: float,
    maximum_m: float,
) -> pd.DataFrame:
    edges = np.arange(0.0, maximum_m + bin_m, bin_m)
    rows = []
    for scope in ("symmetric", *ROBOTS):
        values = (
            frame.distance_m.to_numpy(float)
            if scope == "symmetric"
            else frame.loc[frame.source_robot.eq(scope), "distance_m"].to_numpy(float)
        )
        counts, _ = np.histogram(values, bins=edges)
        for lower, upper, count in zip(edges[:-1], edges[1:], counts):
            rows.append({
                "variant": variant,
                "scope": scope,
                "bin_lower_m": float(lower),
                "bin_upper_m": float(upper),
                "is_overflow": False,
                "point_count": int(count),
                "fraction": float(count / len(values)),
            })
        overflow = int((values >= maximum_m).sum())
        rows.append({
            "variant": variant,
            "scope": scope,
            "bin_lower_m": float(maximum_m),
            "bin_upper_m": None,
            "is_overflow": True,
            "point_count": overflow,
            "fraction": float(overflow / len(values)),
        })
    return pd.DataFrame(rows)


def analyze_variant(
    label: str,
    trajectory: pd.DataFrame,
    reference_trajectory: pd.DataFrame,
    keyframe_tables: dict[str, Path],
    lidar_roots: dict[str, Path],
    args: argparse.Namespace,
    free_thresholds: tuple[float, float] | None,
) -> tuple[
    dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    maps = build_labeled_map(
        trajectory,
        keyframe_tables,
        lidar_roots,
        args.map_frame_stride,
        args.map_point_stride,
        args.range_min_m,
        args.range_max_m,
    )
    voxelized = {
        robot: voxel_with_labels(points, frames, args.map_voxel_m)
        for robot, (points, frames) in maps.items()
    }
    first = directed_nearest_neighbours(
        "robot1", "robot3", *voxelized["robot1"], *voxelized["robot3"]
    )
    second = directed_nearest_neighbours(
        "robot3", "robot1", *voxelized["robot3"], *voxelized["robot1"]
    )
    directed = pd.concat([first, second], ignore_index=True)
    directed = add_reference_regions(directed, reference_trajectory, args.region_size_m)
    overall_stats = stats(directed.distance_m.to_numpy(float))
    if free_thresholds is None:
        free_p90 = float(overall_stats["p90"])
        free_p95 = float(overall_stats["p95"])
    else:
        free_p90, free_p95 = free_thresholds
    directional = {}
    for robot in ROBOTS:
        values = directed.loc[directed.source_robot.eq(robot), "distance_m"].to_numpy(float)
        directional[robot] = {**stats(values), **threshold_summary(values, args.distance_thresholds)}
    overall = {
        "variant": label,
        "symmetric": {**overall_stats, **threshold_summary(
            directed.distance_m.to_numpy(float), args.distance_thresholds
        )},
        "directional": directional,
        "points_robot1": int(len(voxelized["robot1"][0])),
        "points_robot3": int(len(voxelized["robot3"][0])),
        "free_global_tail_thresholds_m": {"p90": free_p90, "p95": free_p95},
    }
    frame_summary = grouped_distance_summary(
        directed, ["source_robot", "source_keyframe_id"], free_p90, free_p95
    )
    pair_summary = grouped_distance_summary(
        directed,
        ["source_robot", "source_keyframe_id", "target_robot", "target_keyframe_id"],
        free_p90,
        free_p95,
    )
    region_summary = grouped_distance_summary(
        directed,
        ["source_robot", "region_x_index", "region_y_index"],
        free_p90,
        free_p95,
    )
    tail_sample = directed.nlargest(args.tail_sample_points, "distance_m").copy()
    histogram = distance_histogram(
        directed, label, args.histogram_bin_m, args.histogram_max_m
    )
    return overall, frame_summary, pair_summary, region_summary, tail_sample, histogram


def nearest_loop_context(frames: pd.DataFrame, loops: pd.DataFrame) -> pd.DataFrame:
    endpoints = pd.concat([
        loops[["arrival_index", "frozen_loop_id", "query_robot", "query_keyframe_id"]]
        .rename(columns={"query_robot": "robot_id", "query_keyframe_id": "loop_keyframe_id"}),
        loops[["arrival_index", "frozen_loop_id", "candidate_robot", "candidate_keyframe_id"]]
        .rename(columns={"candidate_robot": "robot_id", "candidate_keyframe_id": "loop_keyframe_id"}),
    ], ignore_index=True)
    rows = []
    for row in frames.itertuples():
        candidates = endpoints[endpoints.robot_id.eq(row.source_robot)].copy()
        candidates["keyframe_gap"] = (
            candidates.loop_keyframe_id.astype(int) - int(row.source_keyframe_id)
        ).abs()
        nearest = candidates.sort_values(["keyframe_gap", "arrival_index"]).iloc[0]
        rows.append({
            "source_robot": row.source_robot,
            "source_keyframe_id": int(row.source_keyframe_id),
            "nearest_loop_arrival_index": int(nearest.arrival_index),
            "nearest_frozen_loop_id": int(nearest.frozen_loop_id),
            "nearest_loop_keyframe_id": int(nearest.loop_keyframe_id),
            "nearest_loop_keyframe_gap": int(nearest.keyframe_gap),
        })
    return frames.merge(
        pd.DataFrame(rows),
        on=["source_robot", "source_keyframe_id"],
        validate="one_to_one",
    )


def pose_delta_summary(free: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    merged = free.merge(
        candidate,
        on=["robot_id", "keyframe_id"],
        suffixes=("_free", "_candidate"),
        validate="one_to_one",
    )
    rows = []
    up = np.array([0.0, 0.0, 1.0])
    for row in merged.itertuples():
        free_rotation = Rotation.from_quat([
            row.qx_free, row.qy_free, row.qz_free, row.qw_free
        ])
        candidate_rotation = Rotation.from_quat([
            row.qx_candidate, row.qy_candidate, row.qz_candidate, row.qw_candidate
        ])
        relative = candidate_rotation * free_rotation.inv()
        translation = np.array([
            row.tx_candidate - row.tx_free,
            row.ty_candidate - row.ty_free,
            row.tz_candidate - row.tz_free,
        ])
        cosine = np.clip(
            np.dot(free_rotation.apply(up), candidate_rotation.apply(up)), -1.0, 1.0
        )
        rows.append({
            "robot_id": row.robot_id,
            "keyframe_id": int(row.keyframe_id),
            "translation_delta_m": float(np.linalg.norm(translation)),
            "rotation_delta_deg": float(np.degrees(relative.magnitude())),
            "up_delta_deg": float(np.degrees(np.arccos(cosine))),
        })
    return pd.DataFrame(rows)


def loop_local_map_consistency(
    trajectory: pd.DataFrame,
    loops: pd.DataFrame,
    keyframe_tables: dict[str, Path],
    lidar_roots: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    poses = {
        (str(row.robot_id), int(row.keyframe_id)): matrix_from_row(row)
        for row in trajectory.itertuples()
    }
    endpoint_ids = {
        robot: sorted(set(
            loops.loc[loops.query_robot.eq(robot), "query_keyframe_id"].astype(int)
        ) | set(
            loops.loc[loops.candidate_robot.eq(robot), "candidate_keyframe_id"].astype(int)
        ))
        for robot in ROBOTS
    }
    lidar_indices = {
        robot: load_lidar_index(keyframe_tables[robot], endpoint_ids[robot])
        for robot in ROBOTS
    }
    scan_cache: dict[tuple[str, int], np.ndarray] = {}

    def transformed_scan(node: tuple[str, int]) -> np.ndarray:
        if node in scan_cache:
            return scan_cache[node]
        robot, keyframe_id = node
        lidar_path = lidar_roots[robot] / str(
            lidar_indices[robot].loc[keyframe_id].lidar_file
        )
        points = np.load(lidar_path)[::args.map_point_stride, :3]
        ranges = np.linalg.norm(points, axis=1)
        points = points[
            np.isfinite(points).all(axis=1)
            & (ranges > args.range_min_m)
            & (ranges < args.range_max_m)
        ]
        pose = poses[node]
        points = points @ pose[:3, :3].T + pose[:3, 3]
        points, _ = voxel_with_labels(
            points, np.full(len(points), keyframe_id), args.map_voxel_m
        )
        scan_cache[node] = points
        return points

    rows = []
    all_distances = []
    for loop in loops.itertuples():
        query = (str(loop.query_robot), int(loop.query_keyframe_id))
        candidate = (str(loop.candidate_robot), int(loop.candidate_keyframe_id))
        distances = symmetric_distances(
            transformed_scan(query), transformed_scan(candidate)
        )
        all_distances.append(distances)
        distance_stats = stats(distances)
        row = {
            "arrival_index": int(loop.arrival_index),
            "frozen_loop_id": int(loop.frozen_loop_id),
            "query_robot": query[0],
            "query_keyframe_id": query[1],
            "candidate_robot": candidate[0],
            "candidate_keyframe_id": candidate[1],
            "point_count": int(distance_stats.pop("count")),
            **{f"distance_{key}_m": value for key, value in distance_stats.items()},
            **threshold_summary(distances, args.distance_thresholds),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    pooled = np.concatenate(all_distances)
    summary = {
        "pooled": {**stats(pooled), **threshold_summary(pooled, args.distance_thresholds)},
        "per_loop_macro": {
            "loops": int(len(frame)),
            "median_of_loop_medians_m": float(frame.distance_median_m.median()),
            "median_of_loop_p90_m": float(frame.distance_p90_m.median()),
            "median_of_loop_p95_m": float(frame.distance_p95_m.median()),
            "p95_of_loop_p95_m": float(frame.distance_p95_m.quantile(0.95)),
        },
    }
    return frame, summary


def loop_local_sensitivity(
    free: pd.DataFrame,
    candidate: pd.DataFrame,
    loops: pd.DataFrame,
    keyframe_tables: dict[str, Path],
    lidar_roots: dict[str, Path],
    args: argparse.Namespace,
) -> pd.DataFrame:
    settings = {
        (int(stride), float(args.map_voxel_m))
        for stride in args.sensitivity_point_strides
    } | {
        (int(args.map_point_stride), float(voxel))
        for voxel in args.sensitivity_voxels
    }
    rows = []
    for point_stride, voxel_m in sorted(settings):
        setting = copy.copy(args)
        setting.map_point_stride = point_stride
        setting.map_voxel_m = voxel_m
        _, free_summary = loop_local_map_consistency(
            free, loops, keyframe_tables, lidar_roots, setting
        )
        _, candidate_summary = loop_local_map_consistency(
            candidate, loops, keyframe_tables, lidar_roots, setting
        )
        row = {
            "point_stride": point_stride,
            "voxel_m": voxel_m,
        }
        for key in ("mean", "median", "p90", "p95", "p99"):
            free_value = float(free_summary["pooled"][key])
            candidate_value = float(candidate_summary["pooled"][key])
            row[f"{key}_m_free"] = free_value
            row[f"{key}_m_candidate"] = candidate_value
            row[f"{key}_m_delta"] = candidate_value - free_value
        rows.append(row)
    return pd.DataFrame(rows)


def loop_residual_comparison(
    loops: pd.DataFrame,
    free_path: Path,
    candidate_path: Path,
) -> pd.DataFrame:
    frames = []
    for label, path in (("free", free_path), ("candidate", candidate_path)):
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = set(RESIDUAL_COLUMNS) - header
        if missing:
            raise ValueError(f"loop residual CSV missing columns: {sorted(missing)}")
        frame = pd.read_csv(path, usecols=RESIDUAL_COLUMNS).rename(columns={
            "translation_residual_m": f"translation_residual_m_{label}",
            "rotation_residual_deg": f"rotation_residual_deg_{label}",
        })
        frames.append(frame)
    merged = frames[0].merge(
        frames[1],
        on=["arrival_index", "query_keyframe_id", "candidate_keyframe_id"],
        validate="one_to_one",
    )
    merged = loops[LOOP_COLUMNS].merge(
        merged,
        on=["arrival_index", "query_keyframe_id", "candidate_keyframe_id"],
        validate="one_to_one",
    )
    merged["translation_residual_delta_m"] = (
        merged.translation_residual_m_candidate - merged.translation_residual_m_free
    )
    merged["rotation_residual_delta_deg"] = (
        merged.rotation_residual_deg_candidate - merged.rotation_residual_deg_free
    )
    return merged


def merge_variant_summaries(
    free: pd.DataFrame,
    candidate: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    merged = free.merge(candidate, on=keys, suffixes=("_free", "_candidate"), validate="one_to_one")
    for column in ("mean_m", "median_m", "p90_m", "p95_m", "p99_m", "max_m",
                   "fraction_above_free_global_p90", "fraction_above_free_global_p95"):
        merged[f"{column}_delta"] = merged[f"{column}_candidate"] - merged[f"{column}_free"]
    return merged


def load_loops(path: Path) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(LOOP_COLUMNS) - header
    if missing:
        raise ValueError(f"loop table missing columns: {sorted(missing)}")
    loops = pd.read_csv(path, usecols=LOOP_COLUMNS)
    if loops.arrival_index.duplicated().any():
        raise ValueError("loop table has duplicate arrival indices")
    return loops


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    free = load_trajectory(args.free_trajectory)
    candidate = load_trajectory(args.candidate_trajectory)
    if set(zip(free.robot_id, free.keyframe_id)) != set(zip(candidate.robot_id, candidate.keyframe_id)):
        raise ValueError("free and candidate trajectories cover different semantic nodes")
    loops = load_loops(args.loop_edges)
    keyframe_tables = {"robot1": args.robot1_keyframes, "robot3": args.robot3_keyframes}
    lidar_roots = {"robot1": args.robot1_lidar_root, "robot3": args.robot3_lidar_root}

    free_result = analyze_variant(
        "free_pose3_replay", free, free, keyframe_tables, lidar_roots, args, None
    )
    free_overall, free_frames, free_pairs, free_regions, free_tail, free_histogram = free_result
    free_thresholds = (
        float(free_overall["symmetric"]["p90"]),
        float(free_overall["symmetric"]["p95"]),
    )
    candidate_result = analyze_variant(
        args.candidate_label,
        candidate,
        free,
        keyframe_tables,
        lidar_roots,
        args,
        free_thresholds,
    )
    (
        candidate_overall, candidate_frames, candidate_pairs, candidate_regions,
        candidate_tail, candidate_histogram,
    ) = candidate_result

    frame_comparison = merge_variant_summaries(
        free_frames, candidate_frames, ["source_robot", "source_keyframe_id"]
    )
    frame_comparison = nearest_loop_context(frame_comparison, loops)
    region_comparison = merge_variant_summaries(
        free_regions,
        candidate_regions,
        ["source_robot", "region_x_index", "region_y_index"],
    )
    pose_deltas = pose_delta_summary(free, candidate)
    loop_comparison = loop_residual_comparison(
        loops, args.free_loop_residuals, args.candidate_loop_residuals
    )
    free_loop_map, free_loop_map_summary = loop_local_map_consistency(
        free, loops, keyframe_tables, lidar_roots, args
    )
    candidate_loop_map, candidate_loop_map_summary = loop_local_map_consistency(
        candidate, loops, keyframe_tables, lidar_roots, args
    )
    loop_map_comparison = free_loop_map.merge(
        candidate_loop_map,
        on=[
            "arrival_index", "frozen_loop_id", "query_robot", "query_keyframe_id",
            "candidate_robot", "candidate_keyframe_id",
        ],
        suffixes=("_free", "_candidate"),
        validate="one_to_one",
    )
    for column in ("distance_mean_m", "distance_median_m", "distance_p90_m",
                   "distance_p95_m", "distance_p99_m", "distance_max_m"):
        loop_map_comparison[f"{column}_delta"] = (
            loop_map_comparison[f"{column}_candidate"]
            - loop_map_comparison[f"{column}_free"]
        )

    frame_comparison.sort_values("p95_m_delta", ascending=False).to_csv(
        output / "frame_tail_comparison.csv", index=False
    )
    free_pairs.sort_values(
        ["points_above_free_global_p90", "p95_m"], ascending=False
    ).head(args.top_pairs).to_csv(output / "top_frame_pairs_free.csv", index=False)
    candidate_pairs.sort_values(
        ["points_above_free_global_p90", "p95_m"], ascending=False
    ).head(args.top_pairs).to_csv(output / "top_frame_pairs_candidate.csv", index=False)
    region_comparison.sort_values("p95_m_delta", ascending=False).to_csv(
        output / "region_tail_comparison.csv", index=False
    )
    pose_deltas.to_csv(output / "pose_delta_by_node.csv", index=False)
    loop_comparison.sort_values("rotation_residual_delta_deg", ascending=False).to_csv(
        output / "loop_residual_comparison.csv", index=False
    )
    loop_map_comparison.sort_values("distance_p95_m_delta", ascending=False).to_csv(
        output / "loop_local_map_comparison.csv", index=False
    )
    sensitivity = loop_local_sensitivity(
        free, candidate, loops, keyframe_tables, lidar_roots, args
    )
    sensitivity.to_csv(output / "loop_local_sensitivity.csv", index=False)
    pd.concat([
        free_tail.assign(variant="free_pose3_replay"),
        candidate_tail.assign(variant=args.candidate_label),
    ], ignore_index=True).to_csv(output / "largest_distance_point_sample.csv", index=False)
    pd.concat([free_histogram, candidate_histogram], ignore_index=True).to_csv(
        output / "distance_histogram.csv", index=False
    )

    top_frames = frame_comparison.nlargest(20, "p95_m_delta")
    candidate_tail_total = int(candidate_frames.points_above_free_global_p90.sum())
    top10_tail = int(
        candidate_frames.nlargest(10, "points_above_free_global_p90")
        .points_above_free_global_p90.sum()
    )
    summary = {
        "created_UTC": pd.Timestamp.utcnow().isoformat(),
        "GT_used": False,
        "candidate_label": args.candidate_label,
        "metric_definition": "symmetric cross-robot nearest-neighbour distance after per-robot voxelization",
        "configuration": {
            "map_frame_stride": args.map_frame_stride,
            "map_point_stride": args.map_point_stride,
            "range_m": [args.range_min_m, args.range_max_m],
            "voxel_m": args.map_voxel_m,
            "region_size_m": args.region_size_m,
            "distance_thresholds_m": args.distance_thresholds,
        },
        "inputs": {
            "free_trajectory": sha256(args.free_trajectory),
            "candidate_trajectory": sha256(args.candidate_trajectory),
            "loop_edges": sha256(args.loop_edges),
            "free_loop_residuals": sha256(args.free_loop_residuals),
            "candidate_loop_residuals": sha256(args.candidate_loop_residuals),
            "robot1_keyframes": sha256(args.robot1_keyframes),
            "robot3_keyframes": sha256(args.robot3_keyframes),
        },
        "free": free_overall,
        "candidate": candidate_overall,
        "candidate_minus_free": {
            key: float(candidate_overall["symmetric"][key] - free_overall["symmetric"][key])
            for key in ("mean", "median", "p90", "p95", "p99", "max")
        },
        "tail_concentration": {
            "candidate_points_above_free_global_p90": candidate_tail_total,
            "share_in_top10_frames_by_tail_count": (
                top10_tail / candidate_tail_total if candidate_tail_total else 0.0
            ),
            "top20_frames_by_p95_delta": top_frames[[
                "source_robot", "source_keyframe_id", "p95_m_free", "p95_m_candidate",
                "p95_m_delta", "nearest_frozen_loop_id", "nearest_loop_keyframe_gap",
            ]].to_dict(orient="records"),
        },
        "pose_delta": {
            "translation_m": stats(pose_deltas.translation_delta_m.to_numpy(float)),
            "rotation_deg": stats(pose_deltas.rotation_delta_deg.to_numpy(float)),
            "up_deg": stats(pose_deltas.up_delta_deg.to_numpy(float)),
        },
        "loop_local_map": {
            "definition": "symmetric NN between the two transformed endpoint scans for each accepted loop",
            "free": free_loop_map_summary,
            "candidate": candidate_loop_map_summary,
            "candidate_minus_free": {
                key: float(
                    candidate_loop_map_summary["pooled"][key]
                    - free_loop_map_summary["pooled"][key]
                )
                for key in ("mean", "median", "p90", "p95", "p99", "max")
            },
            "loops_with_lower_p95": int((loop_map_comparison.distance_p95_m_delta < 0).sum()),
            "loops_with_higher_p95": int((loop_map_comparison.distance_p95_m_delta > 0).sum()),
            "sensitivity": {
                "settings": int(len(sensitivity)),
                "p95_delta_m": stats(sensitivity.p95_m_delta.to_numpy(float)),
                "settings_with_lower_p95": int((sensitivity.p95_m_delta < 0).sum()),
                "settings_with_higher_p95": int((sensitivity.p95_m_delta > 0).sum()),
            },
        },
        "saved_detail": {
            "full_frame_rows": int(len(frame_comparison)),
            "saved_top_pair_rows_per_variant": {
                "free": min(args.top_pairs, len(free_pairs)),
                "candidate": min(args.top_pairs, len(candidate_pairs)),
            },
            "region_rows": int(len(region_comparison)),
            "saved_tail_sample_points_per_variant": {
                "free": int(len(free_tail)),
                "candidate": int(len(candidate_tail)),
            },
        },
    }
    write_json(output / "map_tail_diagnostic_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--free-trajectory", type=Path, required=True)
    command.add_argument("--candidate-trajectory", type=Path, required=True)
    command.add_argument("--candidate-label", default="ekf_gravity_soft_cov_x8")
    command.add_argument("--loop-edges", type=Path, required=True)
    command.add_argument("--free-loop-residuals", type=Path, required=True)
    command.add_argument("--candidate-loop-residuals", type=Path, required=True)
    command.add_argument("--robot1-keyframes", type=Path, required=True)
    command.add_argument("--robot3-keyframes", type=Path, required=True)
    command.add_argument("--robot1-lidar-root", type=Path, required=True)
    command.add_argument("--robot3-lidar-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--map-frame-stride", type=int, default=10)
    command.add_argument("--map-point-stride", type=int, default=16)
    command.add_argument("--range-min-m", type=float, default=1.0)
    command.add_argument("--range-max-m", type=float, default=60.0)
    command.add_argument("--map-voxel-m", type=float, default=0.5)
    command.add_argument("--region-size-m", type=float, default=20.0)
    command.add_argument(
        "--distance-thresholds", type=float, nargs="+", default=[1, 3, 5, 10, 20, 50, 100]
    )
    command.add_argument("--tail-sample-points", type=int, default=5000)
    command.add_argument("--top-pairs", type=int, default=1000)
    command.add_argument("--histogram-bin-m", type=float, default=1.0)
    command.add_argument("--histogram-max-m", type=float, default=200.0)
    command.add_argument(
        "--sensitivity-point-strides", type=int, nargs="+", default=[8, 16, 32]
    )
    command.add_argument(
        "--sensitivity-voxels", type=float, nargs="+", default=[0.25, 0.5, 1.0]
    )
    return command


def main() -> None:
    args = parser().parse_args()
    if min(args.map_frame_stride, args.map_point_stride, args.tail_sample_points, args.top_pairs) <= 0:
        raise ValueError("stride and output count parameters must be positive")
    if not (0 < args.range_min_m < args.range_max_m):
        raise ValueError("range limits must satisfy 0 < min < max")
    if min(args.map_voxel_m, args.region_size_m, args.histogram_bin_m, args.histogram_max_m) <= 0:
        raise ValueError("voxel, region and histogram sizes must be positive")
    if not args.distance_thresholds or any(value <= 0 for value in args.distance_thresholds):
        raise ValueError("distance thresholds must be positive")
    if (
        not args.sensitivity_point_strides
        or min(args.sensitivity_point_strides) <= 0
        or not args.sensitivity_voxels
        or min(args.sensitivity_voxels) <= 0
    ):
        raise ValueError("sensitivity strides and voxels must be positive")
    run(args)


if __name__ == "__main__":
    main()
