#!/usr/bin/env python3
"""Offline-only GT and point-cloud evaluation for frozen PGO trajectories.

This program is deliberately separate from the solver.  It freezes trajectory
and policy hashes before reading GT-bearing keyframe tables, never optimizes a
pose, and never changes an online/pre-GT decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROBOTS = ("robot1", "robot3")
POSE_COLUMNS = ["robot_id", "keyframe_id", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
GT_COLUMNS = ["keyframe_id", "x", "y", "z", "qx", "qy", "qz", "qw", "lidar_file"]
VARIANTS = ("free_pose3_replay", "ekf_gravity_hard_diagnostic")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pose_matrix(row: object) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    matrix[:3, 3] = [row.tx, row.ty, row.tz]
    return matrix


def gt_matrix(row: object) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    matrix[:3, 3] = [row.x, row.y, row.z]
    return matrix


def rotation_angle_deg(matrix: np.ndarray) -> float:
    return float(np.rad2deg(Rotation.from_matrix(matrix[:3, :3]).magnitude()))


def rigid_align(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No-scale Kabsch alignment, matching the Stage12D joint ATE protocol."""
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("alignment inputs must have matching Nx3 shapes")
    predicted_center = predicted.mean(axis=0)
    reference_center = reference.mean(axis=0)
    left, _, right = np.linalg.svd((predicted - predicted_center).T @ (reference - reference_center))
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = reference_center - rotation @ predicted_center
    aligned = predicted @ rotation.T + translation
    return aligned, rotation, translation


def load_trajectory(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = set(POSE_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"trajectory missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=POSE_COLUMNS)
    if set(frame.robot_id) != set(ROBOTS):
        raise ValueError(f"expected {ROBOTS}, found {sorted(set(frame.robot_id))}")
    if frame.duplicated(["robot_id", "keyframe_id"]).any():
        raise ValueError("trajectory contains duplicate semantic ids")
    values = frame[["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("trajectory contains non-finite values")
    return frame


def load_gt_keyframes(path: Path, required_ids: Iterable[int]) -> pd.DataFrame:
    """Load GT only inside the offline phase and only for required keyframes."""
    header = pd.read_csv(path, nrows=0).columns
    missing = set(GT_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"GT keyframe table missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=GT_COLUMNS)
    if frame.keyframe_id.duplicated().any():
        raise ValueError(f"duplicate GT keyframe ids in {path}")
    required = set(int(value) for value in required_ids)
    frame = frame[frame.keyframe_id.astype(int).isin(required)].copy()
    found = set(frame.keyframe_id.astype(int))
    if found != required:
        raise ValueError(f"GT keyframe coverage mismatch; missing={sorted(required - found)[:10]}")
    numeric = frame[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("GT keyframe table contains non-finite values")
    frame["keyframe_id"] = frame.keyframe_id.astype(int)
    return frame.set_index("keyframe_id", drop=False).sort_index()


def semantic_arrays(
    trajectory: pd.DataFrame,
    gt_by_robot: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted, reference, robots = [], [], []
    for row in trajectory.itertuples():
        gt = gt_by_robot[row.robot_id].loc[int(row.keyframe_id)]
        predicted.append([row.tx, row.ty, row.tz])
        reference.append([gt.x, gt.y, gt.z])
        robots.append(row.robot_id)
    return np.asarray(predicted, float), np.asarray(reference, float), np.asarray(robots, object)


def ate_rows(trajectory: pd.DataFrame, gt_by_robot: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    predicted, reference, robots = semantic_arrays(trajectory, gt_by_robot)
    aligned, _, _ = rigid_align(predicted, reference)
    joint_error = np.linalg.norm(aligned - reference, axis=1)
    rows: list[dict[str, object]] = []
    for scope, mask in [("joint", np.ones(len(robots), dtype=bool))] + [
        (robot, robots == robot) for robot in ROBOTS
    ]:
        error = joint_error[mask]
        rows.append({
            "metric_type": "ATE_joint_rigid_alignment",
            "scope": scope,
            "sample_count": int(len(error)),
            "ATE_RMSE_m": float(np.sqrt(np.mean(np.square(error)))),
            "ATE_median_m": float(np.median(error)),
            "ATE_p95_m": float(np.percentile(error, 95)),
        })
    for robot in ROBOTS:
        mask = robots == robot
        independently_aligned, _, _ = rigid_align(predicted[mask], reference[mask])
        error = np.linalg.norm(independently_aligned - reference[mask], axis=1)
        rows.append({
            "metric_type": "ATE_independent_rigid_alignment",
            "scope": robot,
            "sample_count": int(len(error)),
            "ATE_RMSE_m": float(np.sqrt(np.mean(np.square(error)))),
            "ATE_median_m": float(np.median(error)),
            "ATE_p95_m": float(np.percentile(error, 95)),
        })
    return rows


def trajectory_lookup(trajectory: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    return {(row.robot_id, int(row.keyframe_id)): pose_matrix(row) for row in trajectory.itertuples()}


def gt_lookup(gt_by_robot: dict[str, pd.DataFrame]) -> dict[tuple[str, int], np.ndarray]:
    return {
        (robot, int(row.keyframe_id)): gt_matrix(row)
        for robot, frame in gt_by_robot.items()
        for row in frame.itertuples()
    }


def rpe_rows(
    trajectory: pd.DataFrame,
    gt_by_robot: dict[str, pd.DataFrame],
    interval: int,
) -> list[dict[str, object]]:
    predicted = trajectory_lookup(trajectory)
    reference = gt_lookup(gt_by_robot)
    rows: list[dict[str, object]] = []
    combined_translation: list[float] = []
    combined_rotation: list[float] = []
    for robot in ROBOTS:
        ids = trajectory[trajectory.robot_id.eq(robot)].keyframe_id.astype(int).tolist()
        translations, rotations = [], []
        for first, second in zip(ids[:-interval], ids[interval:]):
            if second - first != interval:
                raise ValueError(f"{robot} keyframe ids are not consecutive at {first}->{second}")
            gt_relative = np.linalg.inv(reference[(robot, first)]) @ reference[(robot, second)]
            predicted_relative = np.linalg.inv(predicted[(robot, first)]) @ predicted[(robot, second)]
            error = np.linalg.inv(gt_relative) @ predicted_relative
            translations.append(float(np.linalg.norm(error[:3, 3])))
            rotations.append(rotation_angle_deg(error))
        combined_translation.extend(translations)
        combined_rotation.extend(rotations)
        rows.append({
            "metric_type": "RPE",
            "scope": robot,
            "sample_count": len(translations),
            "interval_keyframes": interval,
            "translation_RMSE_m": float(np.sqrt(np.mean(np.square(translations)))),
            "rotation_RMSE_deg": float(np.sqrt(np.mean(np.square(rotations)))),
        })
    rows.append({
        "metric_type": "RPE",
        "scope": "combined",
        "sample_count": len(combined_translation),
        "interval_keyframes": interval,
        "translation_RMSE_m": float(np.sqrt(np.mean(np.square(combined_translation)))),
        "rotation_RMSE_deg": float(np.sqrt(np.mean(np.square(combined_rotation)))),
    })
    return rows


def voxel(points: np.ndarray, size_m: float) -> np.ndarray:
    keys = np.floor(points / size_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def symmetric_map_consistency(robot1: np.ndarray, robot3: np.ndarray, voxel_m: float) -> dict[str, object]:
    first, second = voxel(robot1, voxel_m), voxel(robot3, voxel_m)
    distances = np.r_[cKDTree(second).query(first, k=1)[0], cKDTree(first).query(second, k=1)[0]]
    return {
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "points_robot1": int(len(first)),
        "points_robot3": int(len(second)),
        "definition": "symmetric nearest-neighbour distance after per-robot voxelization",
    }


def build_map_points(
    trajectory: pd.DataFrame,
    gt_by_robot: dict[str, pd.DataFrame],
    lidar_roots: dict[str, Path],
    frame_stride: int,
    point_stride: int,
    range_min_m: float,
    range_max_m: float,
) -> dict[str, np.ndarray]:
    poses = trajectory_lookup(trajectory)
    points_by_robot: dict[str, np.ndarray] = {}
    for robot in ROBOTS:
        ids = trajectory[trajectory.robot_id.eq(robot)].keyframe_id.astype(int).tolist()[::frame_stride]
        clouds = []
        for keyframe_id in ids:
            lidar_file = str(gt_by_robot[robot].loc[keyframe_id].lidar_file)
            path = lidar_roots[robot] / lidar_file
            points = np.load(path)[::point_stride, :3]
            ranges = np.linalg.norm(points, axis=1)
            points = points[np.isfinite(points).all(axis=1) & (ranges > range_min_m) & (ranges < range_max_m)]
            pose = poses[(robot, keyframe_id)]
            clouds.append(points @ pose[:3, :3].T + pose[:3, 3])
        points_by_robot[robot] = np.vstack(clouds)
    return points_by_robot


def expected_metric_audit(
    measured: pd.DataFrame,
    expected_path: Path,
    free_variant: str,
    tolerance: float,
) -> dict[str, object]:
    expected = pd.read_csv(expected_path)
    expected = expected[expected.condition.eq("ALL_75_LOOPS_CLEAN_GTSAM")]
    checks = []
    for row in expected.itertuples():
        actual = measured[
            measured.variant.eq(free_variant)
            & measured.metric_type.eq(row.metric_type)
            & measured.scope.eq(row.scope)
        ]
        if len(actual) != 1:
            raise ValueError(f"missing measured baseline row: {row.metric_type}/{row.scope}")
        actual_row = actual.iloc[0]
        columns = ["ATE_RMSE_m"] if str(row.metric_type).startswith("ATE") else [
            "translation_RMSE_m", "rotation_RMSE_deg"
        ]
        for column in columns:
            delta = abs(float(actual_row[column]) - float(getattr(row, column)))
            checks.append({"metric_type": row.metric_type, "scope": row.scope, "column": column, "absolute_delta": delta})
    passed = all(item["absolute_delta"] <= tolerance for item in checks)
    return {"passed": passed, "absolute_tolerance": tolerance, "checks": checks, "expected_sha256": sha256(expected_path)}


def expected_map_audit(
    measured: pd.DataFrame,
    expected_path: Path,
    free_variant: str,
    tolerance: float,
) -> dict[str, object]:
    expected = pd.read_csv(expected_path)
    expected = expected[expected.condition.eq("ALL_75_LOOPS_CLEAN_GTSAM")]
    actual = measured[measured.variant.eq(free_variant)]
    if len(expected) != 1 or len(actual) != 1:
        raise ValueError("expected exactly one free baseline map row")
    checks = []
    for column in ("median_m", "p90_m", "p95_m"):
        checks.append({"column": column, "absolute_delta": abs(float(actual.iloc[0][column]) - float(expected.iloc[0][column]))})
    for column in ("points_robot1", "points_robot3"):
        checks.append({"column": column, "absolute_delta": abs(int(actual.iloc[0][column]) - int(expected.iloc[0][column]))})
    passed = all(
        item["absolute_delta"] <= (0 if item["column"].startswith("points_") else tolerance)
        for item in checks
    )
    return {"passed": passed, "absolute_tolerance_m": tolerance, "checks": checks, "expected_sha256": sha256(expected_path)}


def freeze_before_gt(args: argparse.Namespace, output: Path) -> dict[str, object]:
    pre_gt = json.loads(args.pre_gt_decision.read_text(encoding="utf-8"))
    if pre_gt.get("GT_used") is not False:
        raise ValueError("pre-GT decision does not assert GT_used=false")
    value = {
        "created_UTC": utc_now(),
        "phase": "OFFLINE_EVALUATION_INPUTS_FROZEN_BEFORE_GT_LOAD",
        "pre_gt_decision": {"path": str(args.pre_gt_decision.resolve()), "sha256": sha256(args.pre_gt_decision)},
        "trajectories": {
            "free_pose3_replay": {"path": str(args.free_trajectory.resolve()), "sha256": sha256(args.free_trajectory)},
            "ekf_gravity_hard_diagnostic": {
                "path": str(args.gravity_trajectory.resolve()), "sha256": sha256(args.gravity_trajectory)
            },
        },
        "offline_GT_inputs_not_yet_parsed": {
            "robot1_keyframes": {"path": str(args.robot1_keyframes.resolve()), "sha256": sha256(args.robot1_keyframes)},
            "robot3_keyframes": {"path": str(args.robot3_keyframes.resolve()), "sha256": sha256(args.robot3_keyframes)},
        },
        "evaluation_policy": {
            "ATE": "joint no-scale rigid alignment plus per-robot scoped and independent errors",
            "RPE_interval_keyframes": args.rpe_interval,
            "map_frame_stride": args.map_frame_stride,
            "map_point_stride": args.map_point_stride,
            "map_voxel_m": args.map_voxel_m,
            "map_range_m": [args.range_min_m, args.range_max_m],
        },
        "GT_used": False,
        "GT_values_loaded": False,
    }
    write_json(output / "offline_evaluation_freeze.json", value)
    return value


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = freeze_before_gt(args, output)
    trajectories = {
        "free_pose3_replay": load_trajectory(args.free_trajectory),
        "ekf_gravity_hard_diagnostic": load_trajectory(args.gravity_trajectory),
    }
    if set(zip(trajectories[VARIANTS[0]].robot_id, trajectories[VARIANTS[0]].keyframe_id)) != set(
        zip(trajectories[VARIANTS[1]].robot_id, trajectories[VARIANTS[1]].keyframe_id)
    ):
        raise ValueError("variant trajectories do not cover the same semantic nodes")

    gt_by_robot = {}
    for robot, path in (("robot1", args.robot1_keyframes), ("robot3", args.robot3_keyframes)):
        ids = trajectories[VARIANTS[0]][trajectories[VARIANTS[0]].robot_id.eq(robot)].keyframe_id.astype(int)
        gt_by_robot[robot] = load_gt_keyframes(path, ids)

    metric_rows = []
    for variant, trajectory in trajectories.items():
        for row in ate_rows(trajectory, gt_by_robot) + rpe_rows(trajectory, gt_by_robot, args.rpe_interval):
            metric_rows.append({"variant": variant, **row, "use": "OFFLINE GT ONLY"})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "offline_gt_metrics.csv", index=False)

    baseline_audit = None
    if args.expected_free_metrics:
        baseline_audit = expected_metric_audit(metrics, args.expected_free_metrics, VARIANTS[0], args.reproduction_tolerance)
        write_json(output / "free_baseline_reproduction.json", baseline_audit)
        if not baseline_audit["passed"]:
            raise AssertionError("free baseline offline metrics do not reproduce the frozen Stage12D results")

    map_rows = []
    map_audit = None
    if args.robot1_lidar_root and args.robot3_lidar_root:
        lidar_roots = {"robot1": args.robot1_lidar_root, "robot3": args.robot3_lidar_root}
        for variant, trajectory in trajectories.items():
            points = build_map_points(
                trajectory, gt_by_robot, lidar_roots,
                args.map_frame_stride, args.map_point_stride,
                args.range_min_m, args.range_max_m,
            )
            map_rows.append({"variant": variant, **symmetric_map_consistency(
                points["robot1"], points["robot3"], args.map_voxel_m
            )})
        map_frame = pd.DataFrame(map_rows)
        map_frame.to_csv(output / "map_consistency.csv", index=False)
        if args.expected_free_map:
            map_audit = expected_map_audit(map_frame, args.expected_free_map, VARIANTS[0], args.reproduction_tolerance)
            write_json(output / "free_map_reproduction.json", map_audit)
            if not map_audit["passed"]:
                raise AssertionError("free baseline map metrics do not reproduce the frozen Stage12D results")
    elif args.robot1_lidar_root or args.robot3_lidar_root:
        raise ValueError("both LiDAR roots are required for map evaluation")

    for item in freeze["trajectories"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise AssertionError(f"trajectory changed during offline evaluation: {item['path']}")
    for item in freeze["offline_GT_inputs_not_yet_parsed"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise AssertionError(f"GT keyframe input changed during evaluation: {item['path']}")
    summary = {
        "created_UTC": utc_now(),
        "offline_GT_rows": len(metrics),
        "map_evaluation_completed": bool(map_rows),
        "free_baseline_reproduction": baseline_audit,
        "free_map_reproduction": map_audit,
        "pre_gt_decision_unchanged": sha256(args.pre_gt_decision) == freeze["pre_gt_decision"]["sha256"],
        "solver_rerun": False,
        "GT_used": True,
    }
    write_json(output / "offline_evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--pre-gt-decision", type=Path, required=True)
    command.add_argument("--free-trajectory", type=Path, required=True)
    command.add_argument("--gravity-trajectory", type=Path, required=True)
    command.add_argument("--robot1-keyframes", type=Path, required=True)
    command.add_argument("--robot3-keyframes", type=Path, required=True)
    command.add_argument("--robot1-lidar-root", type=Path)
    command.add_argument("--robot3-lidar-root", type=Path)
    command.add_argument("--expected-free-metrics", type=Path)
    command.add_argument("--expected-free-map", type=Path)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--rpe-interval", type=int, default=10)
    command.add_argument("--map-frame-stride", type=int, default=10)
    command.add_argument("--map-point-stride", type=int, default=16)
    command.add_argument("--map-voxel-m", type=float, default=0.5)
    command.add_argument("--range-min-m", type=float, default=1.0)
    command.add_argument("--range-max-m", type=float, default=60.0)
    command.add_argument("--reproduction-tolerance", type=float, default=1e-6)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.rpe_interval <= 0 or args.map_frame_stride <= 0 or args.map_point_stride <= 0:
        raise ValueError("interval and stride values must be positive")
    if args.map_voxel_m <= 0 or not 0 <= args.range_min_m < args.range_max_m:
        raise ValueError("invalid map voxel or range policy")
    if args.expected_free_map and not (args.robot1_lidar_root and args.robot3_lidar_root):
        raise ValueError("expected free map audit requires both LiDAR roots")
    run(args)


if __name__ == "__main__":
    main()
