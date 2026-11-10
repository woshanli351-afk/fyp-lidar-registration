#!/usr/bin/env python3
"""Replay Stage12D and test an EKF-gravity-preserving Pose3 diagnostic.

This is a GT-free backend experiment.  It consumes the frozen local trajectory
and loop measurements exported by Stage12D; it never reads ground-truth fields.
The transform convention is candidate source, query target:
T_query_from_candidate = X_query^-1 X_candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import gtsam
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


POSE_COLUMNS = ["robot_id", "keyframe_id", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
LOOP_COLUMNS = [
    "query_robot", "query_keyframe_id", "candidate_robot", "candidate_keyframe_id",
    "tx", "ty", "tz", "qx", "qy", "qz", "qw",
]
ROBOTS = ("robot1", "robot3")
ROBOT_SYMBOL = {"robot1": "a", "robot3": "b"}
UPSTREAM_URL = "https://github.com/James-5757/fyp-place-recognition/blob/main/src/cumulti/run_stage12d_clean_backend.py"
UPSTREAM_BLOB_SHA = "3857ca3734c35a6d8bd691cc9da7d625a83f9666"


@dataclass(frozen=True)
class Variant:
    name: str
    preserve_ekf_gravity: bool


VARIANTS = (
    Variant("free_pose3_replay", False),
    Variant("ekf_gravity_hard_diagnostic", True),
)


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


def matrix_from_row(row: object) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    matrix[:3, 3] = [row.tx, row.ty, row.tz]
    return matrix


def pose3(matrix: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(matrix[:3, :3]), gtsam.Point3(*matrix[:3, 3]))


def key(robot: str, keyframe_id: int) -> int:
    if robot not in ROBOT_SYMBOL:
        raise ValueError(f"unsupported robot: {robot}")
    return gtsam.symbol(ROBOT_SYMBOL[robot], int(keyframe_id))


def rotation_angle_deg(matrix: np.ndarray) -> float:
    return float(np.rad2deg(Rotation.from_matrix(matrix[:3, :3]).magnitude()))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def gravity_factor(
    pose_key: int,
    sensor_up: np.ndarray,
    target_common_up: np.ndarray,
    sigma_deg: float,
) -> gtsam.CustomFactor:
    """Constrain one mapped gravity vector while leaving yaw/translation free."""
    sensor_up = np.asarray(sensor_up, dtype=float)
    sensor_up /= np.linalg.norm(sensor_up)
    target_common_up = np.asarray(target_common_up, dtype=float)
    target_common_up /= np.linalg.norm(target_common_up)
    noise = gtsam.noiseModel.Isotropic.Sigma(3, math.radians(sigma_deg))

    def error_function(
        _factor: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: list[np.ndarray] | None,
    ) -> np.ndarray:
        rotation = values.atPose3(pose_key).rotation().matrix()
        if jacobians is not None:
            jacobian = np.zeros((3, 6))
            # Pose3 retracts rotation with a right/body-frame perturbation.
            jacobian[:, :3] = -rotation @ skew(sensor_up)
            jacobians[0] = jacobian
        return rotation @ sensor_up - target_common_up

    return gtsam.CustomFactor(noise, [pose_key], error_function)


def stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("statistics require finite non-empty values")
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def load_inputs(local_path: Path, loops_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_header = pd.read_csv(local_path, nrows=0).columns
    loop_header = pd.read_csv(loops_path, nrows=0).columns
    missing_pose = set(POSE_COLUMNS) - set(local_header)
    missing_loop = set(LOOP_COLUMNS) - set(loop_header)
    if missing_pose or missing_loop:
        raise ValueError(f"missing columns: pose={sorted(missing_pose)}, loops={sorted(missing_loop)}")
    # Values from GT/offline-label columns are never loaded into this process.
    local = pd.read_csv(local_path, usecols=POSE_COLUMNS)
    optional_loop_columns = [column for column in ("arrival_index", "transform_direction") if column in loop_header]
    loops = pd.read_csv(loops_path, usecols=LOOP_COLUMNS + optional_loop_columns)
    if set(local.robot_id) != set(ROBOTS):
        raise ValueError(f"expected {ROBOTS}, found {sorted(set(local.robot_id))}")
    if local.duplicated(["robot_id", "keyframe_id"]).any():
        raise ValueError("duplicate semantic node ids")
    numeric_pose = local[["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(float)
    numeric_loop = loops[["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(float)
    if not np.isfinite(numeric_pose).all() or not np.isfinite(numeric_loop).all():
        raise ValueError("non-finite pose or loop measurement")
    pose_quaternion_norm = np.linalg.norm(local[["qx", "qy", "qz", "qw"]].to_numpy(float), axis=1)
    loop_quaternion_norm = np.linalg.norm(loops[["qx", "qy", "qz", "qw"]].to_numpy(float), axis=1)
    if not np.allclose(pose_quaternion_norm, 1.0, atol=1e-5):
        raise ValueError("local trajectory contains non-unit quaternions")
    if not np.allclose(loop_quaternion_norm, 1.0, atol=1e-5):
        raise ValueError("loop table contains non-unit quaternions")
    if "transform_direction" in loops:
        expected = "T_query_from_candidate: p_query = T_query_from_candidate * p_candidate"
        if not loops.transform_direction.eq(expected).all():
            raise ValueError("loop transform direction is not T_query_from_candidate")
    for robot in ROBOTS:
        frame = local[local.robot_id.eq(robot)]
        ids = frame.keyframe_id.to_numpy(int)
        if not np.all(np.diff(ids) == 1):
            raise ValueError(f"{robot} keyframes are not consecutive")
        if not np.all(np.diff(frame.timestamp.to_numpy(np.int64)) > 0):
            raise ValueError(f"{robot} timestamps are not strictly increasing")
    nodes = set(zip(local.robot_id, local.keyframe_id.astype(int)))
    for row in loops.itertuples():
        if (row.query_robot, int(row.query_keyframe_id)) not in nodes:
            raise ValueError("loop query endpoint is absent from local trajectory")
        if (row.candidate_robot, int(row.candidate_keyframe_id)) not in nodes:
            raise ValueError("loop candidate endpoint is absent from local trajectory")
    return local, loops


def matrices_from_frame(frame: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    return {(row.robot_id, int(row.keyframe_id)): matrix_from_row(row) for row in frame.itertuples()}


def first_loop_initialization(
    local: dict[tuple[str, int], np.ndarray], loops: pd.DataFrame
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, object]]:
    first = loops.sort_values("arrival_index", kind="stable").iloc[0] if "arrival_index" in loops else loops.iloc[0]
    query = (str(first.query_robot), int(first.query_keyframe_id))
    candidate = (str(first.candidate_robot), int(first.candidate_keyframe_id))
    measurement = matrix_from_row(first)
    common_from_candidate_local = local[query] @ measurement @ np.linalg.inv(local[candidate])
    common = {
        node: matrix if node[0] == query[0] else common_from_candidate_local @ matrix
        for node, matrix in local.items()
    }
    residual = np.linalg.inv(measurement) @ np.linalg.inv(common[query]) @ common[candidate]
    translation_residual = float(np.linalg.norm(residual[:3, 3]))
    rotation_residual = rotation_angle_deg(residual)
    if translation_residual >= 1e-8 or rotation_residual >= 1e-8:
        raise AssertionError("first-loop initialization does not satisfy its measurement")
    return common, {
        "query": list(query),
        "candidate": list(candidate),
        "common_from_candidate_local": common_from_candidate_local.tolist(),
        "translation_residual_m": translation_residual,
        "rotation_residual_deg": rotation_residual,
        "GT_used": False,
    }


def build_graph(
    local_frame: pd.DataFrame,
    local: dict[tuple[str, int], np.ndarray],
    loops: pd.DataFrame,
    common_initial: dict[tuple[str, int], np.ndarray],
    variant: Variant,
    gravity_sigma_deg: float,
) -> tuple[gtsam.NonlinearFactorGraph, dict[str, int]]:
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([math.radians(5)] * 3 + [1.0] * 3)
    )
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), odometry_noise
    )
    graph = gtsam.NonlinearFactorGraph()
    first_robot1 = int(local_frame[local_frame.robot_id.eq("robot1")].keyframe_id.iloc[0])
    anchor = ("robot1", first_robot1)
    graph.add(gtsam.PriorFactorPose3(key(*anchor), pose3(local[anchor]),
                                     gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)))

    odometry_count = 0
    for robot in ROBOTS:
        ids = local_frame[local_frame.robot_id.eq(robot)].keyframe_id.astype(int).tolist()
        for before, after in zip(ids, ids[1:]):
            measurement = np.linalg.inv(local[(robot, before)]) @ local[(robot, after)]
            graph.add(gtsam.BetweenFactorPose3(key(robot, before), key(robot, after), pose3(measurement), odometry_noise))
            odometry_count += 1

    for row in loops.itertuples():
        graph.add(gtsam.BetweenFactorPose3(
            key(str(row.query_robot), int(row.query_keyframe_id)),
            key(str(row.candidate_robot), int(row.candidate_keyframe_id)),
            pose3(matrix_from_row(row)),
            loop_noise,
        ))

    gravity_factor_count = 0
    if variant.preserve_ekf_gravity:
        # The factor compares mapped EKF up vectors directly. Rotation around
        # that vector (yaw) and all translation components lie in its nullspace.
        for node, center in common_initial.items():
            if node == anchor:
                continue
            sensor_up = local[node][:3, :3].T @ np.array([0.0, 0.0, 1.0])
            target_up = center[:3, :3] @ sensor_up
            graph.add(gravity_factor(key(*node), sensor_up, target_up, gravity_sigma_deg))
            gravity_factor_count += 1

    counts = {
        "nodes": len(local),
        "odometry_factors": odometry_count,
        "loop_factors": len(loops),
        "anchor_priors": 1,
        "gravity_factors": gravity_factor_count,
        "total_factors": int(graph.size()),
    }
    return graph, counts


def values_from_matrices(matrices: dict[tuple[str, int], np.ndarray]) -> gtsam.Values:
    values = gtsam.Values()
    for node, matrix in matrices.items():
        values.insert(key(*node), pose3(matrix))
    return values


def solve(
    graph: gtsam.NonlinearFactorGraph,
    initial: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, object]]:
    values = values_from_matrices(initial)
    initial_error = float(graph.error(values))
    parameters = gtsam.LevenbergMarquardtParams()
    start = time.perf_counter()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, parameters)
    result = optimizer.optimize()
    runtime = time.perf_counter() - start
    final_error = float(graph.error(result))
    solved = {node: result.atPose3(key(*node)).matrix() for node in initial}
    if not all(np.isfinite(matrix).all() for matrix in solved.values()):
        raise AssertionError("optimizer produced a non-finite pose")
    if final_error > initial_error + 1e-8:
        raise AssertionError("optimizer increased graph error")
    return solved, {
        "initial_graph_error": initial_error,
        "final_graph_error": final_error,
        "relative_reduction": (initial_error - final_error) / max(initial_error, 1e-12),
        "runtime_s": runtime,
        "iterations": int(optimizer.iterations()),
        "max_iterations": int(parameters.getMaxIterations()),
        "normal_return": True,
    }


def trajectory_frame(template: pd.DataFrame, matrices: dict[tuple[str, int], np.ndarray]) -> pd.DataFrame:
    rows: list[list[object]] = []
    for source in template.itertuples():
        matrix = matrices[(source.robot_id, int(source.keyframe_id))]
        quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        rows.append([
            source.robot_id, int(source.keyframe_id), int(source.timestamp),
            *matrix[:3, 3], *quaternion,
        ])
    return pd.DataFrame(rows, columns=POSE_COLUMNS)


def loop_residuals(
    loops: pd.DataFrame, matrices: dict[tuple[str, int], np.ndarray]
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for index, row in enumerate(loops.itertuples(), start=1):
        query = (str(row.query_robot), int(row.query_keyframe_id))
        candidate = (str(row.candidate_robot), int(row.candidate_keyframe_id))
        measurement = matrix_from_row(row)
        error = np.linalg.inv(measurement) @ np.linalg.inv(matrices[query]) @ matrices[candidate]
        rows.append({
            "arrival_index": int(getattr(row, "arrival_index", index)),
            "query_keyframe_id": query[1],
            "candidate_keyframe_id": candidate[1],
            "translation_residual_m": float(np.linalg.norm(error[:3, 3])),
            "rotation_residual_deg": rotation_angle_deg(error),
        })
    frame = pd.DataFrame(rows)
    return frame, {
        "translation_m": stats(frame.translation_residual_m),
        "rotation_deg": stats(frame.rotation_residual_deg),
        "definition": "E = Z^-1 X_query^-1 X_candidate",
    }


def gravity_diagnostics(
    local: dict[tuple[str, int], np.ndarray],
    common_initial: dict[tuple[str, int], np.ndarray],
    solved: dict[tuple[str, int], np.ndarray],
) -> tuple[pd.DataFrame, dict[str, object]]:
    anchor_node = min((node for node in local if node[0] == "robot1"), key=lambda node: node[1])
    sensor_up_anchor = local[anchor_node][:3, :3].T @ np.array([0.0, 0.0, 1.0])
    common_reference_up = solved[anchor_node][:3, :3] @ sensor_up_anchor
    rows = []
    for node in local:
        sensor_up = local[node][:3, :3].T @ np.array([0.0, 0.0, 1.0])
        initial_up = common_initial[node][:3, :3] @ sensor_up
        solved_up = solved[node][:3, :3] @ sensor_up
        rows.append({
            "robot_id": node[0],
            "keyframe_id": node[1],
            "ekf_gravity_deviation_deg": angle_deg(initial_up, solved_up),
            "common_up_disagreement_deg": angle_deg(common_reference_up, solved_up),
        })
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "ekf_gravity_deviation_deg": stats(frame.ekf_gravity_deviation_deg),
        "common_up_disagreement_deg": stats(frame.common_up_disagreement_deg),
        "per_robot": {},
        "definition": (
            "sensor-frame EKF up is R_local^T*[0,0,1]; mapped by optimized pose. "
            "Deviation compares optimized versus first-loop-initialized EKF up; common disagreement uses Robot1 anchor up."
        ),
    }
    for robot in ROBOTS:
        subset = frame[frame.robot_id.eq(robot)]
        summary["per_robot"][robot] = {
            "ekf_gravity_deviation_deg": stats(subset.ekf_gravity_deviation_deg),
            "common_up_disagreement_deg": stats(subset.common_up_disagreement_deg),
        }
    return frame, summary


def continuity_diagnostics(
    template: pd.DataFrame,
    before: dict[tuple[str, int], np.ndarray],
    solved: dict[tuple[str, int], np.ndarray],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for robot in ROBOTS:
        ids = template[template.robot_id.eq(robot)].keyframe_id.astype(int).tolist()
        for first, second in zip(ids, ids[1:]):
            pre = np.linalg.inv(before[(robot, first)]) @ before[(robot, second)]
            post = np.linalg.inv(solved[(robot, first)]) @ solved[(robot, second)]
            pre_translation = float(np.linalg.norm(pre[:3, 3]))
            post_translation = float(np.linalg.norm(post[:3, 3]))
            rows.append({
                "robot_id": robot,
                "keyframe_i": first,
                "keyframe_j": second,
                "pre_translation_step_m": pre_translation,
                "post_translation_step_m": post_translation,
                "post_relative_rotation_deg": rotation_angle_deg(post),
                "new_step_over_5m": bool(post_translation > 5.0 and pre_translation <= 5.0),
                "new_step_over_100m": bool(post_translation >= 100.0 and pre_translation < 100.0),
            })
    frame = pd.DataFrame(rows)
    return frame, {
        "translation_step_m": stats(frame.post_translation_step_m),
        "relative_rotation_deg": stats(frame.post_relative_rotation_deg),
        "new_steps_over_5m": int(frame.new_step_over_5m.sum()),
        "new_steps_over_100m": int(frame.new_step_over_100m.sum()),
    }


def compare_to_baseline(
    baseline_path: Path,
    solved: dict[tuple[str, int], np.ndarray],
) -> dict[str, object]:
    baseline = pd.read_csv(baseline_path, usecols=POSE_COLUMNS)
    reference = matrices_from_frame(baseline)
    if set(reference) != set(solved):
        raise ValueError("baseline trajectory node set differs from replay")
    translation, rotation = [], []
    for node in solved:
        error = np.linalg.inv(reference[node]) @ solved[node]
        translation.append(float(np.linalg.norm(error[:3, 3])))
        rotation.append(rotation_angle_deg(error))
    return {
        "baseline_path": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "translation_difference_m": stats(translation),
        "rotation_difference_deg": stats(rotation),
        "same_frame_no_alignment": True,
    }


def frozen_protocol(
    local_path: Path,
    loops_path: Path,
    baseline_path: Path | None,
    gravity_sigma_deg: float,
) -> dict[str, object]:
    inputs = [local_path, loops_path] + ([baseline_path] if baseline_path else [])
    return {
        "created_UTC": utc_now(),
        "experiment": "Stage12D free Pose3 replay versus EKF-gravity-preserving diagnostic",
        "upstream_source": UPSTREAM_URL,
        "upstream_blob_sha": UPSTREAM_BLOB_SHA,
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "variants": {
            "free_pose3_replay": "exact Stage12D graph: one anchor, sequential odometry, 75 robust loops",
            "ekf_gravity_hard_diagnostic": {
                "extra_factor": "per-node mapped EKF up-vector factor around first-loop common-frame gravity",
                "gravity_vector_sigma_deg_small_angle": gravity_sigma_deg,
                "yaw_and_translation": "unconstrained nullspace of the factor",
                "interpretation": "diagnostic bound; not a production factor model",
            },
        },
        "fixed_noise": {
            "odometry_and_loop_rotation_sigma_deg": 5.0,
            "odometry_and_loop_translation_sigma_m": 1.0,
            "loop_huber_delta": 1.345,
            "noise_tangent_order": "[rx,ry,rz,tx,ty,tz]",
        },
        "pre_GT_criteria": {
            "gravity_p95_target_deg": 2.0,
            "maximum_loop_translation_p95_increase_fraction": 0.20,
            "maximum_new_adjacent_steps_over_5m": 0,
            "decision_note": "criteria characterize the diagnostic; they do not promote it to production",
        },
        "GT_used": False,
    }


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    local_path = args.local_trajectory.resolve()
    loops_path = args.loop_edges.resolve()
    baseline_path = args.baseline_trajectory.resolve() if args.baseline_trajectory else None

    protocol = frozen_protocol(local_path, loops_path, baseline_path, args.gravity_sigma_deg)
    write_json(output / "frozen_protocol.json", protocol)
    local_frame, loops = load_inputs(local_path, loops_path)
    local = matrices_from_frame(local_frame)
    common_initial, initialization = first_loop_initialization(local, loops)
    write_json(output / "first_loop_initialization.json", initialization)

    summaries: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        print(json.dumps({"stage": "SOLVING", "variant": variant.name, "UTC": utc_now()}), flush=True)
        graph, counts = build_graph(
            local_frame, local, loops, common_initial, variant,
            args.gravity_sigma_deg,
        )
        solved, solver = solve(graph, common_initial)
        trajectory_frame(local_frame, solved).to_csv(output / f"{variant.name}_trajectory.csv", index=False)
        residual_frame, residual_summary = loop_residuals(loops, solved)
        residual_frame.to_csv(output / f"{variant.name}_loop_residuals.csv", index=False)
        gravity_frame, gravity_summary = gravity_diagnostics(local, common_initial, solved)
        gravity_frame.to_csv(output / f"{variant.name}_gravity.csv", index=False)
        continuity_frame, continuity_summary = continuity_diagnostics(local_frame, common_initial, solved)
        continuity_frame.to_csv(output / f"{variant.name}_continuity.csv", index=False)
        summaries[variant.name] = {
            "factor_counts": counts,
            "solver": solver,
            "loop_residuals": residual_summary,
            "gravity": gravity_summary,
            "continuity": continuity_summary,
            "GT_used": False,
        }
        if baseline_path and variant.name == "free_pose3_replay":
            summaries[variant.name]["upstream_reproduction"] = compare_to_baseline(baseline_path, solved)
        write_json(output / f"{variant.name}_summary.json", summaries[variant.name])

    free = summaries["free_pose3_replay"]
    hard = summaries["ekf_gravity_hard_diagnostic"]
    free_loop_p95 = float(free["loop_residuals"]["translation_m"]["p95"])
    hard_loop_p95 = float(hard["loop_residuals"]["translation_m"]["p95"])
    hard_gravity_p95 = float(hard["gravity"]["common_up_disagreement_deg"]["p95"])
    criteria = protocol["pre_GT_criteria"]
    checks = {
        "gravity_p95_at_or_below_target": hard_gravity_p95 <= criteria["gravity_p95_target_deg"],
        "loop_translation_p95_within_increase_limit": hard_loop_p95 <= free_loop_p95 * (
            1.0 + criteria["maximum_loop_translation_p95_increase_fraction"]
        ),
        "no_new_adjacent_steps_over_5m": hard["continuity"]["new_steps_over_5m"] == 0,
    }
    decision = {
        "created_UTC": utc_now(),
        "decision": "DIAGNOSTIC_CRITERIA_PASS" if all(checks.values()) else "DIAGNOSTIC_CRITERIA_NOT_MET",
        "checks": checks,
        "comparison": {
            "free_common_up_p95_deg": free["gravity"]["common_up_disagreement_deg"]["p95"],
            "hard_common_up_p95_deg": hard_gravity_p95,
            "free_loop_translation_p95_m": free_loop_p95,
            "hard_loop_translation_p95_m": hard_loop_p95,
            "free_new_steps_over_5m": free["continuity"]["new_steps_over_5m"],
            "hard_new_steps_over_5m": hard["continuity"]["new_steps_over_5m"],
        },
        "interpretation": (
            "A pass supports the hypothesis that unconstrained tilt is a major backend failure mode; "
            "it does not establish global accuracy or justify deployment without offline GT evaluation."
        ),
        "GT_used": False,
    }
    write_json(output / "pre_gt_decision.json", decision)
    write_json(output / "run_environment.json", {
        "created_UTC": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "gtsam_version": getattr(gtsam, "__version__", "4.2 wheel"),
        "command": sys.argv,
    })
    for frozen_input in protocol["inputs"]:
        if sha256(Path(frozen_input["path"])) != frozen_input["sha256"]:
            raise AssertionError(f"frozen input changed during solve: {frozen_input['path']}")
    print(json.dumps(decision, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--local-trajectory", type=Path, required=True)
    command.add_argument("--loop-edges", type=Path, required=True)
    command.add_argument("--baseline-trajectory", type=Path)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--gravity-sigma-deg", type=float, default=0.01)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.gravity_sigma_deg <= 0:
        raise ValueError("gravity-vector noise sigma must be positive")
    run(args)


if __name__ == "__main__":
    main()
