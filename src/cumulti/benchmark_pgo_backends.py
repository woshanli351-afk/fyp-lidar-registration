#!/usr/bin/env python3
"""Benchmark odometry chaining, batch LM, or persistent iSAM2 on frozen Stage12D inputs.

This process is GT-free.  Each invocation runs exactly one backend so peak RSS
is comparable across independent processes.  Full trajectories stay in the
private experiment directory; the public evaluator emits aggregate results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import gtsam
import numpy as np
import pandas as pd

import run_pgo_gravity_validation as pgo


MODES = ("odometry_only", "batch_lm", "isam2_default")
FIRST_QUERY_KEYFRAME = 283
FIRST_CANDIDATE_KEYFRAME = 252
ROBOT1_START = 150
ROBOT3_START = 234
ROBOT1_END = 1999
ROBOT3_END = 4179
EXPECTED_NODES = 5796
EXPECTED_ODOMETRY = 5794
EXPECTED_LOOPS = 75
EXPECTED_FACTORS = 5870


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def latency_stats(values_s: list[float]) -> dict[str, float | int]:
    values = np.asarray(values_s, dtype=float) * 1000.0
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("latency statistics require finite non-empty values")
    return {
        "count": int(len(values)),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def graph_subset(template: gtsam.NonlinearFactorGraph, indices: list[int]) -> gtsam.NonlinearFactorGraph:
    graph = gtsam.NonlinearFactorGraph()
    for index in indices:
        graph.add(template.at(index))
    return graph


def pose_difference(
    reference: dict[tuple[str, int], np.ndarray],
    candidate: dict[tuple[str, int], np.ndarray],
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError("pose comparison requires identical semantic node sets")
    translations, rotations = [], []
    for node in reference:
        translations.append(float(np.linalg.norm(reference[node][:3, 3] - candidate[node][:3, 3])))
        delta = reference[node][:3, :3].T @ candidate[node][:3, :3]
        rotations.append(pgo.rotation_angle_deg(np.block([
            [delta, np.zeros((3, 1))],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ])))
    return {
        "translation_m": pgo.stats(translations),
        "rotation_deg": pgo.stats(rotations),
        "same_gauge_no_alignment": True,
    }


def compose_odometry(
    local_frame: pd.DataFrame,
    local: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    start = time.perf_counter()
    composed: dict[tuple[str, int], np.ndarray] = {}
    edges = 0
    for robot in pgo.ROBOTS:
        ids = local_frame[local_frame.robot_id.eq(robot)].keyframe_id.astype(int).tolist()
        composed[(robot, ids[0])] = local[(robot, ids[0])].copy()
        for before, after in zip(ids, ids[1:]):
            measurement = np.linalg.inv(local[(robot, before)]) @ local[(robot, after)]
            composed[(robot, after)] = composed[(robot, before)] @ measurement
            edges += 1
    elapsed = time.perf_counter() - start
    maximum = max(float(np.max(np.abs(composed[node] - local[node]))) for node in composed)
    if maximum >= 1e-8:
        raise AssertionError("odometry composition did not reproduce the frozen local trajectory")
    return composed, {
        "backend_wall_s": elapsed,
        "odometry_edges_composed": edges,
        "connected_components": 2,
        "common_cross_robot_frame": False,
        "maximum_local_pose_matrix_difference": maximum,
    }


def build_full_graph(
    local_frame: pd.DataFrame,
    local: dict[tuple[str, int], np.ndarray],
    loops: pd.DataFrame,
    common: dict[tuple[str, int], np.ndarray],
) -> tuple[gtsam.NonlinearFactorGraph, dict[str, int], float]:
    start = time.perf_counter()
    graph, counts = pgo.build_graph(
        local_frame,
        local,
        loops,
        common,
        pgo.VARIANTS[0],
        gravity_sigma_deg=0.01,
    )
    elapsed = time.perf_counter() - start
    expected = {
        "nodes": EXPECTED_NODES,
        "odometry_factors": EXPECTED_ODOMETRY,
        "loop_factors": EXPECTED_LOOPS,
        "anchor_priors": 1,
        "gravity_factors": 0,
        "gravity_covariance_missing_nodes": 0,
        "total_factors": EXPECTED_FACTORS,
    }
    if counts != expected or graph.size() != EXPECTED_FACTORS:
        raise AssertionError(f"unexpected graph counts: {counts}")
    return graph, counts, elapsed


def run_batch(
    local_frame: pd.DataFrame,
    local: dict[tuple[str, int], np.ndarray],
    loops: pd.DataFrame,
    common: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    backend_start = time.perf_counter()
    graph, counts, graph_s = build_full_graph(local_frame, local, loops, common)
    solved, solver = pgo.solve(graph, common)
    backend_s = time.perf_counter() - backend_start
    return solved, {
        "backend_wall_s": backend_s,
        "graph_construction_s": graph_s,
        "solver_core_s": float(solver["runtime_s"]),
        "estimate_extraction_included_in_backend_wall": True,
        "factor_counts": counts,
        "solver": solver,
    }


def run_isam2(
    local_frame: pd.DataFrame,
    local: dict[tuple[str, int], np.ndarray],
    loops: pd.DataFrame,
    common: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    backend_start = time.perf_counter()
    full_graph, counts, graph_s = build_full_graph(local_frame, local, loops, common)
    ordered = loops.sort_values("arrival_index", kind="stable").copy() if "arrival_index" in loops else loops.copy()
    if len(ordered) != EXPECTED_LOOPS:
        raise AssertionError("iSAM2 replay requires all 75 frozen loops")
    first = ordered.iloc[0]
    if (int(first.query_keyframe_id), int(first.candidate_keyframe_id)) != (
        FIRST_QUERY_KEYFRAME,
        FIRST_CANDIDATE_KEYFRAME,
    ):
        raise AssertionError("unexpected first loop")
    if not ordered.query_keyframe_id.is_monotonic_increasing:
        raise AssertionError("loop arrival order is not monotonic")
    grouped = {int(query): group for query, group in ordered.groupby("query_keyframe_id", sort=False)}
    if any(len(group) > 1 for group in grouped.values()):
        raise AssertionError("frozen scheduler expects at most one loop per Robot1 keyframe")

    params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(params)
    first_indices = (
        [0]
        + list(range(1, FIRST_QUERY_KEYFRAME - ROBOT1_START + 1))
        + list(range(1850, 5795))
        + [5795]
    )
    first_nodes = {
        node: matrix
        for node, matrix in common.items()
        if node[0] == "robot3" or node[1] <= FIRST_QUERY_KEYFRAME
    }
    if len(first_indices) != 4080 or len(first_nodes) != 4080:
        raise AssertionError("first-fusion graph must contain 4080 nodes/factors")

    construction_s = 0.0
    update_by_type: dict[str, list[float]] = {
        "first_fusion": [],
        "odometry_update": [],
        "loop_update": [],
    }
    extraction_s = 0.0
    start = time.perf_counter()
    first_graph = graph_subset(full_graph, first_indices)
    first_values = pgo.values_from_matrices(first_nodes)
    construction_s += time.perf_counter() - start
    start = time.perf_counter()
    isam.update(first_graph, first_values)
    update_by_type["first_fusion"].append(time.perf_counter() - start)
    start = time.perf_counter()
    previous_pose = isam.calculateEstimatePose3(pgo.key("robot1", FIRST_QUERY_KEYFRAME))
    extraction_s += time.perf_counter() - start

    inserted_loops = 1
    inserted_odometry = 4078
    for keyframe in range(FIRST_QUERY_KEYFRAME + 1, ROBOT1_END + 1):
        start = time.perf_counter()
        new_graph = gtsam.NonlinearFactorGraph()
        new_graph.add(full_graph.at(keyframe - ROBOT1_START))
        arrived = grouped.get(keyframe)
        if arrived is not None:
            row = arrived.iloc[0]
            arrival_index = int(getattr(row, "arrival_index", ordered.index.get_loc(row.name) + 1))
            new_graph.add(full_graph.at(5795 + arrival_index - 1))
            inserted_loops += 1
            event_type = "loop_update"
        else:
            event_type = "odometry_update"
        measurement = np.linalg.inv(local[("robot1", keyframe - 1)]) @ local[("robot1", keyframe)]
        guess = previous_pose.matrix() @ measurement
        new_values = gtsam.Values()
        new_values.insert(pgo.key("robot1", keyframe), pgo.pose3(guess))
        construction_s += time.perf_counter() - start
        start = time.perf_counter()
        isam.update(new_graph, new_values)
        update_by_type[event_type].append(time.perf_counter() - start)
        inserted_odometry += 1
        start = time.perf_counter()
        previous_pose = isam.calculateEstimatePose3(pgo.key("robot1", keyframe))
        extraction_s += time.perf_counter() - start

    start = time.perf_counter()
    estimate = isam.calculateEstimate()
    extraction_s += time.perf_counter() - start
    nodes = [(row.robot_id, int(row.keyframe_id)) for row in local_frame.itertuples()]
    solved = {node: estimate.atPose3(pgo.key(*node)).matrix() for node in nodes}
    backend_s = time.perf_counter() - backend_start
    final_factors = int(isam.getFactorsUnsafe().size())
    if (estimate.size(), inserted_odometry, inserted_loops, final_factors) != (
        EXPECTED_NODES,
        EXPECTED_ODOMETRY,
        EXPECTED_LOOPS,
        EXPECTED_FACTORS,
    ):
        raise AssertionError("final iSAM2 graph is not equivalent to the frozen batch graph")
    final_error = float(full_graph.error(estimate))
    return solved, {
        "backend_wall_s": backend_s,
        "full_factor_template_construction_s": graph_s,
        "incremental_event_construction_s": construction_s,
        "estimate_extraction_total_s": extraction_s,
        "update_total_s": float(sum(sum(values) for values in update_by_type.values())),
        "update_latency": {name: latency_stats(values) for name, values in update_by_type.items()},
        "factor_counts": counts,
        "final_graph_error": final_error,
        "isam2_parameters": {
            "relinearize_skip": int(params.relinearizeSkip),
            "factorization": str(params.getFactorization()),
            "optimization": "installed default GaussNewton",
            "threshold": "0.1 installed default; scalar getter unavailable",
            "extra_empty_updates": 0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--local-trajectory", type=Path, required=True)
    parser.add_argument("--loop-edges", type=Path, required=True)
    parser.add_argument("--batch-baseline", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_start = time.perf_counter()
    input_start = time.perf_counter()
    local_frame, loops = pgo.load_inputs(args.local_trajectory.resolve(), args.loop_edges.resolve())
    local = pgo.matrices_from_frame(local_frame)
    common, initialization = pgo.first_loop_initialization(local, loops)
    input_load_s = time.perf_counter() - input_start
    if len(local) != EXPECTED_NODES or len(loops) != EXPECTED_LOOPS:
        raise AssertionError("benchmark inputs do not match the frozen graph")

    if args.mode == "odometry_only":
        solved, backend = compose_odometry(local_frame, local)
    elif args.mode == "batch_lm":
        solved, backend = run_batch(local_frame, local, loops, common)
    else:
        solved, backend = run_isam2(local_frame, local, loops, common)

    serialization_start = time.perf_counter()
    trajectory = pgo.trajectory_frame(local_frame, solved)
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(args.trajectory_output, index=False)
    serialization_s = time.perf_counter() - serialization_start
    baseline = pgo.matrices_from_frame(pd.read_csv(args.batch_baseline, usecols=pgo.POSE_COLUMNS))
    comparison = pose_difference(baseline, solved) if args.mode != "odometry_only" else None

    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "repetition": args.repetition,
        "warmup": bool(args.warmup),
        "source_commit": args.source_commit,
        "code_sha256": {
            "benchmark_pgo_backends.py": sha256(Path(__file__).resolve()),
            "run_pgo_gravity_validation.py": sha256(Path(pgo.__file__).resolve()),
        },
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "gtsam_version": getattr(gtsam, "__version__", "4.2 wheel"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "inputs": {
            "local_trajectory_sha256": sha256(args.local_trajectory),
            "loop_edges_sha256": sha256(args.loop_edges),
            "batch_baseline_sha256": sha256(args.batch_baseline),
        },
        "input_load_s": input_load_s,
        "backend": backend,
        "serialization_s": serialization_s,
        "process_wall_s": time.perf_counter() - process_start,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "trajectory_sha256": sha256(args.trajectory_output),
        "trajectory_rows": len(trajectory),
        "batch_pose_difference": comparison,
        "first_loop_initialization": initialization,
        "GT_used": False,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.summary_output, summary)
    print(json.dumps({
        "mode": args.mode,
        "repetition": args.repetition,
        "warmup": args.warmup,
        "backend_wall_s": backend["backend_wall_s"],
        "peak_rss_kib": summary["peak_rss_kib"],
    }), flush=True)


if __name__ == "__main__":
    main()
