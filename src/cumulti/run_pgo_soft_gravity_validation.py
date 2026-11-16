#!/usr/bin/env python3
"""Freeze a covariance-supported soft-gravity candidate without reading GT.

The experiment reuses the frozen Stage 12D graph and per-keyframe roll/pitch
covariance audited from raw EKF Odometry. A finite, predeclared inflation grid
is evaluated using GT-free diagnostics only. The weakest passing factor is
frozen for later, isolated offline evaluation.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import gtsam
import numpy as np
import pandas as pd

from run_pgo_gravity_validation import (
    ROBOTS,
    Variant,
    build_graph,
    compare_to_baseline,
    continuity_diagnostics,
    first_loop_initialization,
    gravity_diagnostics,
    load_inputs,
    loop_residuals,
    matrices_from_frame,
    sha256,
    solve,
    trajectory_frame,
    utc_now,
    write_json,
)


DEFAULT_SCALES = (1.0, 2.0, 4.0, 8.0)
COVARIANCE_COLUMNS = [
    "robot_id",
    "keyframe_id",
    "decoded_from_raw",
    "roll_variance_rad2",
    "pitch_variance_rad2",
]


def numeric_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("covariance statistics require finite non-empty values")
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def validate_scales(scales: list[float]) -> None:
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("all covariance inflation scales must be positive")
    if list(scales) != sorted(set(scales)):
        raise ValueError("covariance inflation scales must be unique and increasing")


def load_covariance_sigmas(
    paths: list[Path],
    local_frame: pd.DataFrame,
) -> tuple[dict[tuple[str, int], float], dict[str, object]]:
    frames = []
    inputs = []
    for path in paths:
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = set(COVARIANCE_COLUMNS) - header
        if missing:
            raise ValueError(f"covariance CSV missing columns: {sorted(missing)}")
        # GT/offline label columns, if present, are never loaded.
        frame = pd.read_csv(path, usecols=COVARIANCE_COLUMNS)
        frames.append(frame)
        inputs.append({
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    covariance = pd.concat(frames, ignore_index=True)
    if covariance.duplicated(["robot_id", "keyframe_id"]).any():
        raise ValueError("duplicate covariance node IDs")
    graph_nodes = set(zip(local_frame.robot_id, local_frame.keyframe_id.astype(int)))
    covariance_nodes = set(zip(covariance.robot_id, covariance.keyframe_id.astype(int)))
    if covariance_nodes != graph_nodes:
        raise ValueError(
            f"covariance node set differs from graph: "
            f"missing={len(graph_nodes - covariance_nodes)}, extra={len(covariance_nodes - graph_nodes)}"
        )
    decoded = covariance.decoded_from_raw.astype("boolean").fillna(False)
    valid = covariance.loc[decoded].copy()
    invalid_numeric = ~np.isfinite(
        valid[["roll_variance_rad2", "pitch_variance_rad2"]].to_numpy(float)
    ).all(axis=1)
    if invalid_numeric.any():
        raise ValueError("decoded covariance rows contain non-finite tilt variance")
    if (valid[["roll_variance_rad2", "pitch_variance_rad2"]] <= 0).any(axis=None):
        raise ValueError("decoded covariance rows contain non-positive tilt variance")
    valid["base_sigma_deg"] = np.degrees(np.sqrt(
        (valid.roll_variance_rad2 + valid.pitch_variance_rad2) / 2.0
    ))
    sigmas = {
        (str(row.robot_id), int(row.keyframe_id)): float(row.base_sigma_deg)
        for row in valid.itertuples()
    }
    missing_nodes = sorted(graph_nodes - set(sigmas))
    summary: dict[str, object] = {
        "inputs": inputs,
        "graph_nodes": len(graph_nodes),
        "decoded_covariance_nodes": len(sigmas),
        "coverage_fraction": len(sigmas) / len(graph_nodes),
        "missing_nodes": [list(node) for node in missing_nodes],
        "missing_covariance_policy": "omit gravity factor for unmatched node",
        "base_sigma_deg": numeric_stats(valid.base_sigma_deg.to_numpy(float)),
        "per_robot": {},
        "GT_used": False,
    }
    for robot in ROBOTS:
        subset = valid[valid.robot_id.eq(robot)].base_sigma_deg.to_numpy(float)
        summary["per_robot"][robot] = numeric_stats(subset)
    return sigmas, summary


def frozen_protocol(args: argparse.Namespace, covariance_summary: dict[str, object]) -> dict[str, object]:
    graph_paths = [args.local_trajectory, args.loop_edges, args.baseline_trajectory]
    return {
        "created_UTC": utc_now(),
        "experiment": "Stage12D covariance-supported soft gravity pre-GT selection",
        "graph_inputs": [
            {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in graph_paths
        ],
        "covariance_inputs": covariance_summary["inputs"],
        "base_sigma": "sqrt((roll_variance_rad2 + pitch_variance_rad2) / 2), converted to degrees",
        "inflation_scales": list(args.scales),
        "selection_rule": (
            "Among variants meeting every GT-free criterion, freeze the largest inflation scale "
            "(weakest gravity factor). Do not use GT to choose a scale."
        ),
        "missing_covariance_policy": "omit gravity factor for unmatched node",
        "fixed_noise": {
            "odometry_and_loop_rotation_sigma_deg": 5.0,
            "odometry_and_loop_translation_sigma_m": 1.0,
            "loop_huber_delta": 1.345,
        },
        "pre_GT_criteria": {
            "gravity_common_up_p95_max_deg": 2.0,
            "maximum_loop_translation_p95_increase_fraction": 0.20,
            "maximum_new_adjacent_steps_over_5m": 0,
        },
        "GT_used": False,
    }


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.local_trajectory = args.local_trajectory.resolve()
    args.loop_edges = args.loop_edges.resolve()
    args.baseline_trajectory = args.baseline_trajectory.resolve()
    args.covariance_csv = [path.resolve() for path in args.covariance_csv]

    local_frame, loops = load_inputs(args.local_trajectory, args.loop_edges)
    local = matrices_from_frame(local_frame)
    common_initial, initialization = first_loop_initialization(local, loops)
    write_json(output / "first_loop_initialization.json", initialization)
    base_sigmas, covariance_summary = load_covariance_sigmas(
        args.covariance_csv, local_frame
    )
    protocol = frozen_protocol(args, covariance_summary)
    write_json(output / "covariance_input_summary.json", covariance_summary)
    write_json(output / "frozen_protocol.json", protocol)

    variants = [Variant("free_pose3_replay", False)] + [
        Variant(f"ekf_gravity_soft_cov_x{scale:g}", True) for scale in args.scales
    ]
    summaries: dict[str, dict[str, object]] = {}
    for variant in variants:
        scale = None if not variant.preserve_ekf_gravity else float(
            variant.name.rsplit("x", 1)[1]
        )
        scaled = None if scale is None else {
            node: sigma * scale for node, sigma in base_sigmas.items()
        }
        print(json.dumps({
            "stage": "SOLVING", "variant": variant.name, "UTC": utc_now()
        }), flush=True)
        graph, counts = build_graph(
            local_frame,
            local,
            loops,
            common_initial,
            variant,
            gravity_sigma_deg=0.01,
            gravity_sigma_by_node_deg=scaled,
        )
        solved, solver = solve(graph, common_initial)
        trajectory_path = output / f"{variant.name}_trajectory.csv"
        trajectory_frame(local_frame, solved).to_csv(trajectory_path, index=False)
        residual_frame, residual_summary = loop_residuals(loops, solved)
        residual_frame.to_csv(output / f"{variant.name}_loop_residuals.csv", index=False)
        gravity_frame, gravity_summary = gravity_diagnostics(local, common_initial, solved)
        gravity_frame.to_csv(output / f"{variant.name}_gravity.csv", index=False)
        continuity_frame, continuity_summary = continuity_diagnostics(
            local_frame, common_initial, solved
        )
        continuity_frame.to_csv(output / f"{variant.name}_continuity.csv", index=False)
        summaries[variant.name] = {
            "inflation_scale": scale,
            "factor_counts": counts,
            "solver": solver,
            "loop_residuals": residual_summary,
            "gravity": gravity_summary,
            "continuity": continuity_summary,
            "trajectory_sha256": sha256(trajectory_path),
            "GT_used": False,
        }
        if not variant.preserve_ekf_gravity:
            summaries[variant.name]["upstream_reproduction"] = compare_to_baseline(
                args.baseline_trajectory, solved
            )
        write_json(output / f"{variant.name}_summary.json", summaries[variant.name])

    free = summaries["free_pose3_replay"]
    free_loop_p95 = float(free["loop_residuals"]["translation_m"]["p95"])
    criteria = protocol["pre_GT_criteria"]
    checks: dict[str, dict[str, bool]] = {}
    passing = []
    for scale in args.scales:
        name = f"ekf_gravity_soft_cov_x{scale:g}"
        summary = summaries[name]
        variant_checks = {
            "gravity_p95_at_or_below_target": (
                float(summary["gravity"]["common_up_disagreement_deg"]["p95"])
                <= criteria["gravity_common_up_p95_max_deg"]
            ),
            "loop_translation_p95_within_increase_limit": (
                float(summary["loop_residuals"]["translation_m"]["p95"])
                <= free_loop_p95 * (
                    1.0 + criteria["maximum_loop_translation_p95_increase_fraction"]
                )
            ),
            "no_new_adjacent_steps_over_5m": (
                summary["continuity"]["new_steps_over_5m"]
                <= criteria["maximum_new_adjacent_steps_over_5m"]
            ),
        }
        checks[name] = variant_checks
        if all(variant_checks.values()):
            passing.append((float(scale), name))
    selected = max(passing)[1] if passing else None
    decision = {
        "created_UTC": utc_now(),
        "decision": (
            "SOFT_COVARIANCE_GT_FREE_CANDIDATE_FROZEN"
            if selected else "NO_SOFT_COVARIANCE_VARIANT_MET_GT_FREE_GATES"
        ),
        "selected_variant": selected,
        "selected_trajectory_sha256": (
            summaries[selected]["trajectory_sha256"] if selected else None
        ),
        "selection_rule": protocol["selection_rule"],
        "checks": checks,
        "comparison": {
            name: {
                "inflation_scale": summary["inflation_scale"],
                "common_up_p95_deg": summary["gravity"]["common_up_disagreement_deg"]["p95"],
                "loop_translation_p95_m": summary["loop_residuals"]["translation_m"]["p95"],
                "new_steps_over_5m": summary["continuity"]["new_steps_over_5m"],
                "runtime_s": summary["solver"]["runtime_s"],
            }
            for name, summary in summaries.items()
        },
        "offline_evaluation_allowed_only_for_selected_variant": bool(selected),
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
    for item in protocol["graph_inputs"] + protocol["covariance_inputs"]:
        candidates = [
            path for path in [*args.covariance_csv, args.local_trajectory,
                              args.loop_edges, args.baseline_trajectory]
            if path.name == item["filename"]
        ]
        if len(candidates) != 1 or sha256(candidates[0]) != item["sha256"]:
            raise AssertionError(f"frozen input changed during solve: {item['filename']}")
    print(json.dumps(decision, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--local-trajectory", type=Path, required=True)
    command.add_argument("--loop-edges", type=Path, required=True)
    command.add_argument("--baseline-trajectory", type=Path, required=True)
    command.add_argument("--covariance-csv", type=Path, action="append", required=True)
    command.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    command.add_argument("--output-dir", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    validate_scales(args.scales)
    if len(args.covariance_csv) != len(ROBOTS):
        raise ValueError("provide one covariance CSV for each frozen robot")
    run(args)


if __name__ == "__main__":
    main()
