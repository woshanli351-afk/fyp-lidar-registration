#!/usr/bin/env python3
"""Aggregate GT-free backend benchmarks, then run a separated offline GT evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_pgo_gravity_offline as offline


BACKENDS = ("odometry_only", "batch_lm", "isam2_default")
EXPECTED_TEAM_ISAM2 = {
    "translation_median_m": 0.12443065866514713,
    "translation_p95_m": 0.5639854280275796,
    "rotation_median_deg": 0.055319187878979874,
    "rotation_p95_deg": 0.3015371230925412,
    "final_graph_error": 23.840417589292862,
}
AGREEMENT_GATES = {
    "translation_median_m": 0.10,
    "translation_p95_m": 0.50,
    "rotation_median_deg": 0.5,
    "rotation_p95_deg": 2.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("each backend requires at least three finite measured repetitions")
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "empirical_p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def load_runs(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    runs = {backend: [] for backend in BACKENDS}
    for path in paths:
        value = json.loads(path.read_text())
        if value.get("GT_used") is not False:
            raise ValueError(f"benchmark is not GT-free: {path}")
        if value["mode"] not in runs:
            raise ValueError(f"unexpected backend: {value['mode']}")
        if not value.get("warmup"):
            value["_summary_sha256"] = sha256(path)
            runs[value["mode"]].append(value)
    if any(len(rows) < 3 for rows in runs.values()):
        raise ValueError({backend: len(rows) for backend, rows in runs.items()})
    for backend, rows in runs.items():
        rows.sort(key=lambda row: int(row["repetition"]))
        hashes = {tuple(sorted(row["inputs"].items())) for row in rows}
        if len(hashes) != 1:
            raise ValueError(f"input hashes changed across {backend} repetitions")
    all_hashes = {
        tuple(sorted(rows[0]["inputs"].items()))
        for rows in runs.values()
    }
    if len(all_hashes) != 1:
        raise ValueError("backends did not use identical frozen inputs")
    hosts = {row["host"] for rows in runs.values() for row in rows}
    thread_settings = {json.dumps(row["thread_environment"], sort_keys=True) for rows in runs.values() for row in rows}
    if len(hosts) != 1 or len(thread_settings) != 1:
        raise ValueError("benchmark host or thread settings changed")
    return runs


def performance_rows(runs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for backend in BACKENDS:
        group = runs[backend]
        wall = distribution([float(row["backend"]["backend_wall_s"]) for row in group])
        process = distribution([float(row["process_wall_s"]) for row in group])
        rss = distribution([float(row["peak_rss_kib"]) / 1024.0 for row in group])
        rows.append({
            "backend": backend,
            "measured_repetitions": len(group),
            "backend_wall_median_s": wall["median"],
            "backend_wall_empirical_p95_s": wall["empirical_p95"],
            "backend_wall_min_s": wall["min"],
            "backend_wall_max_s": wall["max"],
            "process_wall_median_s": process["median"],
            "peak_rss_median_mib": rss["median"],
            "peak_rss_empirical_p95_mib": rss["empirical_p95"],
            "peak_rss_max_mib": rss["max"],
            "p95_warning": "3 measured repetitions; empirical p95 is descriptive, not a stable tail estimate",
        })
    return rows


def trajectory_metrics(
    label: str,
    trajectory_path: Path,
    gt_by_robot: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    trajectory = offline.load_trajectory(trajectory_path)
    rows: list[dict[str, Any]] = []
    for row in offline.ate_rows(trajectory, gt_by_robot):
        if label == "odometry_only" and row["metric_type"] == "ATE_joint_rigid_alignment":
            continue
        rows.append({"backend": label, "use": "OFFLINE GT ONLY", **row})
    for row in offline.rpe_rows(trajectory, gt_by_robot, interval=10):
        rows.append({"backend": label, "use": "OFFLINE GT ONLY", **row})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=Path, action="append", required=True)
    parser.add_argument("--odometry-trajectory", type=Path, required=True)
    parser.add_argument("--batch-trajectory", type=Path, required=True)
    parser.add_argument("--isam2-trajectory", type=Path, required=True)
    parser.add_argument("--robot1-keyframes", type=Path, required=True)
    parser.add_argument("--robot3-keyframes", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.benchmark_json)
    trajectories = {
        "odometry_only": args.odometry_trajectory,
        "batch_lm": args.batch_trajectory,
        "isam2_default": args.isam2_trajectory,
    }
    for backend, path in trajectories.items():
        expected = {row["trajectory_sha256"] for row in runs[backend]}
        if sha256(path) not in expected:
            raise ValueError(f"{backend} trajectory is not one of the frozen measured outputs")

    isam = runs["isam2_default"][0]
    difference = isam["batch_pose_difference"]
    observed = {
        "translation_median_m": float(difference["translation_m"]["median"]),
        "translation_p95_m": float(difference["translation_m"]["p95"]),
        "rotation_median_deg": float(difference["rotation_deg"]["median"]),
        "rotation_p95_deg": float(difference["rotation_deg"]["p95"]),
        "final_graph_error": float(isam["backend"]["final_graph_error"]),
    }
    reproduction = {
        field: bool(np.isclose(observed[field], expected, rtol=0.0, atol=1e-6))
        for field, expected in EXPECTED_TEAM_ISAM2.items()
    }
    if not all(reproduction.values()):
        raise ValueError(f"new iSAM2 replay does not reproduce frozen Stage13B: {observed}")
    gate_checks = {
        field: observed[field] <= limit
        for field, limit in AGREEMENT_GATES.items()
    }
    pre_gt_decision = {
        "decision": "KEEP_BATCH_FOR_OFFLINE_DEMO",
        "reason": "odometry-only has no cross-robot frame; default iSAM2 reproduces the frozen run but fails same-gauge Batch agreement gates",
        "isam2_observed": observed,
        "isam2_frozen_reference_reproduced": reproduction,
        "agreement_gates": AGREEMENT_GATES,
        "agreement_checks": gate_checks,
        "GT_used": False,
    }
    write_json(output / "pre_gt_decision.json", pre_gt_decision)
    manifest = {
        "source_repository": "James-5757/fyp-place-recognition",
        "source_commit": args.source_commit,
        "benchmark_summary_sha256": {
            backend: [row["_summary_sha256"] for row in rows]
            for backend, rows in runs.items()
        },
        "trajectory_sha256": {backend: sha256(path) for backend, path in trajectories.items()},
        "GT_access_started_after_pre_gt_decision": True,
    }
    write_json(output / "frozen_manifest.json", manifest)

    performance = pd.DataFrame(performance_rows(runs))
    performance.to_csv(output / "performance_summary.csv", index=False, float_format="%.12g")

    # GT is first loaded only after the backend choice and hashes are persisted.
    required = {
        robot: offline.load_trajectory(path).query("robot_id == @robot").keyframe_id.astype(int).tolist()
        for robot in offline.ROBOTS
        for path in [trajectories["batch_lm"]]
    }
    gt_by_robot = {
        "robot1": offline.load_gt_keyframes(args.robot1_keyframes, required["robot1"]),
        "robot3": offline.load_gt_keyframes(args.robot3_keyframes, required["robot3"]),
    }
    accuracy_rows: list[dict[str, Any]] = []
    for backend, path in trajectories.items():
        accuracy_rows.extend(trajectory_metrics(backend, path, gt_by_robot))
    accuracy = pd.DataFrame(accuracy_rows)
    accuracy.to_csv(output / "offline_accuracy_summary.csv", index=False, float_format="%.12g")

    numerical = pd.DataFrame([
        {
            "comparison": "isam2_default_vs_batch_lm",
            **observed,
            "translation_median_gate_m": AGREEMENT_GATES["translation_median_m"],
            "translation_p95_gate_m": AGREEMENT_GATES["translation_p95_m"],
            "rotation_median_gate_deg": AGREEMENT_GATES["rotation_median_deg"],
            "rotation_p95_gate_deg": AGREEMENT_GATES["rotation_p95_deg"],
            "PASS": all(gate_checks.values()),
            "same_gauge_no_alignment": True,
        }
    ])
    numerical.to_csv(output / "numerical_agreement_summary.csv", index=False, float_format="%.12g")

    summary = {
        "status": "PASS",
        "decision": pre_gt_decision["decision"],
        "graph": {"nodes": 5796, "odometry_factors": 5794, "loop_factors": 75, "prior_factors": 1},
        "backend_roles": {
            "odometry_only": "per-robot local baseline; no cross-robot common frame or merged-map claim",
            "batch_lm": "converged offline reference and current Demo backend",
            "isam2_default": "fast incremental diagnostic; not accepted because same-graph pose agreement gates fail",
        },
        "performance": performance.set_index("backend").to_dict(orient="index"),
        "isam2_vs_batch": numerical.iloc[0].to_dict(),
        "GT_used_for_backend_or_selection": False,
        "offline_GT_metrics_written": True,
        "limitations": [
            "three measured repetitions per backend; p95 is descriptive",
            "backend-only benchmark excludes SC/GICP, networking, scan loading and map publication",
            "odometry-only trajectories are separate robot-local frames, so joint ATE and merged-map consistency are not applicable",
            "Robot3 is a preloaded reference map while Robot1 is replayed; this is not simultaneous live multi-robot operation",
        ],
    }
    write_json(output / "comparison_summary.json", summary)
    print(json.dumps({"status": "PASS", "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
