#!/usr/bin/env python3
"""Attach the frozen Gravity diagnostic and map sample to the replay demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "cumulti_pgo_gravity_validation"
DEFAULT_DEMO = REPO / "demo" / "progressive_replay_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_diagnostic(map_export: dict) -> tuple[dict, list[Path]]:
    status_path = RESULTS / "soft_covariance_validation_status.json"
    selected_path = RESULTS / "soft_covariance_selected_summary.json"
    tail_path = RESULTS / "map_tail_diagnostic_summary.json"
    pre_gt_path = RESULTS / "pre_gt_summary.json"
    sources = [status_path, selected_path, tail_path, pre_gt_path]
    status, selected, tail, pre_gt = map(read_json, sources)

    if status["graph"]["selected_trajectory_sha256"] != map_export["source_sha256"]["soft_trajectory"]:
        raise ValueError("Gravity map does not use the frozen Soft x8 trajectory")
    if tail["input_sha256"]["free_trajectory"] != map_export["source_sha256"]["free_trajectory"]:
        raise ValueError("Gravity map does not use the frozen Free trajectory")
    if tail["corrected_decision"] != "SOFT_GRAVITY_REMAINS_OFFLINE_ONLY":
        raise ValueError("Unexpected corrected Gravity decision")
    if status["phase_order"]["pre_gt_selection_used_GT"]:
        raise ValueError("Soft Gravity selection must remain GT-free")

    free_runtime = float(pre_gt["metrics"]["free_pose3_replay"]["runtime_s_representative"])
    soft_runtime = float(selected["solver"]["runtime_s"])
    directional = tail["directional_decomposition"]["robot1_to_robot3"]
    loop_local = tail["accepted_loop_local_map"]
    return {
        "status": tail["corrected_decision"],
        "selected_variant": "ekf_gravity_soft_cov_x8",
        "display_label": "Soft Gravity ×8",
        "claim_boundary": (
            "Offline diagnostic only. The final replay remains the frozen K75 Batch PGO map; "
            "Gravity is not used by the final backend."
        ),
        "graph": {
            "nodes": int(status["graph"]["nodes"]),
            "odometry_factors": int(selected["factor_counts"]["odometry_factors"]),
            "loop_factors": int(status["graph"]["loop_factors"]),
            "gravity_factors": int(selected["factor_counts"]["gravity_factors"]),
        },
        "selection": {
            "used_gt": False,
            "rule": status["pre_gt_result"]["selection_rule"],
            "free_common_up_p95_deg": float(status["pre_gt_result"]["free_common_up_p95_deg"]),
            "soft_common_up_p95_deg": float(status["pre_gt_result"]["soft_common_up_p95_deg"]),
        },
        "offline_evaluation": {
            "used_gt": True,
            "free_joint_ate_rmse_m": float(status["offline_result"]["free"]["joint_ATE_RMSE_m"]),
            "soft_joint_ate_rmse_m": float(status["offline_result"]["soft_x8"]["joint_ATE_RMSE_m"]),
            "joint_ate_relative_change_percent": float(
                status["offline_result"]["soft_x8_relative_to_free_percent"]["joint_ATE_RMSE"]
            ),
        },
        "map_quality": {
            "better_covered_direction": "robot1_to_robot3",
            "free_directional_p95_m": float(directional["free"]["p95_m"]),
            "soft_directional_p95_m": float(directional["soft_x8"]["p95_m"]),
            "directional_p95_relative_change_percent": float(
                directional["soft_relative_change_percent"]["p95"]
            ),
            "free_loop_local_p95_m": float(loop_local["free"]["pooled_p95_m"]),
            "soft_loop_local_p95_m": float(loop_local["soft_x8"]["pooled_p95_m"]),
            "loop_local_p95_delta_m": float(loop_local["soft_p95_delta_m"]),
            "loop_local_p95_relative_change_percent": float(
                loop_local["soft_p95_relative_change_percent"]
            ),
            "loops_with_lower_p95": int(loop_local["loops_with_lower_p95"]),
            "loops_with_higher_p95": int(loop_local["loops_with_higher_p95"]),
            "whole_map_tail_caveat": tail["diagnosis"],
        },
        "runtime": {
            "free_s": free_runtime,
            "soft_s": soft_runtime,
            "soft_over_free_ratio": soft_runtime / free_runtime,
            "implementation_note": (
                "Measured with the current Python CustomFactor diagnostic implementation; "
                "this is not a claim that every Gravity implementation has the same overhead."
            ),
        },
        "map_export": {
            "file": "data/gravity_map.json",
            "schema": map_export["schema"],
            "mode": map_export["mode"],
            "configuration": map_export["configuration"],
            "voxelized_point_counts": map_export["voxelized_point_counts"],
            "sha256": None,
        },
        "decision": {
            "map_improved": False,
            "final_backend_changed": False,
            "summary": (
                "Soft Gravity corrects the tilt failure mode and lowers trajectory ATE, "
                "but it does not improve the frozen accepted-loop local map gate and is too "
                "slow in the current diagnostic implementation."
            ),
        },
    }, sources


def validate_map_export(value: dict) -> None:
    if value.get("schema") != "FYP_GRAVITY_DEMO_MAP_V2":
        raise ValueError("Unexpected Gravity map schema")
    if value.get("mode") != "GT_FREE_PRESENTATION_EXPORT":
        raise ValueError("Gravity map must be a GT-free presentation export")
    if set(value.get("variants", {})) != {"free", "soft_x8"}:
        raise ValueError("Gravity map must contain Free and Soft x8 variants")
    for variant in ("free", "soft_x8"):
        robots = value["variants"][variant]
        if set(robots) != {"robot1", "robot3"} or any(not robots[r] for r in robots):
            raise ValueError(f"Incomplete Gravity map variant: {variant}")
        for robot, points in robots.items():
            if any(len(point) != 4 for point in points):
                raise ValueError(f"Point row must be XYZ plus keyframe ID in {variant}/{robot}")
    if set(value.get("keyframe_ranges", {})) != {"robot1", "robot3"}:
        raise ValueError("Gravity map must publish Robot1/Robot3 keyframe ranges")


def run(demo_root: Path, map_path: Path) -> None:
    target_bundle = demo_root / "data" / "demo.json"
    if not target_bundle.is_file():
        raise FileNotFoundError(target_bundle)
    map_export = read_json(map_path)
    validate_map_export(map_export)
    diagnostic, aggregate_sources = build_diagnostic(map_export)

    target_map = demo_root / "data" / "gravity_map.json"
    target_map.parent.mkdir(parents=True, exist_ok=True)
    if map_path.resolve() != target_map.resolve():
        shutil.copyfile(map_path, target_map)
    diagnostic["map_export"]["sha256"] = sha256(target_map)

    bundle = read_json(target_bundle)
    bundle["gravity_diagnostic"] = diagnostic
    bundle["metadata"]["gravity_diagnostic"] = (
        "Soft Gravity x8 is shown only as an offline diagnostic and does not replace K75."
    )
    provenance = [
        item for item in bundle.get("provenance", [])
        if not str(item.get("file", "")).startswith("gravity_diagnostic/")
    ]
    provenance.extend(
        {"file": f"gravity_diagnostic/{path.name}", "sha256": sha256(path)}
        for path in aggregate_sources
    )
    provenance.append({
        "file": "gravity_diagnostic/gravity_demo_map.json",
        "sha256": sha256(target_map),
    })
    bundle["provenance"] = provenance
    target_bundle.write_text(
        json.dumps(bundle, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    print(f"Updated {target_bundle}")
    print(f"Copied {target_map} ({target_map.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-export", type=Path, required=True)
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO)
    args = parser.parse_args()
    run(args.demo_root.resolve(), args.map_export.resolve())


if __name__ == "__main__":
    main()
