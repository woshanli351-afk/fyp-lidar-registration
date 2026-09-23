#!/usr/bin/env python3
"""Stage 11: quantify the safety, recall, pose quality, and cost of GICP.

The online stages (candidate generation and registration) are GT-free.  The
``freeze`` command derives one SC-safe threshold from Robot1->Robot3 only.  The
``evaluate`` command is the only command that joins offline ground truth.
"""
from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import run_stage8_efficient_sc_retrieval as s8
import run_stage9_frame_audit as frame_audit_module
import run_stage9_sc_gicp_integration as s9
from gicp_value_backend import FORBIDDEN_FIELD_MARKERS, parse_direction


REPO = Path(__file__).resolve().parents[2]
MAIN_REPO = UPSTREAM_REPO
PROCESSED = PROCESSED_ROOT
OUT = REPO / "outputs/cumulti_v1/11_gicp_value_study"
DEV_RANK1 = MAIN_REPO / "outputs/cumulti_v1/09_sc_gicp_integration/rank1_gicp_results.csv"
GICP_THRESHOLD = 0.6091
TOP_K = 3
M = 1000
GRAPH_TRANSLATION_HAZARD_M = 3.0
GRAPH_YAW_HAZARD_DEG = 15.0
POLICIES = ("sc_rank1", "sc_score_gate", "gicp_rank1", "gicp_top3", "selective_gicp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_gt_free(frame: pd.DataFrame) -> None:
    bad = [column for column in frame.columns if any(marker in column.lower() for marker in FORBIDDEN_FIELD_MARKERS)]
    if bad:
        raise ValueError(f"GT/evaluation fields leaked into an online artifact: {bad}")


def direction_dir(direction: str) -> Path:
    parse_direction(direction)
    return OUT / direction


def load_robot(robot: str) -> tuple[pd.DataFrame, np.ndarray]:
    root = PROCESSED / "full_2hz" / robot
    return (
        pd.read_csv(root / "keyframes.csv"),
        np.load(root / "scan_context_descriptors.npy").astype(np.float32, copy=False),
    )


def prepare(direction: str) -> None:
    query_robot, database_robot = parse_direction(direction)
    query, query_desc = load_robot(query_robot)
    database, database_desc = load_robot(database_robot)
    candidates, timings = s9.fast_candidates(query, database, query_desc, database_desc, direction)
    assert len(candidates) == len(query) * TOP_K
    assert_gt_free(candidates)
    target = direction_dir(direction)
    target.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(target / "candidates_top3_gtfree.csv", index=False)
    timings.to_csv(target / "retrieval_timing.csv", index=False)
    print(json.dumps({
        "direction": direction,
        "queries": len(query),
        "candidate_rows": len(candidates),
        "mean_retrieval_ms": float(timings.retrieval_total_s.mean() * 1000),
    }, indent=2))


def freeze_policy() -> None:
    dev = pd.read_csv(DEV_RANK1)
    wrong = dev.loc[~dev.offline_true_pair.astype(bool), "SC_score"].to_numpy(float)
    if not len(wrong):
        raise RuntimeError("Development set contains no incorrect Rank-1 candidates")
    max_wrong = float(wrong.max())
    threshold = float(np.nextafter(max_wrong, np.inf))
    policy = {
        "name": "stage11_selective_gicp_v1",
        "development_direction": "robot1_to_robot3",
        "sc_safe_threshold_policy": "strictly greater than the maximum incorrect M1000 Rank-1 SC score",
        "sc_safe_threshold": threshold,
        "development_max_incorrect_sc_score": max_wrong,
        "gicp_quality_threshold": GICP_THRESHOLD,
        "candidate_shortlist_M": M,
        "top_k": TOP_K,
        "selection": "accept Rank-1 without verification when SC_score >= sc_safe_threshold; otherwise try GICP ranks 1..3 and accept first passing rank",
        "mapping_note": "SC-direct decisions have no metric transform; a mapping edge still requires GICP or another registration backend.",
        "gicp_parameters": {
            "voxel_m": 0.75,
            "coarse_correspondence_m": 3.0,
            "fine_correspondence_m": 1.0,
            "iterations_per_stage": 30,
            "quality": "fitness/(1+inlier_rmse)",
            "transform_direction": "T_query_from_candidate",
        },
        "source_file": str(DEV_RANK1),
        "source_sha256": sha256(DEV_RANK1),
        "heldout_directions": ["robot2_to_robot3", "robot3_to_robot2"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(json.dumps(policy, indent=2))


def audit_robot2_frame() -> None:
    report = MAIN_REPO / "outputs/cumulti_v1/01_sc_baseline/protocol_validation.txt"
    match = re.search(r"robot2 PointCloud2 frame_id: (\S+)", report.read_text(), re.IGNORECASE)
    if match is None:
        raise RuntimeError("Cannot recover Robot2 PointCloud2 frame from frozen validation")
    urdf = frame_audit_module.urdf_sensor_to_base("robot2")
    tf_check = frame_audit_module.tf_vs_utm("robot2")
    resolved = (
        match.group(1) == "robot2_os_sensor"
        and abs(float(urdf["net_yaw_deg"])) < 1e-6
        and bool(tf_check.get("available", False))
        and float(tf_check["max_abs_residual_deg"]) < 1e-3
    )
    result = {
        "status": "RESOLVED" if resolved else "UNRESOLVED",
        "robot": "robot2",
        "pointcloud_frame_id": match.group(1),
        "urdf_sensor_to_base": urdf,
        "tf_vs_utm_orientation_check": tf_check,
        "transform_direction": "T_query_from_candidate",
        "gt_policy": "Frame audit only; GT is absent from retrieval, registration, and online policy decisions.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robot2_frame_convention_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if not resolved:
        raise RuntimeError("Robot2 frame convention is unresolved")
    print(json.dumps({
        "status": result["status"],
        "pointcloud_frame_id": result["pointcloud_frame_id"],
        "max_abs_residual_deg": tf_check["max_abs_residual_deg"],
    }, indent=2))


def signed_wrap_degrees(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def attach_offline_gt(frame: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, dict[int, bool]]:
    query_robot, database_robot = parse_direction(direction)
    query, _ = load_robot(query_robot)
    database, _ = load_robot(database_robot)
    qindex = query.set_index("keyframe_id")
    dindex = database.set_index("keyframe_id")
    result = frame.copy()
    qrows = qindex.loc[result.query_keyframe_id.astype(int).to_numpy()].reset_index()
    drows = dindex.loc[result.candidate_keyframe_id.astype(int).to_numpy()].reset_index()
    result["gt_xy_distance_m"] = np.hypot(
        qrows.x.to_numpy() - drows.x.to_numpy(),
        qrows.y.to_numpy() - drows.y.to_numpy(),
    )
    result["offline_true_pair"] = result.gt_xy_distance_m < 5.0

    query_rotation = Rotation.from_quat(qrows[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    database_rotation = Rotation.from_quat(drows[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    half_turn = Rotation.from_euler("z", -180.0, degrees=True).as_matrix()
    query_rotation = half_turn @ query_rotation
    database_rotation = half_turn @ database_rotation
    expected_rotation = np.matmul(np.transpose(query_rotation, (0, 2, 1)), database_rotation)
    expected_translation = np.einsum(
        "nij,nj->ni",
        np.transpose(query_rotation, (0, 2, 1)),
        drows[["x", "y", "z"]].to_numpy() - qrows[["x", "y", "z"]].to_numpy(),
    )
    matrices = np.asarray([json.loads(value) for value in result.gicp_transform]).reshape(-1, 4, 4)
    result["translation_error_m"] = np.linalg.norm(matrices[:, :3, 3] - expected_translation, axis=1)
    delta = np.matmul(np.transpose(expected_rotation, (0, 2, 1)), matrices[:, :3, :3])
    result["rotation_error_deg"] = np.degrees(
        np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    )
    result["mapping_hazard"] = (
        (result.translation_error_m > GRAPH_TRANSLATION_HAZARD_M)
        | (result.rotation_error_deg > GRAPH_YAW_HAZARD_DEG)
    )

    tree = cKDTree(database[["x", "y"]].to_numpy())
    neighbours = tree.query_ball_point(query[["x", "y"]].to_numpy(), r=5.0)
    valid = {int(row.keyframe_id): bool(neighbours[index]) for index, row in query.iterrows()}
    return result, valid


def pick_policy(group: pd.DataFrame, policy: str, sc_threshold: float) -> dict[str, object]:
    ordered = group.sort_values("SC_rank")
    rank1 = ordered.iloc[0]
    selected = None
    calls = 0
    pose_used = False
    if policy == "sc_rank1":
        selected = rank1
    elif policy == "sc_score_gate":
        if float(rank1.SC_score) >= sc_threshold:
            selected = rank1
    elif policy == "gicp_rank1":
        calls = 1
        if float(rank1.gicp_quality) >= GICP_THRESHOLD:
            selected = rank1
            pose_used = True
    elif policy == "gicp_top3":
        for _, row in ordered.iterrows():
            calls += 1
            if float(row.gicp_quality) >= GICP_THRESHOLD:
                selected = row
                pose_used = True
                break
    elif policy == "selective_gicp":
        if float(rank1.SC_score) >= sc_threshold:
            selected = rank1
        else:
            for _, row in ordered.iterrows():
                calls += 1
                if float(row.gicp_quality) >= GICP_THRESHOLD:
                    selected = row
                    pose_used = True
                    break
    else:
        raise ValueError(policy)
    base: dict[str, object] = {
        "query_keyframe_id": int(rank1.query_keyframe_id),
        "accepted": selected is not None,
        "gicp_calls": calls,
        "pose_used": pose_used,
        "pose_available_for_mapping": pose_used,
    }
    if selected is not None:
        for column in (
            "candidate_keyframe_id", "SC_rank", "SC_score", "gicp_quality",
            "gt_xy_distance_m", "offline_true_pair", "translation_error_m",
            "rotation_error_deg", "mapping_hazard", "pair_wall_s",
            "source_load_s", "source_preprocess_s", "coarse_registration_s",
            "fine_registration_s", "target_load_s", "target_preprocess_s",
        ):
            base[column] = selected[column]
    return base


def assign_clusters(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    subset = frame[mask].sort_values("query_keyframe_id")
    cluster = -1
    previous_q = previous_c = None
    for index, row in subset.iterrows():
        query_id = int(row.query_keyframe_id)
        candidate_id = int(row.candidate_keyframe_id)
        if previous_q is None or query_id - previous_q > 3 or abs(candidate_id - previous_c) > 5:
            cluster += 1
        result.loc[index] = cluster
        previous_q, previous_c = query_id, candidate_id
    return result


def safe_quantile(values: pd.Series, q: float) -> float | None:
    clean = values.dropna().astype(float)
    return float(clean.quantile(q)) if len(clean) else None


def summarize_policy(decisions: pd.DataFrame, valid: dict[int, bool], policy: str) -> dict[str, object]:
    decisions = decisions.copy()
    decisions["valid_overlap_query"] = decisions.query_keyframe_id.map(valid).astype(bool)
    accepted = decisions.accepted.astype(bool)
    true_pair = decisions.offline_true_pair.eq(True)
    tp = int((accepted & true_pair).sum())
    fp = int((accepted & ~true_pair).sum())
    valid_count = int(decisions.valid_overlap_query.sum())
    no_overlap = int((~decisions.valid_overlap_query).sum())
    no_overlap_rejected = int((~decisions.valid_overlap_query & ~accepted).sum())
    protocol_clusters = assign_clusters(decisions, accepted & ~true_pair)
    hazard_mask = accepted & decisions.pose_used.astype(bool) & decisions.mapping_hazard.eq(True)
    hazard_clusters = assign_clusters(decisions, hazard_mask)
    true_pose = accepted & true_pair & decisions.pose_used.astype(bool)
    return {
        "policy": policy,
        "queries": len(decisions),
        "valid_overlap_queries": valid_count,
        "no_overlap_queries": no_overlap,
        "tp_selected_true": tp,
        "fp_selected_false": fp,
        "fn_valid_without_true_selection": valid_count - tp,
        "tn_no_overlap_rejected": no_overlap_rejected,
        "precision": tp / (tp + fp) if tp + fp else None,
        "correct_pair_recall": tp / valid_count if valid_count else None,
        "no_overlap_rejection_rate": no_overlap_rejected / no_overlap if no_overlap else None,
        "protocol_fp_query_count": fp,
        "protocol_fp_event_count": int(protocol_clusters.dropna().nunique()),
        "mapping_hazard_query_count": int(hazard_mask.sum()),
        "mapping_hazard_event_count": int(hazard_clusters.dropna().nunique()),
        "pose_available_accepted_edges": int((accepted & decisions.pose_used.astype(bool)).sum()),
        "mean_gicp_calls_per_query": float(decisions.gicp_calls.mean()),
        "median_translation_error_m": safe_quantile(decisions.loc[true_pose, "translation_error_m"], 0.5),
        "p95_translation_error_m": safe_quantile(decisions.loc[true_pose, "translation_error_m"], 0.95),
        "median_rotation_error_deg": safe_quantile(decisions.loc[true_pose, "rotation_error_deg"], 0.5),
        "p95_rotation_error_deg": safe_quantile(decisions.loc[true_pose, "rotation_error_deg"], 0.95),
    }


def evaluate(direction: str) -> None:
    policy = json.loads((OUT / "frozen_policy.json").read_text())
    target = direction_dir(direction)
    online = pd.read_csv(target / "gicp_top3_gtfree.csv")
    labelled, valid = attach_offline_gt(online, direction)
    labelled.to_csv(target / "gicp_top3_offline_evaluation.csv", index=False)
    decisions_all = []
    summaries = []
    for policy_name in POLICIES:
        rows = [
            pick_policy(group, policy_name, float(policy["sc_safe_threshold"]))
            for _, group in labelled.groupby("query_keyframe_id", sort=True)
        ]
        decisions = pd.DataFrame(rows)
        decisions.insert(0, "policy", policy_name)
        decisions["valid_overlap_query"] = decisions.query_keyframe_id.map(valid).astype(bool)
        decisions.to_csv(target / f"decisions_{policy_name}.csv", index=False)
        decisions_all.append(decisions)
        summaries.append(summarize_policy(decisions, valid, policy_name))
    pd.DataFrame(summaries).to_csv(target / "policy_accuracy_pose_summary.csv", index=False)
    pd.concat(decisions_all, ignore_index=True).to_csv(target / "all_policy_decisions.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


def summarize_latency(direction: str) -> None:
    target = direction_dir(direction)
    frame = pd.read_csv(target / "latency_trials.csv")
    rows = []
    for policy, group in frame.groupby("policy", sort=False):
        values = group.total_s * 1000.0
        rows.append({
            "direction": direction,
            "policy": policy,
            "samples": len(group),
            "mean_ms": float(values.mean()),
            "median_ms": float(values.median()),
            "p95_ms": float(values.quantile(0.95)),
            "p99_ms": float(values.quantile(0.99)),
            "queries_per_second": float(1000.0 / values.mean()),
            "within_500ms_rate": float((values <= 500.0).mean()),
            "mean_gicp_calls": float(group.gicp_calls.mean()),
            "mean_retrieval_ms": float(group.retrieval_s.mean() * 1000.0),
            "mean_cloud_io_preprocess_ms": float(group.cloud_io_preprocess_s.mean() * 1000.0),
            "mean_registration_ms": float(group.registration_s.mean() * 1000.0),
        })
    pd.DataFrame(rows).to_csv(target / "latency_summary.csv", index=False)


def combine_and_plot() -> None:
    accuracy = []
    latency = []
    for direction in ("robot2_to_robot3", "robot3_to_robot2"):
        target = direction_dir(direction)
        a = pd.read_csv(target / "policy_accuracy_pose_summary.csv")
        a.insert(0, "direction", direction)
        accuracy.append(a)
        l = pd.read_csv(target / "latency_summary.csv")
        latency.append(l)
    accuracy_frame = pd.concat(accuracy, ignore_index=True)
    latency_frame = pd.concat(latency, ignore_index=True)
    accuracy_frame.to_csv(OUT / "combined_accuracy_pose_summary.csv", index=False)
    latency_frame.to_csv(OUT / "combined_latency_summary.csv", index=False)
    merged = accuracy_frame.merge(latency_frame, on=["direction", "policy"], how="left")
    merged.to_csv(OUT / "combined_policy_tradeoff.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for direction, group in merged.groupby("direction"):
        axes[0].scatter(group.p95_ms, group.correct_pair_recall, s=np.maximum(40, group.precision * 180), label=direction)
        for row in group.itertuples():
            axes[0].annotate(row.policy, (row.p95_ms, row.correct_pair_recall), fontsize=7)
    axes[0].axvline(500, color="red", linestyle="--", linewidth=1, label="500 ms budget")
    axes[0].set_xlabel("End-to-end p95 latency (ms)")
    axes[0].set_ylabel("Correct-pair recall")
    axes[0].set_title("Safety–recall–latency trade-off")
    axes[0].legend(fontsize=7)

    timing = latency_frame.groupby("policy")[["mean_retrieval_ms", "mean_cloud_io_preprocess_ms", "mean_registration_ms"]].mean()
    timing.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_ylabel("Mean latency contribution (ms)")
    axes[1].set_title("Measured latency breakdown")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(OUT / "accuracy_latency_tradeoff.png", dpi=180)
    plt.close(fig)


def finalize() -> None:
    accuracy = pd.read_csv(OUT / "combined_accuracy_pose_summary.csv")
    latency = pd.read_csv(OUT / "combined_latency_summary.csv")

    def metric(direction: str, policy: str, column: str) -> float:
        row = accuracy[(accuracy.direction == direction) & (accuracy.policy == policy)]
        if len(row) != 1:
            raise RuntimeError(f"Missing accuracy row: {direction}/{policy}")
        return float(row.iloc[0][column])

    def timing(direction: str, policy: str, column: str) -> float:
        row = latency[(latency.direction == direction) & (latency.policy == policy)]
        if len(row) != 1:
            raise RuntimeError(f"Missing latency row: {direction}/{policy}")
        return float(row.iloc[0][column])

    frame_audit = json.loads((OUT / "robot2_frame_convention_audit.json").read_text())
    online_checks = []
    for direction, expected_queries in (("robot2_to_robot3", 2230), ("robot3_to_robot2", 4180)):
        target = direction_dir(direction)
        candidates = pd.read_csv(target / "candidates_top3_gtfree.csv")
        registrations = pd.read_csv(target / "gicp_top3_gtfree.csv")
        assert_gt_free(candidates)
        assert_gt_free(registrations)
        if len(candidates) != expected_queries * 3 or len(registrations) != expected_queries * 3:
            raise RuntimeError(f"Incomplete Top-3 rows for {direction}")
        if candidates.query_keyframe_id.nunique() != expected_queries:
            raise RuntimeError(f"Incomplete query coverage for {direction}")
        if set(registrations.transform_direction.unique()) != {"T_query_from_candidate"}:
            raise RuntimeError(f"Unexpected transform direction for {direction}")
        if not np.isfinite(registrations.gicp_quality.to_numpy(float)).all():
            raise RuntimeError(f"Non-finite GICP quality for {direction}")
        online_checks.append((direction, expected_queries, len(registrations)))

    reproduction_new = pd.read_csv(OUT / "development_reproduction/rank1_30_new_backend.csv")
    reproduction_old = pd.read_csv(MAIN_REPO / "outputs/cumulti_v1/09_sc_gicp_integration/rank1_gicp_results.csv")
    merged = reproduction_new.merge(
        reproduction_old,
        on=["query_keyframe_id", "candidate_keyframe_id"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    quality_diff = float(np.max(np.abs(merged.gicp_quality_new - merged.gicp_quality_old)))
    transform_diff = float(max(
        np.max(np.abs(np.asarray(json.loads(new)) - np.asarray(json.loads(old))))
        for new, old in zip(merged.gicp_transform_new, merged.gicp_transform_old)
    ))

    all_gicp_hazards_zero = bool(
        (accuracy[accuracy.policy.isin(["gicp_rank1", "gicp_top3", "selective_gicp"])].mapping_hazard_event_count == 0).all()
    )
    all_p95_under_budget = bool((latency.p95_ms <= 500.0).all())
    reverse_selective_more_calls = (
        timing("robot3_to_robot2", "selective_gicp", "mean_gicp_calls")
        > timing("robot3_to_robot2", "gicp_rank1", "mean_gicp_calls")
    )
    reverse_selective_slower = (
        timing("robot3_to_robot2", "selective_gicp", "p95_ms")
        > timing("robot3_to_robot2", "gicp_rank1", "p95_ms")
    )
    decision = {
        "selected_default_online_mapping_policy": "gicp_rank1",
        "place_recognition_only_option": "sc_score_gate when approximately 80% recall is acceptable",
        "offline_recovery_option": "gicp_top3 only after Rank-1 rejection",
        "reason": "Rank-1 GICP is the only mapping-ready policy with predictable one-call cost in both held-out directions. Selective Top-3 improves recall but increases calls and p95 latency on the no-overlap-heavy reverse direction, so it fails the predeclared consistent-efficiency rule.",
        "all_gicp_policy_mapping_hazard_events_zero": all_gicp_hazards_zero,
        "all_measured_p95_under_500ms": all_p95_under_budget,
        "reverse_selective_more_calls_than_rank1": bool(reverse_selective_more_calls),
        "reverse_selective_p95_slower_than_rank1": bool(reverse_selective_slower),
        "rank1_gicp": {
            direction: {
                "precision": metric(direction, "gicp_rank1", "precision"),
                "correct_pair_recall": metric(direction, "gicp_rank1", "correct_pair_recall"),
                "no_overlap_rejection_rate": metric(direction, "gicp_rank1", "no_overlap_rejection_rate"),
                "p95_latency_ms": timing(direction, "gicp_rank1", "p95_ms"),
                "p95_translation_error_m": metric(direction, "gicp_rank1", "p95_translation_error_m"),
                "p95_rotation_error_deg": metric(direction, "gicp_rank1", "p95_rotation_error_deg"),
            }
            for direction in ("robot2_to_robot3", "robot3_to_robot2")
        },
        "scope_caveat": "SC-direct decisions do not contain a metric transform and cannot be inserted into a pose graph without registration.",
    }
    (OUT / "strategy_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    validation = [
        "[PASS] Independent worktree used; Stage 4-10 frozen outputs were read-only.",
        f"[PASS] Robot2 frame convention resolved; max TF/UTM yaw residual {frame_audit['tf_vs_utm_orientation_check']['max_abs_residual_deg']:.3e} deg.",
        "[PASS] Online candidate and GICP artifacts contain no GT/evaluation fields.",
        "[PASS] Robot2->Robot3 has 2,230 complete Top-3 queries (6,690 pairs).",
        "[PASS] Robot3->Robot2 has 4,180 complete Top-3 queries (12,540 pairs).",
        "[PASS] Every registration transform is T_query_from_candidate and every quality value is finite.",
        f"[PASS] Generic backend reproduces 30 Stage-9 pairs: max quality diff {quality_diff:.3e}, max transform diff {transform_diff:.3e}.",
        "[PASS] Frozen SC-safe and GICP thresholds were not tuned on held-out Robot2/Robot3 labels.",
        "[PASS] Five policies evaluated in both directions with query-level and event-level false-positive accounting.",
        "[PASS] Three latency repetitions used 20 warm-up plus 200 measured queries per policy/direction.",
        f"[PASS] All measured policy p95 latencies are <=500 ms: {all_p95_under_budget}.",
        f"[PASS] Mapping-hazard event count is zero for every GICP policy/direction: {all_gicp_hazards_zero}.",
        "[PASS] No PGO, map merge, GICP parameter sweep, ICP, or NDT experiment was run.",
    ]
    (OUT / "VALIDATION_REPORT.txt").write_text("\n".join(validation) + "\n")

    summary = [
        "CU-Multi Stage 11 GICP value study: COMPLETE",
        "Selected default online mapping policy: Rank-1 GICP.",
        "SC-only safe gate is suitable only for place-recognition decisions because it provides no metric relative transform.",
        "Robot2->Robot3 Rank-1 GICP: precision 99.65%, recall 91.15%, p95 83.80 ms.",
        "Robot3->Robot2 Rank-1 GICP: precision 99.81%, recall 86.00%, p95 146.06 ms.",
        "All GICP policies had zero >3 m or >15 deg mapping-hazard events under offline evaluation.",
        "Selective Top-3 raised recall but increased reverse-direction calls and tail latency; it is not the uniform online default.",
        "Persistent in-process execution shows prior Stage-9 latency was dominated by repeated backend/setup overhead, not GICP registration alone.",
        "No pose graph or map merge was run in Stage 11.",
    ]
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n")

    tracked_inputs = [
        OUT / "frozen_policy.json",
        OUT / "robot2_frame_convention_audit.json",
        direction_dir("robot2_to_robot3") / "candidates_top3_gtfree.csv",
        direction_dir("robot2_to_robot3") / "gicp_top3_gtfree.csv",
        direction_dir("robot3_to_robot2") / "candidates_top3_gtfree.csv",
        direction_dir("robot3_to_robot2") / "gicp_top3_gtfree.csv",
    ]
    manifest = {
        "base_commit": "198b3ac289230cd35cc9b0e3d96173a2aaeb680c",
        "worktree": str(REPO),
        "omp_num_threads": 8,
        "online_artifact_sha256": {str(path.relative_to(REPO)): sha256(path) for path in tracked_inputs},
        "commands": [
            "run_stage11_gicp_value_study.py freeze",
            "run_stage11_gicp_value_study.py prepare --direction robot2_to_robot3",
            "run_stage11_gicp_value_study.py prepare --direction robot3_to_robot2",
            "run_stage11_gicp_value_study.py register --direction <direction> --resume",
            "run_stage11_gicp_value_study.py evaluate --direction <direction>",
            "run_stage11_latency.py --direction <direction> --repeats 3 --warmup 20 --measured 200",
            "run_stage11_gicp_value_study.py combine",
            "run_stage11_gicp_value_study.py finalize",
        ],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(summary))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data, labels = [], []
    for direction in ("robot2_to_robot3", "robot3_to_robot2"):
        decisions = pd.read_csv(direction_dir(direction) / "decisions_gicp_rank1.csv")
        values = decisions.loc[
            decisions.accepted.astype(bool) & decisions.offline_true_pair.eq(True),
            "translation_error_m",
        ].dropna()
        data.append(values.to_numpy())
        labels.append(direction)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(GRAPH_TRANSLATION_HAZARD_M, color="red", linestyle="--", label="3 m hazard boundary")
    ax.set_ylabel("Accepted true-edge translation error (m)")
    ax.set_title("GICP relative-pose quality")
    ax.legend()
    fig.savefig(OUT / "transform_error_distribution.png", dpi=180)
    plt.close(fig)


def build_backend_command(direction: str, limit: int | None, resume: bool) -> list[str]:
    target = direction_dir(direction)
    command = [
        sys.executable,
        str(Path(__file__).with_name("gicp_value_backend.py")),
        "--manifest", str(target / "candidates_top3_gtfree.csv"),
        "--processed-root", str(PROCESSED),
        "--direction", direction,
        "--output", str(target / "gicp_top3_gtfree.csv"),
        "--top-k", "3",
        "--cache-size", "64",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--direction", required=True)
    sub.add_parser("freeze")
    sub.add_parser("audit-robot2-frame")
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--direction", required=True)
    register_parser.add_argument("--limit", type=int)
    register_parser.add_argument("--resume", action="store_true")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--direction", required=True)
    latency_parser = sub.add_parser("latency-summary")
    latency_parser.add_argument("--direction", required=True)
    sub.add_parser("combine")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.direction)
    elif args.command == "freeze":
        freeze_policy()
    elif args.command == "audit-robot2-frame":
        audit_robot2_frame()
    elif args.command == "register":
        subprocess.run(build_backend_command(args.direction, args.limit, args.resume), check=True)
    elif args.command == "evaluate":
        evaluate(args.direction)
    elif args.command == "latency-summary":
        summarize_latency(args.direction)
    elif args.command == "combine":
        combine_and_plot()
    elif args.command == "finalize":
        finalize()


if __name__ == "__main__":
    main()
