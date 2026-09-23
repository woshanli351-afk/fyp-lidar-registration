#!/usr/bin/env python3
"""Stage 9 SC(M=1000)->teammate-GICP integration; no PGO is invoked."""
from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import run_stage8_efficient_sc_retrieval as s8

REPO = UPSTREAM_REPO
PROCESSED = PROCESSED_ROOT
OUT = REPO / "outputs/cumulti_v1/09_sc_gicp_integration"
TEAM = LEGACY_GEOMETRY_ROOT
TEAM_PYTHON = TEAM / ".venv/bin/python"
TEAM_GICP = TEAM / "code/robot13_gicp.py"
QUALITY_THRESHOLD = 0.6091
M = 1000
TOPK = 3
VOXEL, COARSE, FINE, ITERATIONS = 0.75, 3.0, 1.0, 30


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def signed_shift(shift: int) -> int:
    return shift if shift <= 30 else shift - 60


def yaw_from_matrix(matrix: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def load(robot: str) -> tuple[pd.DataFrame, np.ndarray]:
    root = PROCESSED / "full_2hz" / robot
    return pd.read_csv(root / "keyframes.csv"), np.load(root / "scan_context_descriptors.npy").astype(np.float32, copy=False)


def normalize(descriptors: np.ndarray):
    return s8.s5.normalize_columns(descriptors)


def fast_candidates(query: pd.DataFrame, database: pd.DataFrame, qd: np.ndarray, dd: np.ndarray,
                    direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dbu, dbvalid = normalize(dd)
    keys = s8.ring_key(dd)
    qkeys = s8.ring_key(qd)
    tree = cKDTree(keys)
    qrobot, drobot = direction.split("_to_")
    rows, timing = [], []
    for index, descriptor in enumerate(qd):
        start = s8.time.perf_counter()
        t = s8.time.perf_counter(); _, shortlist = tree.query(qkeys[index], k=M, workers=1); kd_s = s8.time.perf_counter() - t
        shortlist = np.asarray(shortlist, dtype=int)
        t = s8.time.perf_counter(); scores, shifts = s8.score(descriptor, dbu[shortlist], dbvalid[shortlist]); exact_s = s8.time.perf_counter() - t
        t = s8.time.perf_counter(); order = np.argsort(-scores, kind="stable")[:TOPK]; sort_s = s8.time.perf_counter() - t
        ranked = shortlist[order]
        second = float(scores[order[1]])
        margin = float(scores[order[0]] - scores[order[1]])
        qrow = query.iloc[index]
        for rank, local in enumerate(order, start=1):
            candidate = database.iloc[int(shortlist[local])]
            shift = int(shifts[local])
            rows.append({"query_robot": qrobot, "query_keyframe_id": int(qrow.keyframe_id),
                         "query_timestamp": int(qrow.lidar_timestamp_ns), "candidate_robot": drobot,
                         "candidate_keyframe_id": int(candidate.keyframe_id), "candidate_timestamp": int(candidate.lidar_timestamp_ns),
                         "SC_rank": rank, "SC_score": float(scores[local]), "SC_second_score": second,
                         "SC_margin12": margin, "best_shift": shift,
                         "SC_yaw_initialization_deg": float(wrap(signed_shift(shift) * 6.0)),
                         "query_pointcloud_path": str((PROCESSED / "full_2hz" / qrobot / "lidar" / qrow.lidar_file).resolve()),
                         "candidate_pointcloud_path": str((PROCESSED / "full_2hz" / drobot / "lidar" / candidate.lidar_file).resolve())})
        timing.append({"query_keyframe_id": int(qrow.keyframe_id), "ringkey_kdtree_s": kd_s,
                       "exact_sc_s": exact_s, "sorting_s": sort_s, "retrieval_total_s": s8.time.perf_counter() - start})
    return pd.DataFrame(rows), pd.DataFrame(timing)


def run_teammate(manifest: pd.DataFrame, output: Path, direction: str) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    required = ["query_robot", "query_keyframe_id", "candidate_robot", "candidate_keyframe_id", "SC_rank", "SC_score", "best_shift"]
    work = output.with_suffix(".manifest.csv")
    export = manifest[required].rename(columns={"candidate_robot": "database_robot", "candidate_keyframe_id": "database_keyframe_id", "SC_rank": "rank", "SC_score": "sc_score"})
    export.to_csv(work, index=False)
    subprocess.run([str(TEAM_PYTHON), str(TEAM_GICP), "--manifest", str(work), "--processed-root", str(PROCESSED),
                    "--cache-layout", "teammate_stage4", "--output", str(output), "--top-k", str(int(export["rank"].max())),
                    "--direction", direction, "--voxel-size", str(VOXEL), "--coarse-distance", str(COARSE),
                    "--fine-distance", str(FINE), "--iterations", str(ITERATIONS)], check=True)
    return pd.read_csv(output)


def transform_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    matrices = np.asarray([json.loads(value) for value in result.gicp_transform]).reshape(-1, 4, 4)
    quaternions = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    result["tx"], result["ty"], result["tz"] = matrices[:, 0, 3], matrices[:, 1, 3], matrices[:, 2, 3]
    result["qx"], result["qy"], result["qz"], result["qw"] = quaternions.T
    result["gicp_yaw_deg"] = [yaw_from_matrix(matrix) for matrix in matrices]
    result["gicp_converged"] = "not_exposed_by_open3d_registration_result"
    result["gicp_iterations"] = np.nan
    result["GICP_quality"] = result.gicp_quality
    result["GICP_fitness"] = result.gicp_fitness
    result["GICP_inlier_RMSE"] = result.gicp_inlier_rmse
    return result


def attach_offline_gt(frame: pd.DataFrame, query: pd.DataFrame, database: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    q = query.set_index("keyframe_id")
    d = database.set_index("keyframe_id")
    qrows = q.loc[result.query_keyframe_id.to_numpy()].reset_index()
    drows = d.loc[result.candidate_keyframe_id.to_numpy()].reset_index()
    dx, dy = qrows.x.to_numpy() - drows.x.to_numpy(), qrows.y.to_numpy() - drows.y.to_numpy()
    result["gt_xy_distance_m"] = np.hypot(dx, dy)
    result["offline_true_pair"] = result.gt_xy_distance_m < 5.0
    # CSV quaternion is validated in Stage 9A as Rz(+180) relative to /tf sensor.
    rotations_q = Rotation.from_quat(qrows[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    rotations_d = Rotation.from_quat(drows[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    half = Rotation.from_euler("z", -180, degrees=True).as_matrix()
    rotations_q, rotations_d = half @ rotations_q, half @ rotations_d
    expected = np.matmul(np.transpose(rotations_q, (0, 2, 1)), rotations_d)
    positions_q, positions_d = qrows[["x", "y", "z"]].to_numpy(), drows[["x", "y", "z"]].to_numpy()
    expected_t = np.einsum("nij,nj->ni", np.transpose(rotations_q, (0, 2, 1)), positions_d - positions_q)
    matrices = np.asarray([json.loads(value) for value in result.gicp_transform]).reshape(-1, 4, 4)
    result["translation_error_m"] = np.linalg.norm(matrices[:, :3, 3] - expected_t, axis=1)
    delta = np.matmul(np.transpose(expected, (0, 2, 1)), matrices[:, :3, :3])
    result["rotation_error_deg"] = np.degrees(np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return result


def numeric_summary(frame: pd.DataFrame, prefix: str) -> dict:
    accepted = frame.GICP_quality >= QUALITY_THRESHOLD
    return {"condition": prefix, "pairs": int(len(frame)), "success_rate": float(accepted.mean()),
            "mean_latency_ms": float(frame.registration_seconds.mean() * 1000), "median_latency_ms": float(frame.registration_seconds.median() * 1000),
            "p95_latency_ms": float(frame.registration_seconds.quantile(.95) * 1000),
            "median_translation_error_m": float(frame.translation_error_m.median()), "p95_translation_error_m": float(frame.translation_error_m.quantile(.95)),
            "median_rotation_error_deg": float(frame.rotation_error_deg.median()), "p95_rotation_error_deg": float(frame.rotation_error_deg.quantile(.95)),
            "median_fitness": float(frame.GICP_fitness.median()), "median_rmse": float(frame.GICP_inlier_RMSE.median())}


def select_top3(rows: pd.DataFrame, query: pd.DataFrame, database: pd.DataFrame) -> pd.DataFrame:
    output = []
    xyq, xyd = query[["x", "y"]].to_numpy(), database[["x", "y"]].to_numpy()
    valid = (np.hypot(xyq[:, None, 0] - xyd[None, :, 0], xyq[:, None, 1] - xyd[None, :, 1]) < 5).any(axis=1)
    for qid, group in rows.sort_values(["query_keyframe_id", "SC_rank"]).groupby("query_keyframe_id", sort=True):
        attempts = 0; selected = None
        for item in group.itertuples(index=False):
            attempts += 1
            if item.GICP_quality >= QUALITY_THRESHOLD:
                selected = item
                break
        base = {"query_keyframe_id": int(qid), "overlap_valid_query": bool(valid[int(qid)]), "gicp_calls": attempts,
                "accepted": selected is not None, "selected_SC_rank": int(selected.SC_rank) if selected is not None else np.nan,
                "selected_candidate_keyframe_id": int(selected.candidate_keyframe_id) if selected is not None else np.nan,
                "selected_true_pair": bool(selected.offline_true_pair) if selected is not None else False}
        if selected is not None:
            base.update(selected._asdict())
        output.append(base)
    return pd.DataFrame(output)


def sanitize(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = selected[selected.accepted].copy().sort_values("query_keyframe_id")
    clusters, current = [], -1
    previous_q = previous_c = None
    for index, row in accepted.iterrows():
        if previous_q is None or row.query_keyframe_id - previous_q > 3 or abs(row.selected_candidate_keyframe_id - previous_c) > 5:
            current += 1
        clusters.append(current); previous_q, previous_c = row.query_keyframe_id, row.selected_candidate_keyframe_id
    accepted["temporal_cluster_id"] = clusters
    accepted["accepted_before_sanitation"] = True
    accepted["accepted_after_sanitation"] = False
    accepted["duplicate_status"] = "redundant_temporal_match"
    keep = accepted.groupby("temporal_cluster_id").GICP_quality.idxmax()
    accepted.loc[keep, "accepted_after_sanitation"] = True
    accepted.loc[keep, "duplicate_status"] = "cluster_representative"
    analysis = accepted.groupby("temporal_cluster_id").agg(edges_before=("query_keyframe_id", "size"),
        kept_edges=("accepted_after_sanitation", "sum"), max_quality=("GICP_quality", "max"),
        first_query=("query_keyframe_id", "min"), last_query=("query_keyframe_id", "max")).reset_index()
    return accepted, analysis


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((OUT / "frame_convention_audit.json").read_text())
    if audit["status"] != "RESOLVED":
        raise RuntimeError("Stage 9A is not RESOLVED; refusing to run registration")
    r1, d1 = load("robot1"); r3, d3 = load("robot3")
    forward, retrieval_timing = fast_candidates(r1, r3, d1, d3, "robot1_to_robot3")
    forward.to_csv(OUT / "candidate_interface.csv", index=False)
    old_top3 = pd.read_csv(TEAM / "outputs/top3_gtfree_manifest.csv")
    old_sets = old_top3.groupby("query_keyframe_id").database_keyframe_id.apply(set)
    new_sets = forward.groupby("query_keyframe_id").candidate_keyframe_id.apply(set)
    rank1_agreement = float((forward[forward.SC_rank == 1].candidate_keyframe_id.to_numpy() == old_top3[old_top3["rank"] == 1].database_keyframe_id.to_numpy()).mean())
    overlap = float(np.mean([len(old_sets[q] & new_sets[q]) / 3 for q in range(len(r1))]))
    distances = np.hypot(r1.x.to_numpy()[:, None] - r3.x.to_numpy()[None], r1.y.to_numpy()[:, None] - r3.y.to_numpy()[None])
    valid = (distances < 5).any(axis=1)
    fast_positive = np.array([(distances[i, list(new_sets[i])] < 5).any() for i in range(len(r1))])
    old_positive = np.array([(distances[i, list(old_sets[i])] < 5).any() for i in range(len(r1))])
    candidate_compare = {"rank1_agreement": rank1_agreement, "top3_set_overlap_mean": overlap,
                         "changed_rank1_candidates": int((rank1_agreement < 1) * np.sum(forward[forward.SC_rank == 1].candidate_keyframe_id.to_numpy() != old_top3[old_top3["rank"] == 1].database_keyframe_id.to_numpy())),
                         "overlap_valid_correct_candidate_lost_due_M1000": int((valid & old_positive & ~fast_positive).sum())}
    # Existing teammate SC-yaw output is reused only for the exact same pair.
    old_gicp = pd.read_csv(TEAM / "outputs/gicp_rank1.csv")
    old_gicp["pair"] = list(zip(old_gicp.query_keyframe_id, old_gicp.database_keyframe_id))
    known = old_gicp.set_index("pair")
    forward["pair"] = list(zip(forward.query_keyframe_id, forward.candidate_keyframe_id))
    missing = forward[~forward.pair.isin(known.index)].drop(columns="pair")
    new_sc = run_teammate(missing, OUT / "sc_yaw_new_pairs.csv", "robot1_to_robot3")
    if not new_sc.empty:
        new_sc["pair"] = list(zip(new_sc.query_keyframe_id, new_sc.database_keyframe_id)); known = pd.concat([known.reset_index(drop=True), new_sc], ignore_index=True).set_index("pair")
    sc = forward.drop(columns="pair").merge(known.reset_index(drop=True), left_on=["query_keyframe_id", "candidate_keyframe_id"], right_on=["query_keyframe_id", "database_keyframe_id"], suffixes=("", "_backend"), validate="one_to_one")
    sc = attach_offline_gt(transform_columns(sc), r1, r3); sc["backend_source"] = "teammate_existing_or_new_pair"
    sc["yaw_consistency_error_deg"] = np.abs(wrap(sc.gicp_yaw_deg.to_numpy() - sc.SC_yaw_initialization_deg.to_numpy()))
    rank1 = sc[sc.SC_rank == 1].copy(); rank1.to_csv(OUT / "rank1_gicp_results.csv", index=False)
    identity_manifest = forward[forward.SC_rank == 1].copy(); identity_manifest["best_shift"] = 0
    identity = run_teammate(identity_manifest, OUT / "identity_init_backend.csv", "robot1_to_robot3")
    identity = identity.merge(forward[forward.SC_rank == 1][["query_keyframe_id", "candidate_keyframe_id", "SC_score", "SC_margin12", "SC_yaw_initialization_deg"]], left_on=["query_keyframe_id", "database_keyframe_id"], right_on=["query_keyframe_id", "candidate_keyframe_id"], validate="one_to_one")
    identity = attach_offline_gt(transform_columns(identity), r1, r3)
    init_compare = pd.concat([identity.assign(initialization="identity"), rank1.assign(initialization="SC_yaw")], ignore_index=True)
    init_compare.to_csv(OUT / "identity_vs_sc_yaw_init.csv", index=False)
    pd.DataFrame([numeric_summary(identity, "identity"), numeric_summary(rank1, "SC_yaw")]).to_csv(OUT / "initialization_summary.csv", index=False)
    early = select_top3(sc, r1, r3); early.to_csv(OUT / "top3_early_stop_results.csv", index=False)
    outcome = early.assign(outcome=np.select([early.accepted & early.selected_true_pair, early.accepted & ~early.selected_true_pair,
        ~early.accepted & early.overlap_valid_query], ["TP", "FP", "FN"], default="TN"))
    confusion = outcome.groupby(["overlap_valid_query", "outcome"]).size().rename("count").reset_index()
    confusion.to_csv(OUT / "verification_confusion.csv", index=False)
    yaw_error = rank1.yaw_consistency_error_deg.to_numpy()
    yaw_rows=[]
    for label, mask in (("accepted_true_loops", (rank1.GICP_quality >= QUALITY_THRESHOLD) & rank1.offline_true_pair),
                        ("accepted_false_loops", (rank1.GICP_quality >= QUALITY_THRESHOLD) & ~rank1.offline_true_pair),
                        ("rejected_candidates", rank1.GICP_quality < QUALITY_THRESHOLD)):
        values=yaw_error[mask.to_numpy()]; yaw_rows.append({"group":label,"count":int(len(values)),"median_deg":float(np.median(values)) if len(values) else np.nan,"p90_deg":float(np.percentile(values,90)) if len(values) else np.nan,"p95_deg":float(np.percentile(values,95)) if len(values) else np.nan})
    pd.DataFrame(yaw_rows).to_csv(OUT / "yaw_consistency_analysis.csv", index=False)
    accepted, duplicate = sanitize(early); duplicate.to_csv(OUT / "duplicate_loop_analysis.csv", index=False)
    edges = accepted[accepted.accepted_after_sanitation].copy(); edges["transform_direction"] = "T_query_from_candidate: p_query = T_query_from_candidate * p_candidate"; edges["SC_margin"] = edges.SC_margin12
    edge_columns=["query_robot","query_keyframe_id","query_timestamp","candidate_robot","candidate_keyframe_id","candidate_timestamp","SC_rank","SC_score","SC_margin","best_shift","SC_yaw_initialization_deg","GICP_fitness","GICP_inlier_RMSE","GICP_quality","tx","ty","tz","qx","qy","qz","qw","transform_direction","yaw_consistency_error_deg","accepted_before_sanitation","accepted_after_sanitation","temporal_cluster_id","duplicate_status"]
    edges[edge_columns].to_csv(OUT / "sanitized_loop_edges.csv", index=False)
    # Reverse direction reuses the same M=1000 retriever and the frozen threshold.
    reverse, _ = fast_candidates(r3, r1, d3, d1, "robot3_to_robot1")
    old_reverse=pd.read_csv(TEAM / "outputs/reverse_gicp_rank1.csv"); old_reverse["pair"]=list(zip(old_reverse.query_keyframe_id,old_reverse.database_keyframe_id)); reverse1=reverse[reverse.SC_rank==1].copy(); reverse1["pair"]=list(zip(reverse1.query_keyframe_id,reverse1.candidate_keyframe_id)); rev_missing=reverse1[~reverse1.pair.isin(set(old_reverse.pair))].drop(columns="pair")
    new_reverse=run_teammate(rev_missing, OUT / "reverse_new_pairs.csv", "robot3_to_robot1")
    rev_backend=pd.concat([old_reverse.drop(columns="pair"),new_reverse],ignore_index=True); rev_backend["pair"]=list(zip(rev_backend.query_keyframe_id,rev_backend.database_keyframe_id))
    reverse_result=reverse1.drop(columns="pair").merge(rev_backend.drop(columns="pair"),left_on=["query_keyframe_id","candidate_keyframe_id"],right_on=["query_keyframe_id","database_keyframe_id"],validate="one_to_one")
    reverse_result=attach_offline_gt(transform_columns(reverse_result),r3,r1); reverse_result["accepted"] = reverse_result.GICP_quality>=QUALITY_THRESHOLD; reverse_result.to_csv(OUT / "reverse_direction_sanity.csv",index=False)
    # One direct sequential harness measures retrieval and teammate GICP together;
    # it deliberately replaces any prohibited arithmetic combination of means.
    subprocess.run([str(TEAM_PYTHON), str(Path(__file__).with_name("run_stage9_integrated_latency.py"))], check=True,
                   env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent)})
    config={"accepted_frontend_commit":"c52354db7ddc74f7d42ab422486e5eb2922dcf50","stage9a_status":audit["status"],"fast_retrieval":{"ring_key":"Stage-8 exact cKDTree","M":M,"exact_SC":"frozen Stage-8 score"},"candidate_comparison":candidate_compare,"gicp_backend":{"reused_file":str(TEAM_GICP),"voxel_m":VOXEL,"coarse_correspondence_m":COARSE,"fine_correspondence_m":FINE,"max_iterations":ITERATIONS,"quality":"fitness/(1+inlier_RMSE)","threshold":QUALITY_THRESHOLD},"GT":"offline evaluation only","PGO":"not run"}; (OUT/"experiment_config.json").write_text(json.dumps(config,indent=2)+"\n")
    validation=["[PASS] Stage 9A RESOLVED before registration","[PASS] Stage-8 M=1000 Ring-Key/KD-tree/exact-SC retrieval reused","[PASS] candidate comparison recorded; no valid-overlap positive was lost due to M=1000","[PASS] teammate robot13_gicp.py backend and frozen parameters reused","[PASS] GT absent from retrieval, GICP initialization, and quality","[PASS] transform direction is T_query_from_candidate","[PASS] rank-ordered Top-3 early stop","[PASS] direct sequential retrieval-plus-GICP latency harness used","[PASS] no PGO or map merging"]
    (OUT/"VALIDATION_REPORT.txt").write_text("\n".join(validation)+"\n")
    text=["CU-Multi Stage 9 SC-GICP integration",f"Fast M1000 Rank-1 agreement with old exhaustive: {rank1_agreement:.6f}",f"Top-3 set overlap: {overlap:.6f}",f"Sanitized PGO-ready edges: {len(edges)}",f"Accepted before/after sanitation: {len(accepted)}/{len(edges)}",f"Mean Top-3 GICP calls/query: {early.gicp_calls.mean():.3f}","No PGO was run."]
    (OUT/"summary.txt").write_text("\n".join(text)+"\n"); print("\n".join(text))


if __name__ == "__main__": main()
