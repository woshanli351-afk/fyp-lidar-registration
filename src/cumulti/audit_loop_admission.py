#!/usr/bin/env python3
"""Audit frozen CU-Multi loop-admission counts without loading runtime GT.

The Stage-9 confusion table is an already aggregated offline evaluation artifact.
Rank-1 sanitation and stable-endpoint admission are audited from GT-free columns
only.  The script emits aggregate public summaries, never query-level records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUALITY_THRESHOLD = 0.6091
QUERY_START = 150
CANDIDATE_START = 234
TRANSFORM_DIRECTION = "T_query_from_candidate"

GT_FIELDS = {
    "gt_xy_distance_m",
    "offline_true_pair",
    "translation_error_m",
    "rotation_error_deg",
}
EDGE_KEY = [
    "query_robot",
    "query_keyframe_id",
    "candidate_robot",
    "candidate_keyframe_id",
]
PRESERVED_FIELDS = [
    "query_timestamp",
    "candidate_timestamp",
    "SC_score",
    "best_shift",
    "SC_yaw_initialization_deg",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    "GICP_quality",
    "transform_direction",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(columns) - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return pd.read_csv(path, usecols=columns)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _valid_direction(values: pd.Series) -> bool:
    return bool(values.astype(str).str.startswith(TRANSFORM_DIRECTION).all())


def _check_close(left: pd.Series, right: pd.Series, field: str) -> None:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        ok = np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=0.0, atol=1e-12,
                        equal_nan=True).all()
    else:
        ok = left.astype(str).eq(right.astype(str)).all()
    if not ok:
        raise ValueError(f"stable-endpoint output changed preserved field: {field}")


def build_audit(team_root: Path, source_commit: str) -> tuple[dict[str, Any], pd.DataFrame]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full 40-character lowercase Git SHA")
    checkout_verified = False
    if (team_root / ".git").exists():
        actual_commit = subprocess.run(
            ["git", "-C", str(team_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_commit != source_commit:
            raise ValueError(f"team checkout is {actual_commit}, not declared source commit {source_commit}")
        checkout_verified = True

    output_root = team_root / "outputs/cumulti_v1"
    stage9 = output_root / "09_sc_gicp_integration"
    stage10 = output_root / "10_2_pgo_policy_selection"
    stage12 = output_root / "12d_clean_backend"

    paths = {
        "top3_confusion": stage9 / "verification_confusion.csv",
        "top3_clusters": stage9 / "duplicate_loop_analysis.csv",
        "top3_representatives": stage9 / "sanitized_loop_edges.csv",
        "rank1_summary": stage10 / "rank1_sanitation_summary.json",
        "rank1_representatives": stage10 / "rank1_sanitized_edges.csv",
        "clean_provenance": stage12 / "clean_loop_provenance_audit.json",
        "clean_loops": stage12 / "ordered_rank1_clean_loops.csv",
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(absent))

    # This is the only GT-bearing input, and it is already a four-row aggregate
    # offline evaluation table.  No GT is involved in the admission decisions.
    confusion = _read_csv(paths["top3_confusion"], ["overlap_valid_query", "outcome", "count"])
    outcomes = {str(row.outcome): int(row.count) for row in confusion.itertuples(index=False)}
    expected_outcomes = {"TP", "FP", "FN", "TN"}
    if set(outcomes) != expected_outcomes:
        raise ValueError(f"unexpected confusion outcomes: {sorted(outcomes)}")
    query_count = int(sum(outcomes.values()))
    top3_accepted = outcomes["TP"] + outcomes["FP"]
    top3_rejected = outcomes["FN"] + outcomes["TN"]

    clusters = _read_csv(
        paths["top3_clusters"],
        ["temporal_cluster_id", "edges_before", "kept_edges", "max_quality", "first_query", "last_query"],
    )
    top3_edges = _read_csv(
        paths["top3_representatives"],
        ["SC_rank", "GICP_quality", "accepted_after_sanitation", "transform_direction"],
    )
    top3_representatives = len(top3_edges)
    if int(clusters.edges_before.sum()) != top3_accepted:
        raise ValueError("Top-3 accepted count disagrees with temporal-cluster input count")
    if int(clusters.kept_edges.sum()) != top3_representatives or len(clusters) != top3_representatives:
        raise ValueError("Top-3 representative count disagrees with cluster summary")
    if not clusters.kept_edges.eq(1).all() or not top3_edges.accepted_after_sanitation.astype(bool).all():
        raise ValueError("Top-3 sanitation did not retain exactly one representative per cluster")
    if not top3_edges.GICP_quality.ge(QUALITY_THRESHOLD).all():
        raise ValueError("Top-3 representative below frozen GICP threshold")
    if not _valid_direction(top3_edges.transform_direction):
        raise ValueError("Top-3 transform direction mismatch")

    rank1_summary = json.loads(paths["rank1_summary"].read_text())
    if not np.isclose(float(rank1_summary["quality_threshold"]), QUALITY_THRESHOLD, rtol=0.0, atol=1e-12):
        raise ValueError("Rank-1 summary uses a different GICP threshold")
    rank1_passed = int(rank1_summary["raw_accepted_rank1_edge_count"])
    rank1_representatives = int(rank1_summary["sanitized_rank1_edge_count"])
    rank1_duplicates = int(rank1_summary["temporal_duplicates_removed"])
    if rank1_passed - rank1_representatives != rank1_duplicates:
        raise ValueError("Rank-1 pass, representative, and duplicate counts do not reconcile")
    if int(rank1_summary["cluster_count"]) != rank1_representatives:
        raise ValueError("Rank-1 cluster count does not equal representative count")

    rank1_columns = EDGE_KEY + PRESERVED_FIELDS + [
        "SC_rank",
        "temporal_cluster_id",
        "accepted_after_sanitation",
    ]
    rank1_edges = _read_csv(paths["rank1_representatives"], rank1_columns)
    if len(rank1_edges) != rank1_representatives:
        raise ValueError("Rank-1 representative CSV disagrees with sanitation summary")
    if rank1_edges.duplicated(EDGE_KEY).any() or rank1_edges.temporal_cluster_id.nunique() != rank1_representatives:
        raise ValueError("Rank-1 representatives are not unique by edge and temporal cluster")
    if not rank1_edges.SC_rank.eq(1).all() or not rank1_edges.accepted_after_sanitation.astype(bool).all():
        raise ValueError("Rank-1 representative file contains a non-rank1 or non-admitted row")
    if not rank1_edges.GICP_quality.ge(QUALITY_THRESHOLD).all():
        raise ValueError("Rank-1 representative below frozen GICP threshold")
    if not _valid_direction(rank1_edges.transform_direction):
        raise ValueError("Rank-1 transform direction mismatch")

    clean_provenance = json.loads(paths["clean_provenance"].read_text())
    rank1_hash = sha256(paths["rank1_representatives"])
    if clean_provenance.get("source_sha256") != rank1_hash:
        raise ValueError("stable-endpoint provenance source hash mismatch")
    if clean_provenance.get("status") != "PASS":
        raise ValueError("stable-endpoint provenance audit is not PASS")
    if not clean_provenance.get("arrival_order_GT_free"):
        raise ValueError("stable-endpoint arrival order is not declared GT-free")

    clean_columns = ["arrival_index", "frozen_loop_id"] + EDGE_KEY + PRESERVED_FIELDS
    clean_edges = _read_csv(paths["clean_loops"], clean_columns)
    final_admitted = len(clean_edges)
    if int(clean_provenance["original_rows"]) != rank1_representatives:
        raise ValueError("stable-endpoint original count disagrees with Rank-1 representatives")
    if int(clean_provenance["retained_rows"]) != final_admitted:
        raise ValueError("stable-endpoint retained count disagrees with clean loop CSV")
    if int(clean_provenance["removed_only_by_endpoint_starts"]) != rank1_representatives - final_admitted:
        raise ValueError("stable-endpoint removal count does not reconcile")
    if clean_edges.duplicated(EDGE_KEY).any():
        raise ValueError("clean loop output contains duplicate edge keys")
    if not clean_edges.query_keyframe_id.ge(QUERY_START).all():
        raise ValueError("clean loop output contains a query before the frozen stable start")
    if not clean_edges.candidate_keyframe_id.ge(CANDIDATE_START).all():
        raise ValueError("clean loop output contains a candidate before the frozen stable start")
    if not clean_edges.GICP_quality.ge(QUALITY_THRESHOLD).all():
        raise ValueError("clean loop output contains an edge below the frozen GICP threshold")
    if not _valid_direction(clean_edges.transform_direction):
        raise ValueError("clean loop transform direction mismatch")

    joined = clean_edges.merge(rank1_edges, on=EDGE_KEY, how="left", suffixes=("_clean", "_rank1"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("clean loop output is not an exact subset of Rank-1 representatives")
    for field in PRESERVED_FIELDS:
        _check_close(joined[f"{field}_clean"], joined[f"{field}_rank1"], field)

    files = {
        name: {
            "path": str(path.relative_to(team_root)),
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }
    top3_duplicates = top3_accepted - top3_representatives
    final_removed = rank1_representatives - final_admitted
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "source": {
            "repository": "James-5757/fyp-place-recognition",
            "commit": source_commit,
            "files": files,
        },
        "frozen_policy": {
            "gicp_quality_threshold": QUALITY_THRESHOLD,
            "temporal_cluster_rule": rank1_summary["rule"],
            "stable_query_start_keyframe": QUERY_START,
            "stable_candidate_start_keyframe": CANDIDATE_START,
            "transform_direction": TRANSFORM_DIRECTION,
        },
        "top3_early_stop_historical": {
            "query_count": query_count,
            "gicp_accepted": top3_accepted,
            "gicp_rejected": top3_rejected,
            "gicp_acceptance_rate": _rate(top3_accepted, query_count),
            "offline_confusion": outcomes,
            "temporal_cluster_representatives": top3_representatives,
            "temporal_duplicates_removed": top3_duplicates,
            "temporal_duplicate_removal_rate_of_accepted": _rate(top3_duplicates, top3_accepted),
            "multi_edge_cluster_count": int(clusters.edges_before.gt(1).sum()),
            "representative_rank_counts": {
                str(int(rank)): int(count)
                for rank, count in top3_edges.SC_rank.value_counts().sort_index().items()
            },
        },
        "rank1_current_backend": {
            "query_count": query_count,
            "gicp_passed": rank1_passed,
            "gicp_rejected": query_count - rank1_passed,
            "gicp_pass_rate": _rate(rank1_passed, query_count),
            "temporal_cluster_representatives": rank1_representatives,
            "temporal_duplicates_removed": rank1_duplicates,
            "temporal_duplicate_removal_rate_of_passed": _rate(rank1_duplicates, rank1_passed),
            "stable_endpoint_admitted": final_admitted,
            "removed_by_stable_endpoint_filter": final_removed,
            "final_admission_rate_of_queries": _rate(final_admitted, query_count),
            "final_retention_rate_of_gicp_passed": _rate(final_admitted, rank1_passed),
            "final_retention_rate_of_temporal_representatives": _rate(final_admitted, rank1_representatives),
        },
        "validation": {
            "source_checkout_commit_verified": checkout_verified,
            "counts_reconciled": True,
            "rank1_source_hash_matches_clean_provenance": True,
            "clean_loops_are_exact_rank1_subset": True,
            "preserved_values_unchanged": True,
            "clean_output_has_no_gt_fields": not bool(GT_FIELDS & set(clean_edges.columns)),
            "admission_decisions_use_gt": False,
        },
        "evidence_boundary": {
            "full_pre_sanitation_rank1_rows_available_in_source_commit": False,
            "rank1_gicp_pass_count_source": "frozen rank1_sanitation_summary.json",
            "independently_checked_rows": {
                "rank1_temporal_representatives": rank1_representatives,
                "stable_endpoint_admitted": final_admitted,
            },
            "note": "The 1,233 pre-sanitation Rank-1 pass count is verified against the frozen summary, not recomputed from an uncommitted full candidate table.",
        },
    }

    funnel = pd.DataFrame(
        [
            ["rank1_current_backend", "queries", query_count, query_count, 1.0, 1.0],
            ["rank1_current_backend", "gicp_passed", query_count, rank1_passed,
             _rate(rank1_passed, query_count), _rate(rank1_passed, query_count)],
            ["rank1_current_backend", "temporal_cluster_representatives", rank1_passed, rank1_representatives,
             _rate(rank1_representatives, query_count), _rate(rank1_representatives, rank1_passed)],
            ["rank1_current_backend", "stable_endpoint_admitted", rank1_representatives, final_admitted,
             _rate(final_admitted, query_count), _rate(final_admitted, rank1_representatives)],
            ["top3_early_stop_historical", "queries", query_count, query_count, 1.0, 1.0],
            ["top3_early_stop_historical", "gicp_accepted", query_count, top3_accepted,
             _rate(top3_accepted, query_count), _rate(top3_accepted, query_count)],
            ["top3_early_stop_historical", "temporal_cluster_representatives", top3_accepted, top3_representatives,
             _rate(top3_representatives, query_count), _rate(top3_representatives, top3_accepted)],
        ],
        columns=["policy", "stage", "input_count", "output_count", "rate_of_queries", "retention_from_previous"],
    )
    return summary, funnel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-root", type=Path, required=True, help="checkout of James-5757/fyp-place-recognition")
    parser.add_argument("--source-commit", required=True, help="full source commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for aggregate public summaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, funnel = build_audit(args.team_root.resolve(), args.source_commit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "admission_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    funnel.to_csv(args.output_dir / "admission_funnel.csv", index=False, float_format="%.12g")
    print(
        "PASS: Rank-1 2000 -> "
        f"{summary['rank1_current_backend']['gicp_passed']} -> "
        f"{summary['rank1_current_backend']['temporal_cluster_representatives']} -> "
        f"{summary['rank1_current_backend']['stable_endpoint_admitted']}"
    )


if __name__ == "__main__":
    main()
