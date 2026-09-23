#!/usr/bin/env python3
"""Direct sequential latency harness for Stage-11 GICP policies."""
from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import run_stage8_efficient_sc_retrieval as s8
from gicp_value_backend import GICPBackend, parse_direction, signed_shift


REPO = Path(__file__).resolve().parents[2]
PROCESSED = PROCESSED_ROOT
OUT = REPO / "outputs/cumulti_v1/11_gicp_value_study"
M = 1000
TOP_K = 3
GICP_THRESHOLD = 0.6091
POLICIES = ("sc_rank1", "sc_score_gate", "gicp_rank1", "gicp_top3", "selective_gicp")


def load_robot(robot: str) -> tuple[pd.DataFrame, np.ndarray]:
    root = PROCESSED / "full_2hz" / robot
    return (
        pd.read_csv(root / "keyframes.csv"),
        np.load(root / "scan_context_descriptors.npy").astype(np.float32, copy=False),
    )


class Retriever:
    def __init__(self, direction: str) -> None:
        self.query_robot, self.database_robot = parse_direction(direction)
        self.query, self.query_desc = load_robot(self.query_robot)
        self.database, self.database_desc = load_robot(self.database_robot)
        self.database_unit, self.database_valid = s8.s5.normalize_columns(self.database_desc)
        self.query_keys = s8.ring_key(self.query_desc)
        self.tree = cKDTree(s8.ring_key(self.database_desc))

    def retrieve(self, query_index: int) -> tuple[list[dict[str, object]], float]:
        start = time.perf_counter()
        _, shortlist = self.tree.query(self.query_keys[query_index], k=M, workers=1)
        shortlist = np.asarray(shortlist, dtype=int)
        scores, shifts = s8.score(
            self.query_desc[query_index],
            self.database_unit[shortlist],
            self.database_valid[shortlist],
        )
        order = np.argsort(-scores, kind="stable")[:TOP_K]
        query_row = self.query.iloc[query_index]
        rows = []
        for rank, local in enumerate(order, start=1):
            candidate = self.database.iloc[int(shortlist[local])]
            shift = int(shifts[local])
            rows.append({
                "query_robot": self.query_robot,
                "query_keyframe_id": int(query_row.keyframe_id),
                "candidate_robot": self.database_robot,
                "candidate_keyframe_id": int(candidate.keyframe_id),
                "SC_rank": rank,
                "SC_score": float(scores[local]),
                "best_shift": shift,
                "SC_yaw_initialization_deg": signed_shift(shift) * 6.0,
            })
        return rows, time.perf_counter() - start


def execute_policy(
    policy: str,
    candidates: list[dict[str, object]],
    backend: GICPBackend,
    sc_threshold: float,
) -> tuple[int, bool, float, float]:
    calls = 0
    accepted = False
    cloud_s = 0.0
    registration_s = 0.0

    def call(candidate: dict[str, object]) -> dict[str, object]:
        nonlocal calls, cloud_s, registration_s
        calls += 1
        result = backend.register(candidate)
        cloud_s += sum(float(result[name]) for name in (
            "source_load_s", "source_preprocess_s", "target_load_s", "target_preprocess_s"
        ))
        registration_s += float(result["registration_s"])
        return result

    rank1 = candidates[0]
    if policy == "sc_rank1":
        accepted = True
    elif policy == "sc_score_gate":
        accepted = float(rank1["SC_score"]) >= sc_threshold
    elif policy == "gicp_rank1":
        accepted = float(call(rank1)["gicp_quality"]) >= GICP_THRESHOLD
    elif policy == "gicp_top3":
        for candidate in candidates:
            if float(call(candidate)["gicp_quality"]) >= GICP_THRESHOLD:
                accepted = True
                break
    elif policy == "selective_gicp":
        if float(rank1["SC_score"]) >= sc_threshold:
            accepted = True
        else:
            for candidate in candidates:
                if float(call(candidate)["gicp_quality"]) >= GICP_THRESHOLD:
                    accepted = True
                    break
    else:
        raise ValueError(policy)
    return calls, accepted, cloud_s, registration_s


def run(direction: str, repeats: int, warmup: int, measured: int) -> None:
    policy_data = json.loads((OUT / "frozen_policy.json").read_text())
    sc_threshold = float(policy_data["sc_safe_threshold"])
    retriever = Retriever(direction)
    sample_count = warmup + measured
    sample = np.linspace(0, len(retriever.query) - 1, sample_count, dtype=int)
    if len(np.unique(sample)) != sample_count:
        raise RuntimeError("Latency sample indices are not unique")
    rows = []
    for policy in POLICIES:
        for repeat in range(repeats):
            backend = GICPBackend(PROCESSED, direction, cache_size=64)
            for position, query_index in enumerate(sample):
                start = time.perf_counter()
                candidates, retrieval_s = retriever.retrieve(int(query_index))
                calls, accepted, cloud_s, registration_s = execute_policy(
                    policy, candidates, backend, sc_threshold
                )
                total_s = time.perf_counter() - start
                if position >= warmup:
                    rows.append({
                        "direction": direction,
                        "policy": policy,
                        "repeat": repeat,
                        "query_keyframe_id": int(retriever.query.iloc[int(query_index)].keyframe_id),
                        "retrieval_s": retrieval_s,
                        "cloud_io_preprocess_s": cloud_s,
                        "registration_s": registration_s,
                        "other_s": max(0.0, total_s - retrieval_s - cloud_s - registration_s),
                        "total_s": total_s,
                        "gicp_calls": calls,
                        "accepted": accepted,
                    })
    target = OUT / direction
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target / "latency_trials.csv", index=False)
    print(json.dumps({
        "direction": direction,
        "rows": len(rows),
        "repeats": repeats,
        "warmup_queries_per_policy_repeat": warmup,
        "measured_queries_per_policy_repeat": measured,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=200)
    args = parser.parse_args()
    parse_direction(args.direction)
    run(args.direction, args.repeats, args.warmup, args.measured)


if __name__ == "__main__":
    main()
