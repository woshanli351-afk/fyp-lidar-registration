#!/usr/bin/env python3
"""Fuse an external learned score with Scan Context and re-rank candidates.

The learned score CSV must contain ``query_frame``, ``candidate_frame``, and a
score column.  Ground-truth labels are carried to the output for evaluation but
are never used in score fusion or ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_candidates import evaluate, read_rows


def load_learned_scores(path: Path, column: str) -> dict[tuple[int, int], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_frame", "candidate_frame", column}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Learned-score CSV is missing columns: {sorted(missing)}")
    result: dict[tuple[int, int], float] = {}
    for row in rows:
        key = int(row["query_frame"]), int(row["candidate_frame"])
        if key in result:
            raise ValueError(f"Duplicate learned score for pair {key}")
        result[key] = float(row[column])
    return result


def normalize(values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return values
    if method == "query_minmax":
        low, high = float(values.min()), float(values.max())
        return np.zeros_like(values) if high == low else (values - low) / (high - low)
    if method == "query_zscore":
        std = float(values.std())
        return np.zeros_like(values) if std == 0.0 else (values - float(values.mean())) / std
    raise ValueError(f"Unknown normalization method: {method}")


def rerank(
    rows: list[dict[str, str]],
    learned: dict[tuple[int, int], float],
    alpha: float,
    normalization: str,
) -> list[dict[str, object]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["query_frame"])].append(row)
    output: list[dict[str, object]] = []
    missing: list[tuple[int, int]] = []
    for query_frame, items in sorted(grouped.items()):
        items.sort(key=lambda row: int(row["rank"]))
        keys = [(query_frame, int(row["candidate_frame"])) for row in items]
        missing.extend(key for key in keys if key not in learned)
        if missing:
            continue
        scan_scores = normalize(np.asarray([float(row["scan_context_score"]) for row in items]), normalization)
        learned_scores = normalize(np.asarray([learned[key] for key in keys]), normalization)
        final_scores = alpha * scan_scores + (1.0 - alpha) * learned_scores
        ranked = sorted(
            zip(items, learned_scores, final_scores),
            key=lambda item: float(item[2]),
            reverse=True,
        )
        for new_rank, (row, learned_score, final_score) in enumerate(ranked, start=1):
            updated: dict[str, object] = dict(row)
            updated["old_rank"] = int(row["rank"])
            updated["rank"] = new_rank
            updated["learned_score"] = float(learned_score)
            updated["final_score"] = float(final_score)
            output.append(updated)
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"Missing learned scores for {len(missing)} candidate pairs; first pairs: {preview}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("learned_score_csv", type=Path)
    parser.add_argument("--learned-score-column", default="learned_score")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight assigned to Scan Context")
    parser.add_argument("--normalization", choices=["none", "query_minmax", "query_zscore"], default="query_minmax")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.candidate_csv)
    learned = load_learned_scores(args.learned_score_csv, args.learned_score_column)
    reranked = rerank(rows, learned, args.alpha, args.normalization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reranked[0]))
        writer.writeheader(); writer.writerows(reranked)
    print(f"Re-ranked candidates: {args.output}")
    if args.metrics_output:
        metrics = evaluate(
            [{key: str(value) for key, value in row.items()} for row in reranked],
            "final_score", [1, 5, 10], None,
        )
        metrics["fusion"] = {"alpha_scan_context": args.alpha, "normalization": args.normalization}
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
