#!/usr/bin/env python3
"""Evaluate ranked loop-closure candidates, including no-overlap rejection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_frame", "rank", "is_positive", "query_has_positive_in_B"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    return rows


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    if not positives:
        return 0.0
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(scores)
    best = (-1.0, float(thresholds[0]))
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1], best[0]


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def evaluate(rows: list[dict[str, str]], score_column: str, ks: list[int], threshold: float | None) -> dict[str, object]:
    if score_column not in rows[0]:
        raise ValueError(f"Score column not found: {score_column}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["query_frame"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: int(row["rank"]))

    valid = [items for items in grouped.values() if int(items[0]["query_has_positive_in_B"]) == 1]
    first_positive_ranks: list[int] = []
    metrics: dict[str, object] = {
        "total_queries": len(grouped),
        "valid_queries": len(valid),
        "no_overlap_queries": len(grouped) - len(valid),
        "overlap_coverage": len(valid) / len(grouped),
    }
    for items in valid:
        positive_ranks = [int(row["rank"]) for row in items if int(row["is_positive"]) == 1]
        if positive_ranks:
            first_positive_ranks.append(min(positive_ranks))
    reciprocal_ranks = [
        (1.0 / min(positive_ranks)) if positive_ranks else 0.0
        for positive_ranks in (
            [int(row["rank"]) for row in items if int(row["is_positive"]) == 1]
            for items in valid
        )
    ]
    metrics["mrr_within_candidate_list"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["mean_first_positive_rank_retrieved"] = float(np.mean(first_positive_ranks)) if first_positive_ranks else None
    metrics["queries_without_positive_in_candidate_list"] = len(valid) - len(first_positive_ranks)

    ranking: dict[str, float] = {}
    for k in ks:
        per_query = [[int(row["is_positive"]) for row in items[:k]] for items in valid]
        ranking[f"recall_at_{k}_valid"] = float(np.mean([any(labels) for labels in per_query]))
        ranking[f"precision_at_{k}_valid"] = float(np.mean([sum(labels) / k for labels in per_query]))
    metrics["ranking"] = ranking

    pair_labels = np.asarray([int(row["is_positive"]) for row in rows], dtype=int)
    pair_scores = np.asarray([float(row[score_column]) for row in rows], dtype=float)
    diagnostic_threshold, diagnostic_f1 = best_f1_threshold(pair_labels, pair_scores)
    metrics["candidate_pair_average_precision"] = average_precision(pair_labels, pair_scores)
    metrics["diagnostic_best_f1_on_same_data"] = {"threshold": diagnostic_threshold, "f1": diagnostic_f1}
    metrics["candidate_pair_threshold_metrics"] = confusion(
        pair_labels, pair_scores, diagnostic_threshold if threshold is None else threshold
    )

    top1 = [items[0] for items in grouped.values()]
    query_labels = np.asarray([int(row["is_positive"]) for row in top1], dtype=int)
    query_scores = np.asarray([float(row[score_column]) for row in top1], dtype=float)
    query_threshold, query_f1 = best_f1_threshold(query_labels, query_scores)
    metrics["top1_all_query_average_precision"] = average_precision(query_labels, query_scores)
    metrics["top1_diagnostic_best_f1_on_same_data"] = {"threshold": query_threshold, "f1": query_f1}
    metrics["top1_all_query_threshold_metrics"] = confusion(
        query_labels, query_scores, query_threshold if threshold is None else threshold
    )
    metrics["warning"] = (
        "Thresholds marked diagnostic were selected on the evaluated data. Tune a deployable threshold on a separate validation split."
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--score-column", default="scan_context_score")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(read_rows(args.candidate_csv), args.score_column, sorted(set(args.ks)), args.score_threshold)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Metrics: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
