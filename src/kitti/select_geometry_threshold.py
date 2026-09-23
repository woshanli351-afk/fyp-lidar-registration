#!/usr/bin/env python3
"""Select a geometric gate on disjoint query blocks and evaluate it once."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_candidates import average_precision, best_f1_threshold, confusion


def read_top1(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_frame", "rank", "is_positive"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    top1 = [row for row in rows if int(row["rank"]) == 1]
    if not top1:
        raise ValueError("Candidate CSV contains no rank-1 rows")
    return top1


def split_by_block(
    rows: list[dict[str, str]], block_size: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    validation = [
        row for row in rows if (int(row["query_frame"]) // block_size) % 2 == 1
    ]
    test = [row for row in rows if (int(row["query_frame"]) // block_size) % 2 == 0]
    if not validation or not test:
        raise ValueError("Block split produced an empty validation or test set")
    return validation, test


def arrays(
    rows: list[dict[str, str]], score_column: str
) -> tuple[np.ndarray, np.ndarray]:
    if score_column not in rows[0]:
        raise ValueError(f"Score column not found: {score_column}")
    labels = np.asarray([int(row["is_positive"]) for row in rows], dtype=int)
    scores = np.asarray([float(row[score_column]) for row in rows], dtype=float)
    return labels, scores


def subset_metrics(
    rows: list[dict[str, str]], score_column: str, threshold: float
) -> dict[str, object]:
    labels, scores = arrays(rows, score_column)
    return {
        "queries": len(rows),
        "positives": int(labels.sum()),
        "average_precision": average_precision(labels, scores),
        "threshold_metrics": confusion(labels, scores, threshold),
    }


def run_protocol(
    rows: list[dict[str, str]], score_column: str, block_size: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    validation, test = split_by_block(rows, block_size)
    validation_labels, validation_scores = arrays(validation, score_column)
    threshold, validation_f1 = best_f1_threshold(
        validation_labels, validation_scores
    )
    result: dict[str, object] = {
        "score_column": score_column,
        "threshold": threshold,
        "threshold_selection": "maximum F1 on validation blocks only",
        "split": {
            "block_size_frames": block_size,
            "validation_blocks": "odd floor(query_frame / block_size)",
            "test_blocks": "even floor(query_frame / block_size)",
        },
        "validation_best_f1": validation_f1,
        "validation": subset_metrics(validation, score_column, threshold),
        "test": subset_metrics(test, score_column, threshold),
    }
    manifest: list[dict[str, object]] = []
    validation_ids = {id(row) for row in validation}
    for row in rows:
        score = float(row[score_column])
        manifest.append(
            {
                **row,
                "evaluation_split": (
                    "validation" if id(row) in validation_ids else "test"
                ),
                "fixed_geometry_threshold": threshold,
                "accepted_loop_closure": int(score >= threshold),
            }
        )
    return result, manifest


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--score-column", default="gicp_quality")
    parser.add_argument("--block-size", type=int, default=250)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, manifest = run_protocol(
        read_top1(args.candidate_csv), args.score_column, args.block_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.manifest_output:
        write_manifest(args.manifest_output, manifest)
    print(f"Fixed threshold: {result['threshold']:.12f}")
    print(f"Metrics: {args.output}")
    if args.manifest_output:
        print(f"Manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
