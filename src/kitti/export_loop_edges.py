#!/usr/bin/env python3
"""Export accepted geometric verifications as pose-graph loop-edge records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_transform(text: str) -> np.ndarray:
    values = np.asarray(json.loads(text), dtype=float)
    if values.size != 16:
        raise ValueError(f"Expected 16 transform values, got {values.size}")
    transform = values.reshape(4, 4)
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous transform bottom row")
    return transform


def export_edges(
    rows: list[dict[str, str]], split: str
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for row in rows:
        if int(row["accepted_loop_closure"]) != 1:
            continue
        if split != "all" and row["evaluation_split"] != split:
            continue
        transform = parse_transform(row["gicp_transform"])
        edges.append(
            {
                "edge_type": "inter_robot_loop_closure",
                "query_frame": int(row["query_frame"]),
                "candidate_frame": int(row["candidate_frame"]),
                "measurement_convention": "candidate_to_query_velodyne",
                "gicp_quality": float(row["gicp_quality"]),
                "gicp_fitness": float(row["gicp_fitness"]),
                "gicp_inlier_rmse": float(row["gicp_inlier_rmse"]),
                "evaluation_split": row["evaluation_split"],
                "is_positive_for_evaluation_only": int(row["is_positive"]),
                "transform_row_major": json.dumps(transform.reshape(-1).tolist()),
            }
        )
    return edges


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "query_frame", "candidate_frame", "gicp_transform", "gicp_quality",
        "gicp_fitness", "gicp_inlier_rmse", "evaluation_split",
        "accepted_loop_closure", "is_positive",
    }
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Fixed-gate manifest is missing columns: {sorted(missing)}")
    return rows


def write_edges(path: Path, edges: list[dict[str, object]]) -> None:
    if not edges:
        raise ValueError("No accepted edges matched the requested split")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edges[0]))
        writer.writeheader()
        writer.writerows(edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=["all", "validation", "test"], default="test")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges = export_edges(read_rows(args.manifest), args.split)
    write_edges(args.output, edges)
    print(f"Loop edges: {len(edges)} -> {args.output}")


if __name__ == "__main__":
    main()
