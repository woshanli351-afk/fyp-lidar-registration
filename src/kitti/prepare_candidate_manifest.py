#!/usr/bin/env python3
"""Create a file manifest for computing learned scores on candidate pairs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--sequence-dir", type=Path, required=True, help="KITTI sequences/00 directory")
    parser.add_argument("--modality", choices=["velodyne", "image_0", "image_1"], default="velodyne")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = ".bin" if args.modality == "velodyne" else ".png"
    with args.candidate_csv.open(newline="", encoding="utf-8") as handle:
        candidates = [row for row in csv.DictReader(handle) if int(row["rank"]) <= args.top_k]
    rows = []
    missing = []
    for row in candidates:
        query = args.sequence_dir / args.modality / f"{int(row['query_frame']):06d}{suffix}"
        candidate = args.sequence_dir / args.modality / f"{int(row['candidate_frame']):06d}{suffix}"
        exists = query.is_file() and candidate.is_file()
        if not exists:
            missing.append((query, candidate))
        rows.append({
            "query_frame": row["query_frame"],
            "candidate_frame": row["candidate_frame"],
            "old_rank": row["rank"],
            "query_path": str(query.resolve()),
            "candidate_path": str(candidate.resolve()),
            "files_exist": int(exists),
        })
    if args.require_files and missing:
        raise FileNotFoundError(f"{len(missing)} pairs have one or more missing files; first pair: {missing[0]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Manifest: {args.output}; pairs={len(rows)}; missing_pairs={len(missing)}")


if __name__ == "__main__":
    main()
