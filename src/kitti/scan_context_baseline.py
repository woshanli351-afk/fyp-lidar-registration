#!/usr/bin/env python3
"""Reproducible Scan Context baseline for the KITTI odometry dataset.

The script deliberately keeps the implementation dependency-light.  It reads
KITTI Velodyne ``.bin`` scans and pose files, constructs two temporally
separated pseudo-robot trajectories, retrieves Robot B candidates for every
Robot A query, and writes the candidate schema used by the thesis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def load_kitti_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != 12:
        raise ValueError(f"Expected 12 values per KITTI pose, got {values.shape[1]}")
    return values.reshape(-1, 3, 4)


def load_velodyne_xyz(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"Invalid Velodyne scan (not Nx4 float32): {path}")
    return values.reshape(-1, 4)[:, :3]


def make_scan_context(
    xyz: np.ndarray,
    num_rings: int = 20,
    num_sectors: int = 60,
    max_radius: float = 80.0,
    lidar_height: float = 2.0,
) -> np.ndarray:
    """Create a max-height ring-sector descriptor from one LiDAR scan."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    keep = (radius > 0.0) & (radius <= max_radius)
    if not np.any(keep):
        return np.zeros((num_rings, num_sectors), dtype=np.float32)

    radius = radius[keep]
    angles = np.mod(np.arctan2(xyz[keep, 1], xyz[keep, 0]), 2.0 * np.pi)
    heights = xyz[keep, 2] + lidar_height
    ring = np.minimum((radius / max_radius * num_rings).astype(int), num_rings - 1)
    sector = np.minimum((angles / (2.0 * np.pi) * num_sectors).astype(int), num_sectors - 1)

    descriptor = np.zeros((num_rings, num_sectors), dtype=np.float32)
    np.maximum.at(descriptor, (ring, sector), heights.astype(np.float32))
    return descriptor


def aligned_similarity(query: np.ndarray, candidate: np.ndarray) -> tuple[float, int]:
    """Return the best circular-column cosine similarity and column shift."""
    if query.shape != candidate.shape or query.ndim != 2:
        raise ValueError("Descriptors must be two-dimensional and have equal shapes")
    best_score, best_shift = -1.0, 0
    for shift in range(query.shape[1]):
        shifted = np.roll(candidate, shift, axis=1)
        q_norm = np.linalg.norm(query, axis=0)
        c_norm = np.linalg.norm(shifted, axis=0)
        valid = (q_norm > 0.0) & (c_norm > 0.0)
        score = 0.0 if not np.any(valid) else float(
            np.mean(np.sum(query[:, valid] * shifted[:, valid], axis=0) / (q_norm[valid] * c_norm[valid]))
        )
        if score > best_score:
            best_score, best_shift = score, shift
    return best_score, best_shift


def frame_ranges(num_frames: int, split_frame: int, boundary_gap: int, step: int) -> tuple[list[int], list[int]]:
    a_stop = max(0, split_frame - boundary_gap)
    b_start = min(num_frames, split_frame + boundary_gap)
    return list(range(0, a_stop + 1, step)), list(range(b_start, num_frames, step))


def descriptor_for_frame(velodyne_dir: Path, frame: int, args: argparse.Namespace) -> np.ndarray:
    scan_path = velodyne_dir / f"{frame:06d}.bin"
    if not scan_path.is_file():
        raise FileNotFoundError(f"Missing Velodyne frame: {scan_path}")
    return make_scan_context(
        load_velodyne_xyz(scan_path), args.num_rings, args.num_sectors, args.max_radius, args.lidar_height
    )


def nearest_positive(position: np.ndarray, db_positions: np.ndarray, db_frames: list[int]) -> tuple[float, int]:
    distances = np.linalg.norm(db_positions - position, axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), db_frames[index]


def evaluate_rows(rows: Iterable[dict[str, object]], ks: tuple[int, ...]) -> dict[str, float | int]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["query_frame"]), []).append(row)
    valid = [items for items in grouped.values() if int(items[0]["query_has_positive_in_B"]) == 1]
    result: dict[str, float | int] = {"total_queries": len(grouped), "valid_queries": len(valid)}
    result["overlap_coverage"] = len(valid) / len(grouped) if grouped else 0.0
    for k in ks:
        labels = [[int(r["is_positive"]) for r in sorted(items, key=lambda x: int(x["rank"]))[:k]] for items in valid]
        result[f"recall_at_{k}_valid"] = float(np.mean([any(x) for x in labels])) if labels else 0.0
        result[f"precision_at_{k}_valid"] = float(np.mean([sum(x) / k for x in labels])) if labels else 0.0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", type=Path, required=True, help="Directory containing sequences/00/velodyne")
    parser.add_argument("--pose-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--split-frame", type=int, default=2500)
    parser.add_argument("--boundary-gap", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--positive-threshold", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-rings", type=int, default=20)
    parser.add_argument("--num-sectors", type=int, default=60)
    parser.add_argument("--max-radius", type=float, default=80.0)
    parser.add_argument("--lidar-height", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0 or args.top_k <= 0 or args.boundary_gap < 0:
        raise ValueError("frame-step/top-k must be positive and boundary-gap non-negative")
    poses = load_kitti_poses(args.pose_file)
    velodyne_dir = args.kitti_root / "sequences" / args.sequence / "velodyne"
    query_frames, db_frames = frame_ranges(len(poses), args.split_frame, args.boundary_gap, args.frame_step)
    if not query_frames or not db_frames:
        raise ValueError("The split produced an empty Robot A or Robot B trajectory")

    print(f"Building Robot B descriptor database ({len(db_frames)} frames)...")
    db_descriptors = [descriptor_for_frame(velodyne_dir, frame, args) for frame in db_frames]
    positions = poses[:, [0, 2], 3]
    db_positions = positions[db_frames]
    rows: list[dict[str, object]] = []

    for index, query_frame in enumerate(query_frames, start=1):
        query_descriptor = descriptor_for_frame(velodyne_dir, query_frame, args)
        scored = [(*aligned_similarity(query_descriptor, descriptor), db_frame) for descriptor, db_frame in zip(db_descriptors, db_frames)]
        scored.sort(key=lambda item: item[0], reverse=True)
        nearest_dist, nearest_frame = nearest_positive(positions[query_frame], db_positions, db_frames)
        has_positive = int(nearest_dist < args.positive_threshold)
        for rank, (score, shift, candidate_frame) in enumerate(scored[: args.top_k], start=1):
            gt_distance = float(np.linalg.norm(positions[query_frame] - positions[candidate_frame]))
            rows.append({
                "query_frame": query_frame,
                "candidate_frame": candidate_frame,
                "rank": rank,
                "scan_context_score": score,
                "best_shift": shift,
                "gt_distance": gt_distance,
                "is_positive": int(gt_distance < args.positive_threshold),
                "query_has_positive_in_B": has_positive,
                "nearest_gt_dist_to_B": nearest_dist,
                "nearest_gt_frame_in_B": nearest_frame,
            })
        if index % 25 == 0 or index == len(query_frames):
            print(f"Processed {index}/{len(query_frames)} Robot A queries")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"split{args.split_frame}_gap{args.boundary_gap}_step{args.frame_step}_thr{args.positive_threshold:g}"
    csv_path = args.output_dir / f"{stem}_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = evaluate_rows(rows, tuple(sorted({1, 5, args.top_k})))
    summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metrics_path = args.output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Candidates: {csv_path}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
