#!/usr/bin/env python3
"""Generate CVTNet RIV+BEV inputs for only the frames in a candidate CSV.

The official CVTNet KITTI contribution leaves the NCLT binary reader in its
preprocessing entry point.  KITTI odometry scans are instead Nx4 float32.  This
adapter uses the official projection functions with the correct KITTI reader
and writes 10-channel, 32x900 inputs compatible with the public NCLT weights.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

import numpy as np


def candidate_frames(path: Path, top_k: int) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_frame", "candidate_frame", "rank"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    return sorted({
        int(frame)
        for row in rows
        if int(row["rank"]) <= top_k
        for frame in (row["query_frame"], row["candidate_frame"])
    })


def load_kitti_xyz(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"Invalid KITTI Velodyne scan (not Nx4 float32): {path}")
    return values.reshape(-1, 4)[:, :3]


def make_input(
    xyz: np.ndarray,
    range_projection: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    bev_projection: Callable[..., np.ndarray],
    *,
    height: int,
    width: int,
    fov_up: float,
    fov_down: float,
    range_thresholds: list[float],
    height_thresholds: list[float],
) -> np.ndarray:
    if len(range_thresholds) != len(height_thresholds):
        raise ValueError("Range and height threshold lists must have equal length")
    if len(range_thresholds) < 2:
        raise ValueError("At least two range/height thresholds are required")
    max_range = max(range_thresholds)
    output = np.zeros((len(range_thresholds) * 2, height, width), dtype=np.float32)

    output[0], _, _ = range_projection(
        xyz, fov_up=fov_up, fov_down=fov_down, proj_H=height, proj_W=width,
        max_range=max_range, cut_range=True,
        lower_bound=min(range_thresholds), upper_bound=max_range,
    )
    output[len(range_thresholds)] = bev_projection(
        xyz, proj_H=height, proj_W=width, max_range=max_range, cut_height=True,
        lower_bound=min(height_thresholds), upper_bound=max(height_thresholds),
    )
    for index in range(len(range_thresholds) - 1):
        output[index + 1], _, _ = range_projection(
            xyz, fov_up=fov_up, fov_down=fov_down, proj_H=height, proj_W=width,
            max_range=max_range, cut_range=True,
            lower_bound=range_thresholds[index], upper_bound=range_thresholds[index + 1],
        )
        output[index + 1 + len(range_thresholds)] = bev_projection(
            xyz, proj_H=height, proj_W=width, max_range=max_range, cut_height=True,
            lower_bound=height_thresholds[index], upper_bound=height_thresholds[index + 1],
        )
    return output


def parse_float_list(value: str) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--velodyne-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cvtnet-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--height", type=int, default=32, help="32 is required by the public NCLT model")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--fov-up", type=float, default=3.0)
    parser.add_argument("--fov-down", type=float, default=-25.0)
    parser.add_argument("--range-thresholds", type=parse_float_list, default=[0, 15, 30, 45, 80])
    parser.add_argument("--height-thresholds", type=parse_float_list, default=[-3, -1.5, 0, 1.5, 5])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError("top-k, height, and width must be positive")
    sys.path.insert(0, str(args.cvtnet_root.resolve()))
    from tools.utils.utils import bev_projection, range_projection  # type: ignore

    frames = candidate_frames(args.candidate_csv, args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[Path] = []
    generated = skipped = 0
    for index, frame in enumerate(frames, start=1):
        source = args.velodyne_dir / f"{frame:06d}.bin"
        target = args.output_dir / f"{frame:06d}.npy"
        if not source.is_file():
            missing.append(source)
            continue
        if target.is_file() and not args.overwrite:
            skipped += 1
            continue
        tensor = make_input(
            load_kitti_xyz(source), range_projection, bev_projection,
            height=args.height, width=args.width, fov_up=args.fov_up, fov_down=args.fov_down,
            range_thresholds=args.range_thresholds, height_thresholds=args.height_thresholds,
        )
        np.save(target, tensor)
        generated += 1
        if index % 50 == 0 or index == len(frames):
            print(f"Prepared {index}/{len(frames)} unique frames", flush=True)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required scans; first: {missing[0]}")
    print(f"CVTNet inputs: {args.output_dir}; unique={len(frames)} generated={generated} skipped={skipped}")


if __name__ == "__main__":
    main()
