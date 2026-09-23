#!/usr/bin/env python3
"""Extract CVTNet descriptors and score only Scan Context candidate pairs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def read_pairs(path: Path, top_k: int) -> list[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_frame", "candidate_frame", "rank"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    return [
        (int(row["query_frame"]), int(row["candidate_frame"]))
        for row in rows if int(row["rank"]) <= top_k
    ]


def strip_module_prefix(state_dict: dict[str, object]) -> dict[str, object]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cvtnet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor-output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.batch_size <= 0:
        raise ValueError("top-k and batch-size must be positive")
    sys.path.insert(0, str(args.cvtnet_root.resolve()))
    import torch
    from modules.cvtnet import CVTNet  # type: ignore

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    try:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch releases before weights_only was introduced.
        checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = CVTNet(channels=5, use_transformer=True)
    model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    model.to(device).eval()

    pairs = read_pairs(args.candidate_csv, args.top_k)
    frames = sorted({frame for pair in pairs for frame in pair})
    descriptors: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(frames), args.batch_size):
            batch_frames = frames[start:start + args.batch_size]
            arrays = []
            for frame in batch_frames:
                path = args.input_dir / f"{frame:06d}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing CVTNet input: {path}")
                array = np.load(path).astype(np.float32, copy=False)
                if array.shape != (10, 32, 900):
                    raise ValueError(f"Expected (10, 32, 900), got {array.shape}: {path}")
                arrays.append(array)
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            output = model(tensor).detach().cpu().numpy().astype(np.float32)
            for frame, descriptor in zip(batch_frames, output):
                descriptors[frame] = descriptor
            print(f"Described {min(start + args.batch_size, len(frames))}/{len(frames)} frames", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "query_frame", "candidate_frame", "learned_score",
            "cvtnet_cosine_similarity", "cvtnet_squared_l2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query, candidate in pairs:
            query_descriptor = descriptors[query]
            candidate_descriptor = descriptors[candidate]
            denominator = float(np.linalg.norm(query_descriptor) * np.linalg.norm(candidate_descriptor))
            cosine = 0.0 if denominator == 0.0 else float(np.dot(query_descriptor, candidate_descriptor) / denominator)
            squared_l2 = float(np.sum((query_descriptor - candidate_descriptor) ** 2))
            writer.writerow({
                "query_frame": query,
                "candidate_frame": candidate,
                "learned_score": cosine,
                "cvtnet_cosine_similarity": cosine,
                "cvtnet_squared_l2": squared_l2,
            })
    if args.descriptor_output:
        args.descriptor_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.descriptor_output,
            frames=np.asarray(frames, dtype=np.int32),
            descriptors=np.stack([descriptors[frame] for frame in frames]),
        )
    print(f"CVTNet scores: {args.output}; pairs={len(pairs)} unique_frames={len(frames)} device={device}")


if __name__ == "__main__":
    main()
