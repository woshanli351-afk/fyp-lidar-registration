#!/usr/bin/env python3
"""Fuse two keyframe odometry chains with verified inter-robot loop closures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, lil_matrix


def wrap(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def matrix_to_se2(transform: np.ndarray) -> np.ndarray:
    return np.asarray([
        transform[0, 3], transform[1, 3],
        math.atan2(transform[1, 0], transform[0, 0]),
    ])


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(a[2]), math.sin(a[2])
    return np.asarray([
        a[0] + cosine * b[0] - sine * b[1],
        a[1] + sine * b[0] + cosine * b[1],
        wrap(a[2] + b[2]),
    ])


def inverse(a: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(a[2]), math.sin(a[2])
    return np.asarray([
        -cosine * a[0] - sine * a[1],
        sine * a[0] - cosine * a[1],
        wrap(-a[2]),
    ])


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return compose(inverse(a), b)


def read_transform(text: str) -> np.ndarray:
    values = np.asarray(json.loads(text), dtype=float)
    if values.size != 16:
        raise ValueError("Expected a row-major 4x4 transform")
    return values.reshape(4, 4)


@dataclass(frozen=True)
class Edge:
    source_index: int
    target_index: int
    measurement: np.ndarray
    weights: np.ndarray
    kind: str


def load_odometry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 868:
        raise ValueError(f"Expected 868 odometry edges, got {len(rows)}")
    return rows


def load_loops(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Loop-edge file is empty")
    return rows


def initial_chains(
    odometry: list[dict[str, str]], loops: list[dict[str, str]]
) -> tuple[list[tuple[str, int]], np.ndarray]:
    by_agent: dict[str, list[dict[str, str]]] = {"robot_a": [], "robot_b": []}
    for row in odometry:
        by_agent[row["agent"]].append(row)
    node_poses: dict[tuple[str, int], np.ndarray] = {}
    for agent, rows in by_agent.items():
        rows.sort(key=lambda row: int(row["target_frame"]))
        first = int(rows[0]["target_frame"])
        node_poses[(agent, first)] = np.zeros(3)
        for row in rows:
            target = int(row["target_frame"])
            source = int(row["source_frame"])
            measurement = matrix_to_se2(read_transform(row["gicp_transform"]))
            node_poses[(agent, source)] = compose(node_poses[(agent, target)], measurement)

    # Put Robot B in Robot A's coordinate system using the strongest verified loop.
    strongest = max(loops, key=lambda row: float(row["gicp_quality"]))
    query_key = ("robot_a", int(strongest["query_frame"]))
    candidate_key = ("robot_b", int(strongest["candidate_frame"]))
    loop_measurement = matrix_to_se2(read_transform(strongest["transform_row_major"]))
    desired_candidate = compose(node_poses[query_key], loop_measurement)
    alignment = compose(desired_candidate, inverse(node_poses[candidate_key]))
    for key in list(node_poses):
        if key[0] == "robot_b":
            node_poses[key] = compose(alignment, node_poses[key])

    keys = sorted(node_poses, key=lambda key: (key[0], key[1]))
    return keys, np.vstack([node_poses[key] for key in keys])


def make_edges(
    keys: list[tuple[str, int]], odometry: list[dict[str, str]],
    loops: list[dict[str, str]], odom_translation_sigma: float,
    odom_rotation_sigma_degrees: float, loop_translation_sigma: float,
    loop_rotation_sigma_degrees: float,
) -> list[Edge]:
    index = {key: i for i, key in enumerate(keys)}
    odom_weights = np.asarray([
        1.0 / odom_translation_sigma, 1.0 / odom_translation_sigma,
        1.0 / math.radians(odom_rotation_sigma_degrees),
    ])
    loop_weights = np.asarray([
        1.0 / loop_translation_sigma, 1.0 / loop_translation_sigma,
        1.0 / math.radians(loop_rotation_sigma_degrees),
    ])
    edges: list[Edge] = []
    for row in odometry:
        agent = row["agent"]
        edges.append(Edge(
            index[(agent, int(row["target_frame"]))],
            index[(agent, int(row["source_frame"]))],
            matrix_to_se2(read_transform(row["gicp_transform"])),
            odom_weights, "odometry",
        ))
    for row in loops:
        edges.append(Edge(
            index[("robot_a", int(row["query_frame"]))],
            index[("robot_b", int(row["candidate_frame"]))],
            matrix_to_se2(read_transform(row["transform_row_major"])),
            loop_weights, "loop",
        ))
    return edges


def residuals(flat: np.ndarray, anchor: np.ndarray, edges: list[Edge]) -> np.ndarray:
    poses = np.vstack([anchor, flat.reshape(-1, 3)])
    output = np.empty((len(edges), 3), dtype=float)
    for row, edge in enumerate(edges):
        error = relative(edge.measurement, relative(poses[edge.source_index], poses[edge.target_index]))
        error[2] = wrap(error[2])
        output[row] = edge.weights * error
    return output.reshape(-1)


def analytic_jacobian(flat: np.ndarray, anchor: np.ndarray, edges: list[Edge]) -> csr_matrix:
    poses = np.vstack([anchor, flat.reshape(-1, 3)])
    matrix = lil_matrix((3 * len(edges), flat.size), dtype=float)
    for row, edge in enumerate(edges):
        i, j = edge.source_index, edge.target_index
        xi, yi, theta_i = poses[i]
        xj, yj, _ = poses[j]
        dx_world, dy_world = xj - xi, yj - yi
        cosine_i, sine_i = math.cos(theta_i), math.sin(theta_i)
        predicted_x = cosine_i * dx_world + sine_i * dy_world
        predicted_y = -sine_i * dx_world + cosine_i * dy_world
        cosine_z, sine_z = math.cos(edge.measurement[2]), math.sin(edge.measurement[2])
        rotation_z_transpose = np.asarray([[cosine_z, sine_z], [-sine_z, cosine_z]])

        jac_i_pred = np.asarray([
            [-cosine_i, -sine_i, predicted_y],
            [sine_i, -cosine_i, -predicted_x],
            [0.0, 0.0, -1.0],
        ])
        jac_j_pred = np.asarray([
            [cosine_i, sine_i, 0.0],
            [-sine_i, cosine_i, 0.0],
            [0.0, 0.0, 1.0],
        ])
        error_rotation = np.eye(3)
        error_rotation[:2, :2] = rotation_z_transpose
        weighted = np.diag(edge.weights) @ error_rotation
        for node, block in ((i, weighted @ jac_i_pred), (j, weighted @ jac_j_pred)):
            if node > 0:
                matrix[3 * row : 3 * row + 3, 3 * (node - 1) : 3 * node] = block
    return matrix.tocsr()


def jacobian_sparsity(node_count: int, edges: list[Edge]) -> lil_matrix:
    matrix = lil_matrix((3 * len(edges), 3 * (node_count - 1)), dtype=int)
    for row, edge in enumerate(edges):
        for node in (edge.source_index, edge.target_index):
            if node > 0:
                matrix[3 * row : 3 * row + 3, 3 * (node - 1) : 3 * node] = 1
    return matrix


def load_ground_truth(pose_path: Path, calib_path: Path, frames: list[int]) -> np.ndarray:
    camera_poses = np.loadtxt(pose_path).reshape(-1, 3, 4)
    transforms = np.repeat(np.eye(4)[None], camera_poses.shape[0], axis=0)
    transforms[:, :3, :4] = camera_poses
    tr_line = next(
        line for line in calib_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Tr:")
    )
    tr_values = np.asarray([float(value) for value in tr_line.split()[1:]])
    camera_from_velodyne = np.eye(4)
    camera_from_velodyne[:3, :4] = tr_values.reshape(3, 4)
    world_from_velodyne = transforms @ camera_from_velodyne
    anchor_from_world = np.linalg.inv(world_from_velodyne[0])
    relative_poses = anchor_from_world @ world_from_velodyne[frames]
    return np.vstack([matrix_to_se2(transform) for transform in relative_poses])


def trajectory_metrics(estimated: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    translation_errors = np.linalg.norm(estimated[:, :2] - ground_truth[:, :2], axis=1)
    rotation_errors = np.abs(wrap(estimated[:, 2] - ground_truth[:, 2]))
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_mean_m": float(np.mean(translation_errors)),
        "translation_median_m": float(np.median(translation_errors)),
        "rotation_rmse_degrees": float(np.degrees(np.sqrt(np.mean(rotation_errors ** 2)))),
    }


def huber_cost(values: np.ndarray, f_scale: float = 1.0) -> float:
    scaled = (values / f_scale) ** 2
    rho = np.where(scaled <= 1.0, scaled, 2.0 * np.sqrt(scaled) - 1.0)
    return float(0.5 * f_scale ** 2 * np.sum(rho))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry", required=True, type=Path)
    parser.add_argument("--loop-edges", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--odom-translation-sigma", type=float, default=0.50)
    parser.add_argument("--odom-rotation-sigma-degrees", type=float, default=2.0)
    parser.add_argument("--loop-translation-sigma", type=float, default=0.50)
    parser.add_argument("--loop-rotation-sigma-degrees", type=float, default=2.0)
    parser.add_argument("--max-evaluations", type=int, default=1000)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    odometry, loops = load_odometry(args.odometry), load_loops(args.loop_edges)
    keys, initial = initial_chains(odometry, loops)
    edges = make_edges(
        keys, odometry, loops, args.odom_translation_sigma,
        args.odom_rotation_sigma_degrees, args.loop_translation_sigma,
        args.loop_rotation_sigma_degrees,
    )
    result = least_squares(
        residuals, initial[1:].reshape(-1), args=(initial[0], edges),
        jac=analytic_jacobian, method="trf", x_scale="jac",
        loss="huber", f_scale=1.0, max_nfev=args.max_evaluations,
        gtol=args.gradient_tolerance,
        verbose=1,
    )
    optimized = np.vstack([initial[0], result.x.reshape(-1, 3)])
    frames = [frame for _, frame in keys]
    ground_truth = load_ground_truth(args.poses, args.calib, frames)
    metrics = {
        "experiment_type": "real GICP keyframe odometry and loop edges; KITTI GT used only for evaluation",
        "nodes": len(keys), "odometry_edges": len(odometry), "loop_edges": len(loops),
        "optimizer_success": bool(result.success), "optimizer_message": result.message,
        "function_evaluations": int(result.nfev),
        "initial_cost": huber_cost(residuals(initial[1:].reshape(-1), initial[0], edges)),
        "final_cost": float(result.cost),
        "initial_trajectory": trajectory_metrics(initial, ground_truth),
        "optimized_trajectory": trajectory_metrics(optimized, ground_truth),
    }
    for agent in ("robot_a", "robot_b"):
        mask = np.asarray([key[0] == agent for key in keys])
        metrics[f"{agent}_initial"] = trajectory_metrics(initial[mask], ground_truth[mask])
        metrics[f"{agent}_optimized"] = trajectory_metrics(optimized[mask], ground_truth[mask])
    before_rmse = metrics["initial_trajectory"]["translation_rmse_m"]
    after_rmse = metrics["optimized_trajectory"]["translation_rmse_m"]
    before_rotation = metrics["initial_trajectory"]["rotation_rmse_degrees"]
    after_rotation = metrics["optimized_trajectory"]["rotation_rmse_degrees"]
    metrics["improvement"] = {
        "translation_rmse_percent": 100.0 * (before_rmse - after_rmse) / before_rmse,
        "rotation_rmse_percent": 100.0 * (before_rotation - after_rotation) / before_rotation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pose_graph_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "pose_graph_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "frame", "initial_x", "initial_y", "initial_yaw", "optimized_x", "optimized_y", "optimized_yaw", "gt_x", "gt_y", "gt_yaw"])
        for key, before, after, gt in zip(keys, initial, optimized, ground_truth):
            writer.writerow([key[0], key[1], *before, *after, *gt])
    plt.figure(figsize=(10, 7))
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], "k-", linewidth=2, label="KITTI ground truth")
    for agent, color in (("robot_a", "tab:blue"), ("robot_b", "tab:orange")):
        mask = np.asarray([key[0] == agent for key in keys])
        plt.plot(initial[mask, 0], initial[mask, 1], "--", color=color, alpha=0.45, label=f"{agent} before")
        plt.plot(optimized[mask, 0], optimized[mask, 1], "-", color=color, label=f"{agent} optimized")
    plt.title("KITTI 00 multi-robot pose-graph optimization")
    plt.xlabel("Velodyne x in Robot A origin (m)")
    plt.ylabel("Velodyne y in Robot A origin (m)")
    plt.axis("equal"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "pose_graph_trajectory.png", dpi=180)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
