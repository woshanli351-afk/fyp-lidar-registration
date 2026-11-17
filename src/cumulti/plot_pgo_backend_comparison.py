#!/usr/bin/env python3
"""Plot the published odometry / Batch LM / iSAM2 comparison summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "odometry_only": "Odometry only",
    "batch_lm": "Batch LM",
    "isam2_default": "iSAM2 default",
}
COLORS = {
    "odometry_only": "#4C78A8",
    "batch_lm": "#F58518",
    "isam2_default": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accuracy = pd.read_csv(args.result_dir / "offline_accuracy_summary.csv")
    performance = pd.read_csv(args.result_dir / "performance_summary.csv")
    numerical = pd.read_csv(args.result_dir / "numerical_agreement_summary.csv").iloc[0]

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)

    # Independent rigid alignment is the fairest per-robot trajectory comparison.
    ax = axes[0, 0]
    independent = accuracy[accuracy.metric_type == "ATE_independent_rigid_alignment"]
    backends = list(LABELS)
    x = np.arange(len(backends))
    width = 0.34
    for offset, robot in [(-width / 2, "robot1"), (width / 2, "robot3")]:
        values = [
            float(independent[(independent.backend == backend) & (independent.scope == robot)].ATE_RMSE_m.iloc[0])
            for backend in backends
        ]
        bars = ax.bar(x + offset, values, width, label=robot.capitalize(), alpha=0.9)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, [LABELS[value] for value in backends])
    ax.set_ylabel("ATE RMSE (m), independent alignment")
    ax.set_title("A. Per-robot trajectory accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # Joint ATE only exists after a cross-robot common frame has been estimated.
    ax = axes[0, 1]
    joint = accuracy[(accuracy.metric_type == "ATE_joint_rigid_alignment") & (accuracy.scope == "joint")]
    joint_backends = ["batch_lm", "isam2_default"]
    values = [float(joint[joint.backend == backend].ATE_RMSE_m.iloc[0]) for backend in joint_backends]
    bars = ax.bar(
        [LABELS[value] for value in joint_backends],
        values,
        color=[COLORS[value] for value in joint_backends],
    )
    ax.bar_label(bars, fmt="%.2f m", padding=3)
    ax.text(
        0.5,
        0.91,
        "Odometry only: N/A (two separate local frames)",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_ylabel("Joint ATE RMSE (m)")
    ax.set_title("B. Common-frame accuracy")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    perf = performance.set_index("backend").loc[backends]
    wall = perf.backend_wall_median_s.to_numpy(float)
    lower = wall - perf.backend_wall_min_s.to_numpy(float)
    upper = perf.backend_wall_max_s.to_numpy(float) - wall
    bars = ax.bar(
        [LABELS[value] for value in backends],
        wall,
        color=[COLORS[value] for value in backends],
        yerr=np.vstack([lower, upper]),
        capsize=4,
    )
    ax.bar_label(bars, labels=[f"{value:.3f} s" for value in wall], padding=3)
    ax.set_yscale("log")
    ax.set_ylabel("Backend wall time (s, log scale)")
    ax.set_title("C. Whole-workload runtime")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.text(
        0.02,
        0.04,
        "iSAM2 distributes work: update p95 = "
        f"{perf.loc['isam2_default', 'isam2_odometry_update_p95_ms']:.2f} ms odom / "
        f"{perf.loc['isam2_default', 'isam2_loop_update_p95_ms']:.2f} ms loop",
        transform=ax.transAxes,
        fontsize=8.5,
    )

    ax = axes[1, 1]
    memory = perf.peak_rss_median_mib.to_numpy(float)
    memory_upper = perf.peak_rss_max_mib.to_numpy(float) - memory
    bars = ax.bar(
        [LABELS[value] for value in backends],
        memory,
        color=[COLORS[value] for value in backends],
        yerr=np.vstack([np.zeros_like(memory_upper), memory_upper]),
        capsize=4,
    )
    ax.bar_label(bars, labels=[f"{value:.1f} MiB" for value in memory], padding=3)
    ax.set_ylim(0, max(memory) * 1.25)
    ax.set_ylabel("Peak RSS (MiB)")
    ax.set_title("D. Process memory")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.95,
        "iSAM2 vs Batch (same gauge):\n"
        f"translation median/p95 = {numerical.translation_median_m:.3f}/{numerical.translation_p95_m:.3f} m\n"
        "frozen gates = 0.10/0.50 m → FAIL",
        va="top",
        transform=ax.transAxes,
        fontsize=8.5,
    )

    fig.suptitle(
        "Frozen CU-Multi graph: Odometry vs Batch LM vs default iSAM2\n"
        "5,796 nodes · 5,794 odometry factors · 75 loop factors · 1 prior",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Controlled server run: 1 warm-up + 3 measured processes/backend; error bars show measured min/max. "
        "GT was loaded only after the backend decision was frozen.",
        ha="center",
        fontsize=8.5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
