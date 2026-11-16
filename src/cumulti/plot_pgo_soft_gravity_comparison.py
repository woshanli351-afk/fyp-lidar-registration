#!/usr/bin/env python3
"""Plot the frozen free, hard-gravity and covariance-soft PGO comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Free Pose3": "#6B7280",
    "Hard diagnostic": "#D97706",
    "Soft covariance x8": "#2563EB",
}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def annotate_bars(axis: plt.Axes, bars: object, decimals: int = 2) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot(results_dir: Path, output: Path) -> None:
    hard = load_json(results_dir / "offline_evaluation_status.json")
    hard_pre_gt = load_json(results_dir / "pre_gt_summary.json")
    soft = load_json(results_dir / "soft_covariance_validation_status.json")

    labels = ["Free Pose3", "Hard diagnostic", "Soft covariance x8"]
    colors = [COLORS[label] for label in labels]
    ate = [
        hard["metrics"]["free_pose3_replay"]["joint_ATE_RMSE_m"],
        hard["metrics"]["ekf_gravity_hard_diagnostic"]["joint_ATE_RMSE_m"],
        soft["offline_result"]["soft_x8"]["joint_ATE_RMSE_m"],
    ]
    map_p90 = [
        hard["metrics"]["free_pose3_replay"]["map_NN_p90_m"],
        hard["metrics"]["ekf_gravity_hard_diagnostic"]["map_NN_p90_m"],
        soft["offline_result"]["soft_x8"]["map_NN_p90_m"],
    ]
    map_p95 = [
        hard["metrics"]["free_pose3_replay"]["map_NN_p95_m"],
        hard["metrics"]["ekf_gravity_hard_diagnostic"]["map_NN_p95_m"],
        soft["offline_result"]["soft_x8"]["map_NN_p95_m"],
    ]
    common_up = [
        hard_pre_gt["metrics"]["free_pose3_replay"]["common_up_p95_deg"],
        hard_pre_gt["metrics"]["ekf_gravity_hard_diagnostic"]["common_up_p95_deg"],
        soft["pre_gt_result"]["soft_common_up_p95_deg"],
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Covariance-soft gravity fixes tilt, but does not pass the map-tail gate",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    bars = axes[0].bar(labels, ate, color=colors, width=0.68)
    annotate_bars(axes[0], bars)
    axes[0].set_title("A. Joint ATE RMSE (lower is better)", fontweight="bold")
    axes[0].set_ylabel("metres")
    axes[0].set_ylim(0, max(ate) * 1.2)

    x = np.arange(len(labels))
    width = 0.36
    bars90 = axes[1].bar(x - width / 2, map_p90, width, color=colors, alpha=0.72, label="p90")
    bars95 = axes[1].bar(x + width / 2, map_p95, width, color=colors, hatch="//", label="p95")
    annotate_bars(axes[1], bars90, 1)
    annotate_bars(axes[1], bars95, 1)
    axes[1].set_title("B. Cross-robot map NN tail (lower is better)", fontweight="bold")
    axes[1].set_ylabel("metres")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(map_p95) * 1.2)
    axes[1].legend(frameon=False, ncols=2, loc="upper left")
    axes[1].text(
        0.5,
        0.03,
        "Soft x8 vs free: p90 +8.90%, p95 +4.96%  →  gate failed",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#991B1B",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FEE2E2", "edgecolor": "#FCA5A5"},
    )

    bars = axes[2].bar(labels, common_up, color=colors, width=0.68)
    annotate_bars(axes[2], bars, 3)
    axes[2].set_title("C. Common up-direction p95 (GT-free)", fontweight="bold")
    axes[2].set_ylabel("degrees")
    axes[2].set_ylim(0, max(common_up) * 1.2)
    axes[2].text(
        0.5,
        0.72,
        "Tilt failure mode confirmed",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=10,
        color="#14532D",
        fontweight="bold",
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelrotation=18)

    figure.text(
        0.5,
        0.012,
        "Frozen 5,796-node / 75-loop Stage 12D graph. GT is used only in panels A–B after the x8 candidate was frozen.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.015, 0.055, 0.995, 0.94), w_pad=2.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cumulti_pgo_gravity_validation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cumulti_pgo_gravity_validation/soft_covariance_comparison.png"),
    )
    args = parser.parse_args()
    plot(args.results_dir, args.output)


if __name__ == "__main__":
    main()
