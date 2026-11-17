#!/usr/bin/env python3
"""Render the GT-free PGO map-tail attribution figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FREE = "#6B7280"
SOFT = "#2563EB"


def annotate_bars(axis: plt.Axes, bars: object, formatter: str = "{:.1f}") -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            formatter.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot(results_dir: Path, output: Path) -> None:
    summary = json.loads(
        (results_dir / "map_tail_diagnostic_summary.json").read_text(encoding="utf-8")
    )
    sensitivity = pd.read_csv(results_dir / "map_tail_loop_local_sensitivity.csv")
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Whole-map tail reflects unequal coverage; loop-local change is only 9–22 mm",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    whole = summary["whole_map_reproduction"]
    bars = axes[0, 0].bar(
        ["Free Pose3", "Soft covariance ×8"],
        [whole["free"]["p95_m"], whole["soft_x8"]["p95_m"]],
        color=[FREE, SOFT],
        width=0.62,
    )
    annotate_bars(axes[0, 0], bars)
    axes[0, 0].set_title("A. Original whole-map symmetric p95", fontweight="bold")
    axes[0, 0].set_ylabel("nearest-neighbour distance (m)")
    axes[0, 0].set_ylim(0, 145)
    axes[0, 0].text(
        0.5,
        0.05,
        "111,959 tail points: 100% Robot3 and >300 keyframes from a loop endpoint",
        transform=axes[0, 0].transAxes,
        ha="center",
        fontsize=9,
        color="#991B1B",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FEE2E2", "edgecolor": "#FCA5A5"},
    )

    directional = summary["directional_decomposition"]
    labels = ["Robot1 → Robot3\n(better covered)", "Robot3 → Robot1\n(coverage limited)"]
    free_values = [
        directional["robot1_to_robot3"]["free"]["p95_m"],
        directional["robot3_to_robot1"]["free"]["p95_m"],
    ]
    soft_values = [
        directional["robot1_to_robot3"]["soft_x8"]["p95_m"],
        directional["robot3_to_robot1"]["soft_x8"]["p95_m"],
    ]
    x = np.arange(2)
    axes[0, 1].plot(x, free_values, "o-", color=FREE, linewidth=2, markersize=8, label="Free")
    axes[0, 1].plot(x, soft_values, "o-", color=SOFT, linewidth=2, markersize=8, label="Soft ×8")
    for position, values, color, offset in (
        (x, free_values, FREE, (0, -14)),
        (x, soft_values, SOFT, (0, 10)),
    ):
        for px, value in zip(position, values):
            axes[0, 1].annotate(
                f"{value:.2f} m", (px, value), xytext=offset, textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center", ha="center",
            )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("p95 distance (m, log scale)")
    axes[0, 1].set_title("B. Direction reveals the coverage asymmetry", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    thresholds = ["fraction_within_5m", "fraction_within_10m"]
    categories = ["R1→R3 ≤5 m", "R1→R3 ≤10 m", "R3→R1 ≤5 m", "R3→R1 ≤10 m"]
    free_coverage = [
        100 * directional["robot1_to_robot3"]["free"][thresholds[0]],
        100 * directional["robot1_to_robot3"]["free"][thresholds[1]],
        100 * directional["robot3_to_robot1"]["free"][thresholds[0]],
        100 * directional["robot3_to_robot1"]["free"][thresholds[1]],
    ]
    soft_coverage = [
        100 * directional["robot1_to_robot3"]["soft_x8"][thresholds[0]],
        100 * directional["robot1_to_robot3"]["soft_x8"][thresholds[1]],
        100 * directional["robot3_to_robot1"]["soft_x8"][thresholds[0]],
        100 * directional["robot3_to_robot1"]["soft_x8"][thresholds[1]],
    ]
    x = np.arange(len(categories))
    width = 0.36
    free_bars = axes[1, 0].bar(x - width / 2, free_coverage, width, color=FREE, label="Free")
    soft_bars = axes[1, 0].bar(x + width / 2, soft_coverage, width, color=SOFT, label="Soft ×8")
    annotate_bars(axes[1, 0], free_bars, "{:.1f}%")
    annotate_bars(axes[1, 0], soft_bars, "{:.1f}%")
    axes[1, 0].set_xticks(x, categories, rotation=16, ha="right")
    axes[1, 0].set_ylim(0, 110)
    axes[1, 0].set_ylabel("voxels within threshold (%)")
    axes[1, 0].set_title("C. Robot1 is largely covered; Robot3 is not", fontweight="bold")
    axes[1, 0].legend(frameon=False, ncols=2)

    sensitivity = sensitivity.copy()
    sensitivity["setting"] = sensitivity.apply(
        lambda row: f"stride {int(row.point_stride)}, voxel {row.voxel_m:g} m", axis=1
    )
    sensitivity["p95_delta_mm"] = 1000 * sensitivity.p95_m_delta
    bars = axes[1, 1].barh(
        sensitivity.setting, sensitivity.p95_delta_mm, color=SOFT, alpha=0.88
    )
    for bar in bars:
        value = float(bar.get_width())
        axes[1, 1].annotate(
            f"+{value:.1f} mm", (value, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0), textcoords="offset points", va="center",
            fontsize=9, fontweight="bold",
        )
    axes[1, 1].axvline(0, color="#111827", linewidth=0.9)
    axes[1, 1].set_xlim(0, max(sensitivity.p95_delta_mm) * 1.35)
    axes[1, 1].set_xlabel("soft ×8 minus free loop-local p95 (mm)")
    axes[1, 1].set_title("D. Accepted-loop overlap: small, consistent regression", fontweight="bold")

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
    figure.text(
        0.5,
        0.012,
        "GT-free diagnostic on the frozen 5,796-node / 75-loop graph. Bars start at zero; panel B uses a labelled log point scale.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.015, 0.045, 0.995, 0.95), h_pad=2.6, w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results/cumulti_pgo_gravity_validation"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/cumulti_pgo_gravity_validation/map_tail_diagnostic.png"),
    )
    args = parser.parse_args()
    plot(args.results_dir, args.output)


if __name__ == "__main__":
    main()
