#!/usr/bin/env python3
"""Render the public offline GT/map comparison for the frozen gravity study."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FREE = "free_pose3_replay"
GRAVITY = "ekf_gravity_hard_diagnostic"
FREE_LABEL = "Free Pose3"
GRAVITY_LABEL = "Hard gravity diagnostic"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    results = root / "results" / "cumulti_pgo_gravity_validation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-metrics", type=Path, default=results / "offline_gt_metrics.csv")
    parser.add_argument("--map-metrics", type=Path, default=results / "map_consistency.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=results / "offline_gt_map_comparison.png",
    )
    return parser.parse_args()


def one_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def render(gt_path: Path, map_path: Path, output: Path) -> None:
    gt = pd.read_csv(gt_path)
    map_metrics = pd.read_csv(map_path)
    expected_variants = {FREE, GRAVITY}
    if set(gt.variant) != expected_variants or set(map_metrics.variant) != expected_variants:
        raise ValueError("offline result variants do not match the frozen comparison")

    scopes = ["joint", "robot1", "robot3"]
    scope_labels = ["Joint", "Robot 1", "Robot 3"]
    values: dict[str, list[float]] = {FREE: [], GRAVITY: []}
    for variant in (FREE, GRAVITY):
        for scope in scopes:
            row = one_row(
                gt,
                variant=variant,
                metric_type="ATE_joint_rigid_alignment",
                scope=scope,
            )
            values[variant].append(float(row.ATE_RMSE_m))

    free_map = one_row(map_metrics, variant=FREE)
    gravity_map = one_row(map_metrics, variant=GRAVITY)
    map_columns = ["median_m", "p90_m", "p95_m"]
    map_labels = ["Median", "P90", "P95"]
    map_change = np.array([
        (float(gravity_map[column]) - float(free_map[column])) / float(free_map[column]) * 100
        for column in map_columns
    ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.3), constrained_layout=False)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.19, wspace=0.34)
    figure.suptitle(
        "Hard gravity improves global ATE, but not map-tail consistency",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )

    x = np.arange(len(scopes))
    width = 0.35
    free_bars = left.bar(
        x - width / 2,
        values[FREE],
        width,
        label=FREE_LABEL,
        color="#4C566A",
        edgecolor="#2E3440",
    )
    gravity_bars = left.bar(
        x + width / 2,
        values[GRAVITY],
        width,
        label=GRAVITY_LABEL,
        color="#009E73",
        edgecolor="#006B4F",
        hatch="//",
    )
    left.set_title("A. Joint-frame trajectory accuracy")
    left.set_ylabel("ATE RMSE (m) — lower is better")
    left.set_xticks(x, scope_labels)
    left.set_ylim(0, max(values[FREE]) * 1.24)
    left.grid(axis="y", alpha=0.25, linewidth=0.8)
    left.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, loc="upper left")
    for bars in (free_bars, gravity_bars):
        left.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    improvements = [
        (before - after) / before * 100
        for before, after in zip(values[FREE], values[GRAVITY])
    ]
    for index, improvement in enumerate(improvements):
        left.text(
            index,
            max(values[FREE][index], values[GRAVITY][index]) + 2.0,
            f"{improvement:.1f}% lower",
            ha="center",
            va="bottom",
            color="#006B4F",
            fontweight="bold",
            fontsize=9,
        )

    y = np.arange(len(map_labels))
    colors = ["#009E73" if value <= 0 else "#D55E00" for value in map_change]
    bars = right.barh(y, map_change, color=colors, edgecolor="#2E3440", height=0.58)
    right.axvline(0, color="#2E3440", linewidth=1)
    right.set_title("B. Cross-robot map NN change")
    right.set_xlabel("Change vs Free Pose3 (%) — lower is better")
    right.set_yticks(y, map_labels)
    right.invert_yaxis()
    right.set_xlim(min(-2.5, float(map_change.min()) - 1.5), float(map_change.max()) + 4.2)
    right.grid(axis="x", alpha=0.25, linewidth=0.8)
    right.spines[["top", "right", "left"]].set_visible(False)
    for index, (bar, column, change) in enumerate(zip(bars, map_columns, map_change)):
        before = float(free_map[column])
        after = float(gravity_map[column])
        anchor = bar.get_width() + 0.28 if change >= 0 else 0.35
        right.text(
            anchor,
            bar.get_y() + bar.get_height() / 2,
            f"{change:+.2f}%   ({before:.2f}→{after:.2f} m)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    right.text(
        0.98,
        0.98,
        "Green = improved; orange = worsened",
        transform=right.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#4C566A",
    )

    figure.text(
        0.5,
        0.065,
        "Frozen 5,796-node / 75-loop graph. GT and point clouds were read only after the pre-GT decision; "
        "the Free Pose3 GT and map baselines reproduce Stage 12D within 1e-6.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4C566A",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    render(args.gt_metrics, args.map_metrics, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
