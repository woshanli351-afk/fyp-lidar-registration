#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from diagnose_pgo_map_tail import (
    directed_nearest_neighbours,
    distance_histogram,
    load_lidar_index,
    nearest_loop_context,
    pose_delta_summary,
    symmetric_distances,
    voxel_with_labels,
)


class MapTailDiagnosticTests(unittest.TestCase):
    def test_lidar_index_loads_no_gt_columns(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "keyframes.csv"
            pd.DataFrame({
                "keyframe_id": [1, 2],
                "lidar_file": ["one.npy", "two.npy"],
                "x": [999.0, 999.0],
                "yaw_deg": [999.0, 999.0],
            }).to_csv(path, index=False)
            loaded = load_lidar_index(path, [1, 2])
        self.assertEqual(set(loaded.columns), {"keyframe_id", "lidar_file"})

    def test_voxelization_preserves_selected_frame_labels(self) -> None:
        points = np.array([[0.1, 0, 0], [0.2, 0, 0], [1.1, 0, 0]])
        labels = np.array([10, 11, 12])
        voxelized, selected = voxel_with_labels(points, labels, 1.0)
        self.assertTrue(np.allclose(voxelized, [[0.1, 0, 0], [1.1, 0, 0]]))
        self.assertEqual(selected.tolist(), [10, 12])

    def test_directed_nearest_neighbour_retains_source_and_target_frames(self) -> None:
        result = directed_nearest_neighbours(
            "robot1", "robot3",
            np.array([[0.0, 0, 0], [9.0, 0, 0]]), np.array([1, 2]),
            np.array([[1.0, 0, 0], [10.0, 0, 0]]), np.array([7, 8]),
        )
        self.assertEqual(result.source_keyframe_id.tolist(), [1, 2])
        self.assertEqual(result.target_keyframe_id.tolist(), [7, 8])
        self.assertEqual(result.distance_m.tolist(), [1.0, 1.0])

    def test_symmetric_distances_include_both_directions(self) -> None:
        distances = symmetric_distances(
            np.array([[0.0, 0, 0], [2.0, 0, 0]]),
            np.array([[1.0, 0, 0]]),
        )
        self.assertEqual(sorted(distances.tolist()), [1.0, 1.0, 1.0])

    def test_histogram_uses_full_rows_and_has_overflow(self) -> None:
        frame = pd.DataFrame({
            "source_robot": ["robot1", "robot1", "robot3"],
            "distance_m": [0.2, 1.2, 5.0],
        })
        histogram = distance_histogram(frame, "candidate", 1.0, 3.0)
        symmetric = histogram[histogram.scope.eq("symmetric")]
        self.assertEqual(int(symmetric.point_count.sum()), 3)
        self.assertEqual(int(symmetric[symmetric.is_overflow].point_count.iloc[0]), 1)

    def test_nearest_loop_context_is_robot_specific(self) -> None:
        frames = pd.DataFrame({
            "source_robot": ["robot1", "robot3"],
            "source_keyframe_id": [20, 80],
        })
        loops = pd.DataFrame({
            "arrival_index": [1], "frozen_loop_id": [9],
            "query_robot": ["robot1"], "query_keyframe_id": [25],
            "candidate_robot": ["robot3"], "candidate_keyframe_id": [70],
        })
        result = nearest_loop_context(frames, loops)
        self.assertEqual(result.nearest_loop_keyframe_gap.tolist(), [5, 10])

    def test_pose_delta_reports_up_change(self) -> None:
        base = pd.DataFrame({
            "robot_id": ["robot1"], "keyframe_id": [1], "timestamp": [0],
            "tx": [0.0], "ty": [0.0], "tz": [0.0],
            "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
        })
        angle = np.deg2rad(10.0) / 2
        candidate = base.copy()
        candidate.loc[0, ["qx", "qw"]] = [np.sin(angle), np.cos(angle)]
        result = pose_delta_summary(base, candidate)
        self.assertAlmostEqual(float(result.up_delta_deg.iloc[0]), 10.0, places=8)


if __name__ == "__main__":
    unittest.main()
