#!/usr/bin/env python3
from __future__ import annotations

import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from evaluate_pgo_gravity_offline import (
    ate_rows,
    expected_map_audit,
    expected_metric_audit,
    rigid_align,
    rpe_rows,
    run,
    symmetric_map_consistency,
)


def matrix(translation: np.ndarray, yaw_deg: float = 0.0) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    value[:3, 3] = translation
    return value


def trajectory_and_gt() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, gt = [], {}
    world_rotation = Rotation.from_euler("z", 37, degrees=True).as_matrix()
    world_translation = np.array([10.0, -5.0, 2.0])
    for robot, offset in (("robot1", 0.0), ("robot3", 2.0)):
        gt_rows = []
        for keyframe_id in range(12):
            predicted_position = np.array([float(keyframe_id), offset, 0.1 * keyframe_id])
            gt_position = world_rotation @ predicted_position + world_translation
            predicted_rotation = Rotation.from_euler("z", keyframe_id, degrees=True)
            gt_rotation = Rotation.from_matrix(world_rotation) * predicted_rotation
            pq, gq = predicted_rotation.as_quat(), gt_rotation.as_quat()
            rows.append({
                "robot_id": robot, "keyframe_id": keyframe_id, "timestamp": keyframe_id,
                "tx": predicted_position[0], "ty": predicted_position[1], "tz": predicted_position[2],
                "qx": pq[0], "qy": pq[1], "qz": pq[2], "qw": pq[3],
            })
            gt_rows.append({
                "keyframe_id": keyframe_id,
                "x": gt_position[0], "y": gt_position[1], "z": gt_position[2],
                "qx": gq[0], "qy": gq[1], "qz": gq[2], "qw": gq[3],
                "lidar_file": f"{keyframe_id:06d}.npy",
            })
        gt[robot] = pd.DataFrame(gt_rows).set_index("keyframe_id", drop=False)
    return pd.DataFrame(rows), gt


class OfflineMetricTests(unittest.TestCase):
    def test_rigid_alignment_has_no_scale_and_recovers_rigid_transform(self) -> None:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 1]], dtype=float)
        rotation = Rotation.from_euler("xyz", [5, -3, 40], degrees=True).as_matrix()
        reference = points @ rotation.T + np.array([8.0, -4.0, 2.0])
        aligned, recovered_rotation, _ = rigid_align(points, reference)
        self.assertTrue(np.allclose(aligned, reference, atol=1e-10))
        self.assertTrue(np.allclose(recovered_rotation, rotation, atol=1e-10))

    def test_joint_and_independent_ate_are_zero_under_common_rigid_transform(self) -> None:
        trajectory, gt = trajectory_and_gt()
        rows = ate_rows(trajectory, gt)
        self.assertEqual(len(rows), 5)
        self.assertLess(max(float(row["ATE_RMSE_m"]) for row in rows), 1e-10)

    def test_rpe_is_invariant_to_global_rigid_transform(self) -> None:
        trajectory, gt = trajectory_and_gt()
        rows = rpe_rows(trajectory, gt, interval=10)
        self.assertEqual([row["sample_count"] for row in rows], [2, 2, 4])
        self.assertLess(max(float(row["translation_RMSE_m"]) for row in rows), 1e-10)
        self.assertLess(max(float(row["rotation_RMSE_deg"]) for row in rows), 1e-10)

    def test_symmetric_map_consistency(self) -> None:
        first = np.array([[0.1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]])
        second = first + np.array([0.25, 0, 0])
        result = symmetric_map_consistency(first, second, voxel_m=0.1)
        self.assertAlmostEqual(result["median_m"], 0.25)
        self.assertAlmostEqual(result["p95_m"], 0.25)

    def test_frozen_baseline_audits(self) -> None:
        measured = pd.DataFrame([
            {"variant": "free_pose3_replay", "metric_type": "ATE_joint_rigid_alignment", "scope": "joint", "ATE_RMSE_m": 2.0},
            {"variant": "free_pose3_replay", "metric_type": "RPE", "scope": "combined", "translation_RMSE_m": 1.0, "rotation_RMSE_deg": 3.0},
        ])
        measured_map = pd.DataFrame([{
            "variant": "free_pose3_replay", "median_m": 1.0, "p90_m": 2.0, "p95_m": 3.0,
            "points_robot1": 10, "points_robot3": 20,
        }])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected_metrics = root / "metrics.csv"
            expected_map = root / "map.csv"
            pd.DataFrame([
                {"condition": "ALL_75_LOOPS_CLEAN_GTSAM", "metric_type": "ATE_joint_rigid_alignment", "scope": "joint", "ATE_RMSE_m": 2.0},
                {"condition": "ALL_75_LOOPS_CLEAN_GTSAM", "metric_type": "RPE", "scope": "combined", "translation_RMSE_m": 1.0, "rotation_RMSE_deg": 3.0},
            ]).to_csv(expected_metrics, index=False)
            pd.DataFrame([{
                "condition": "ALL_75_LOOPS_CLEAN_GTSAM", "median_m": 1.0, "p90_m": 2.0, "p95_m": 3.0,
                "points_robot1": 10, "points_robot3": 20,
            }]).to_csv(expected_map, index=False)
            self.assertTrue(expected_metric_audit(
                measured, expected_metrics, "free_pose3_replay", 1e-9
            )["passed"])
            self.assertTrue(expected_map_audit(
                measured_map, expected_map, "free_pose3_replay", 1e-9
            )["passed"])

    def test_end_to_end_offline_phase_writes_freeze_before_metrics(self) -> None:
        trajectory, gt = trajectory_and_gt()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            free_path, hard_path = root / "free.csv", root / "hard.csv"
            robot1_path, robot3_path = root / "robot1.csv", root / "robot3.csv"
            decision_path, output = root / "pre_gt.json", root / "output"
            trajectory.to_csv(free_path, index=False)
            trajectory.to_csv(hard_path, index=False)
            gt["robot1"].to_csv(robot1_path, index=False)
            gt["robot3"].to_csv(robot3_path, index=False)
            decision_path.write_text('{"GT_used": false}\n')
            args = Namespace(
                pre_gt_decision=decision_path,
                free_trajectory=free_path,
                gravity_trajectory=hard_path,
                robot1_keyframes=robot1_path,
                robot3_keyframes=robot3_path,
                robot1_lidar_root=None,
                robot3_lidar_root=None,
                expected_free_metrics=None,
                expected_free_map=None,
                output_dir=output,
                rpe_interval=10,
                map_frame_stride=10,
                map_point_stride=16,
                map_voxel_m=0.5,
                range_min_m=1.0,
                range_max_m=60.0,
                reproduction_tolerance=1e-6,
            )
            with redirect_stdout(StringIO()):
                run(args)
            freeze = output / "offline_evaluation_freeze.json"
            metrics = output / "offline_gt_metrics.csv"
            self.assertTrue(freeze.is_file() and metrics.is_file())
            self.assertLessEqual(freeze.stat().st_mtime_ns, metrics.stat().st_mtime_ns)
            self.assertEqual(len(pd.read_csv(metrics)), 16)


if __name__ == "__main__":
    unittest.main()
