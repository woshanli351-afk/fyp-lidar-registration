#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gtsam
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from run_pgo_gravity_validation import (
    Variant,
    angle_deg,
    build_graph,
    first_loop_initialization,
    gravity_factor,
    gravity_diagnostics,
    key,
    load_inputs,
    loop_residuals,
    matrices_from_frame,
)


def pose_row(robot: str, keyframe_id: int, matrix: np.ndarray) -> dict[str, object]:
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "robot_id": robot,
        "keyframe_id": keyframe_id,
        "timestamp": keyframe_id + (0 if robot == "robot1" else 100),
        "tx": matrix[0, 3], "ty": matrix[1, 3], "tz": matrix[2, 3],
        "qx": quaternion[0], "qy": quaternion[1], "qz": quaternion[2], "qw": quaternion[3],
    }


def transform(translation=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    matrix[:3, 3] = translation
    return matrix


class GeometryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        local_rows = [
            pose_row("robot1", 0, transform()),
            pose_row("robot1", 1, transform((1.0, 0.0, 0.0))),
            pose_row("robot3", 0, transform()),
            pose_row("robot3", 1, transform((0.0, 1.0, 0.0))),
        ]
        self.local_frame = pd.DataFrame(local_rows)
        self.local = matrices_from_frame(self.local_frame)
        measurement = transform((1.0, -1.0, 0.0), (0.0, 0.0, 10.0))
        q = Rotation.from_matrix(measurement[:3, :3]).as_quat()
        self.loops = pd.DataFrame([{
            "arrival_index": 1,
            "query_robot": "robot1", "query_keyframe_id": 1,
            "candidate_robot": "robot3", "candidate_keyframe_id": 1,
            "tx": measurement[0, 3], "ty": measurement[1, 3], "tz": measurement[2, 3],
            "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
        }])

    def test_robot_keys_do_not_collide(self) -> None:
        self.assertNotEqual(key("robot1", 1), key("robot3", 1))

    def test_first_loop_initialization_obeys_direction(self) -> None:
        common, audit = first_loop_initialization(self.local, self.loops)
        self.assertLess(audit["translation_residual_m"], 1e-9)
        residuals, _ = loop_residuals(self.loops, common)
        self.assertLess(float(residuals.translation_residual_m.iloc[0]), 1e-9)
        self.assertLess(float(residuals.rotation_residual_deg.iloc[0]), 1e-9)

    def test_up_angle_is_yaw_invariant(self) -> None:
        up = np.array([0.0, 0.0, 1.0])
        yawed = Rotation.from_euler("z", 80, degrees=True).apply(up)
        rolled = Rotation.from_euler("x", 12, degrees=True).apply(up)
        self.assertAlmostEqual(angle_deg(up, yawed), 0.0)
        self.assertAlmostEqual(angle_deg(up, rolled), 12.0)

    def test_gravity_diagnostic_detects_roll_not_yaw(self) -> None:
        common, _ = first_loop_initialization(self.local, self.loops)
        yawed = {node: matrix.copy() for node, matrix in common.items()}
        yawed[("robot1", 1)][:3, :3] = Rotation.from_euler("z", 40, degrees=True).as_matrix() @ common[("robot1", 1)][:3, :3]
        _, yaw_summary = gravity_diagnostics(self.local, common, yawed)
        self.assertLess(yaw_summary["ekf_gravity_deviation_deg"]["max"], 1e-6)
        rolled = {node: matrix.copy() for node, matrix in common.items()}
        rolled[("robot1", 1)][:3, :3] = Rotation.from_euler("x", 8, degrees=True).as_matrix() @ common[("robot1", 1)][:3, :3]
        frame, _ = gravity_diagnostics(self.local, common, rolled)
        value = frame[(frame.robot_id == "robot1") & (frame.keyframe_id == 1)].ekf_gravity_deviation_deg.iloc[0]
        self.assertAlmostEqual(float(value), 8.0, places=6)

    def test_gravity_variant_adds_one_factor_per_non_anchor_node(self) -> None:
        common, _ = first_loop_initialization(self.local, self.loops)
        free_graph, free_counts = build_graph(
            self.local_frame, self.local, self.loops, common,
            Variant("free", False), 0.01,
        )
        hard_graph, hard_counts = build_graph(
            self.local_frame, self.local, self.loops, common,
            Variant("hard", True), 0.01,
        )
        self.assertEqual(free_counts["gravity_factors"], 0)
        self.assertEqual(hard_counts["gravity_factors"], len(self.local) - 1)
        self.assertEqual(hard_graph.size() - free_graph.size(), len(self.local) - 1)

    def test_gravity_factor_leaves_yaw_unpenalized(self) -> None:
        pose_key = key("robot1", 9)
        factor = gravity_factor(pose_key, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.01)
        values = gtsam.Values()
        values.insert(pose_key, gtsam.Pose3(gtsam.Rot3.Rz(np.deg2rad(75)), np.zeros(3)))
        self.assertAlmostEqual(float(factor.error(values)), 0.0, places=12)

    def test_runtime_loader_does_not_load_gt_columns(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            local_path, loops_path = root / "local.csv", root / "loops.csv"
            local_with_gt = self.local_frame.assign(gt_x=123.0)
            loops_with_gt = self.loops.assign(gt_distance_m=0.0)
            local_with_gt.to_csv(local_path, index=False)
            loops_with_gt.to_csv(loops_path, index=False)
            loaded_local, loaded_loops = load_inputs(local_path, loops_path)
            self.assertNotIn("gt_x", loaded_local)
            self.assertNotIn("gt_distance_m", loaded_loops)


if __name__ == "__main__":
    unittest.main()
