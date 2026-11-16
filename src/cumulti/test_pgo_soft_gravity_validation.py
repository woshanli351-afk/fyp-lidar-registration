#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from run_pgo_soft_gravity_validation import load_covariance_sigmas, validate_scales


class SoftGravityInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = pd.DataFrame({
            "robot_id": ["robot1", "robot1", "robot3"],
            "keyframe_id": [0, 1, 0],
        })

    def test_loader_keeps_gt_columns_out_and_skips_unmatched_node(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for robot, ids, decoded in (
                ("robot1", [0, 1], [True, False]),
                ("robot3", [0], [True]),
            ):
                path = root / f"{robot}.csv"
                pd.DataFrame({
                    "robot_id": [robot] * len(ids),
                    "keyframe_id": ids,
                    "decoded_from_raw": decoded,
                    "roll_variance_rad2": [1e-6] * len(ids),
                    "pitch_variance_rad2": [3e-6] * len(ids),
                    "gt_x": [999.0] * len(ids),
                }).to_csv(path, index=False)
                paths.append(path)
            sigmas, summary = load_covariance_sigmas(paths, self.local)
        self.assertEqual(set(sigmas), {("robot1", 0), ("robot3", 0)})
        self.assertEqual(summary["missing_nodes"], [["robot1", 1]])
        self.assertNotIn("gt_x", summary)

    def test_loader_rejects_non_positive_decoded_variance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for robot, ids in (("robot1", [0, 1]), ("robot3", [0])):
                path = root / f"{robot}.csv"
                pd.DataFrame({
                    "robot_id": [robot] * len(ids),
                    "keyframe_id": ids,
                    "decoded_from_raw": [True] * len(ids),
                    "roll_variance_rad2": [0.0] * len(ids),
                    "pitch_variance_rad2": [1e-6] * len(ids),
                }).to_csv(path, index=False)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "non-positive"):
                load_covariance_sigmas(paths, self.local)

    def test_scale_grid_must_be_positive_unique_and_increasing(self) -> None:
        validate_scales([1.0, 2.0, 4.0, 8.0])
        for invalid in ([], [0.0], [2.0, 1.0], [1.0, 1.0]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_scales(invalid)


if __name__ == "__main__":
    unittest.main()
