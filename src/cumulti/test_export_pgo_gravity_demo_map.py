#!/usr/bin/env python3
"""Unit tests for the presentation-safe Gravity map exporter."""

from __future__ import annotations

import unittest

import numpy as np

from export_pgo_gravity_demo_map import deterministic_indices, deterministic_sample


class DeterministicSampleTest(unittest.TestCase):
    def test_keeps_small_input_unchanged(self) -> None:
        points = np.arange(15, dtype=float).reshape(5, 3)
        np.testing.assert_array_equal(deterministic_sample(points, 5), points)

    def test_is_stable_and_includes_endpoints(self) -> None:
        points = np.arange(60, dtype=float).reshape(20, 3)
        sampled = deterministic_sample(points, 6)
        np.testing.assert_array_equal(sampled[0], points[0])
        np.testing.assert_array_equal(sampled[-1], points[-1])
        np.testing.assert_array_equal(sampled, deterministic_sample(points, 6))

    def test_rejects_non_positive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            deterministic_sample(np.zeros((2, 3)), 0)

    def test_indices_can_sample_points_and_labels_together(self) -> None:
        indices = deterministic_indices(20, 6)
        self.assertEqual(indices.tolist(), [0, 3, 7, 11, 15, 19])
        labels = np.arange(100, 120)
        self.assertEqual(labels[indices].tolist(), [100, 103, 107, 111, 115, 119])


if __name__ == "__main__":
    unittest.main()
