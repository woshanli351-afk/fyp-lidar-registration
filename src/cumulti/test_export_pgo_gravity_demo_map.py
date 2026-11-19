#!/usr/bin/env python3
"""Unit tests for the presentation-safe Gravity map exporter."""

from __future__ import annotations

import unittest

import numpy as np

from export_pgo_gravity_demo_map import deterministic_sample


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


if __name__ == "__main__":
    unittest.main()
