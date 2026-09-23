#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from gicp_value_backend import assert_gt_free_fields, parse_direction, signed_shift, yaw_initialization
from run_stage11_gicp_value_study import assign_clusters, pick_policy


def candidate(rank: int, score: float, quality: float, true_pair: bool = True) -> dict[str, object]:
    return {
        "query_keyframe_id": 10,
        "candidate_keyframe_id": 100 + rank,
        "SC_rank": rank,
        "SC_score": score,
        "gicp_quality": quality,
        "gt_xy_distance_m": 1.0 if true_pair else 10.0,
        "offline_true_pair": true_pair,
        "translation_error_m": 0.2,
        "rotation_error_deg": 1.0,
        "mapping_hazard": False,
        "pair_wall_s": 0.1,
        "source_load_s": 0.01,
        "source_preprocess_s": 0.01,
        "coarse_registration_s": 0.02,
        "fine_registration_s": 0.02,
        "target_load_s": 0.01,
        "target_preprocess_s": 0.01,
    }


class BackendContractTests(unittest.TestCase):
    def test_any_cross_robot_direction(self) -> None:
        self.assertEqual(parse_direction("robot2_to_robot3"), ("robot2", "robot3"))
        self.assertEqual(parse_direction("robot3_to_robot2"), ("robot3", "robot2"))
        with self.assertRaises(ValueError):
            parse_direction("robot2_to_robot2")
        with self.assertRaises(ValueError):
            parse_direction("robot4_to_robot1")

    def test_signed_scan_context_yaw(self) -> None:
        self.assertEqual(signed_shift(0), 0)
        self.assertEqual(signed_shift(30), 30)
        self.assertEqual(signed_shift(31), -29)
        transform, yaw = yaw_initialization(59)
        self.assertEqual(yaw, -6.0)
        self.assertTrue(np.allclose(transform[:2, :2], [[np.cos(np.deg2rad(-6)), -np.sin(np.deg2rad(-6))],
                                                        [np.sin(np.deg2rad(-6)), np.cos(np.deg2rad(-6))]]))

    def test_online_manifest_rejects_gt(self) -> None:
        assert_gt_free_fields(["query_robot", "SC_score"])
        with self.assertRaises(ValueError):
            assert_gt_free_fields(["query_robot", "gt_xy_distance_m"])
        with self.assertRaises(ValueError):
            assert_gt_free_fields(["offline_true_pair"])


class PolicyTests(unittest.TestCase):
    def test_strict_sc_gate(self) -> None:
        group = pd.DataFrame([candidate(1, 0.8, 0.1), candidate(2, 0.7, 0.9), candidate(3, 0.6, 0.9)])
        frozen = np.nextafter(0.75, np.inf)
        rejected = pick_policy(group.assign(SC_score=[0.75, 0.7, 0.6]), "sc_score_gate", frozen)
        accepted = pick_policy(group.assign(SC_score=[frozen, 0.7, 0.6]), "sc_score_gate", frozen)
        self.assertFalse(rejected["accepted"])
        self.assertTrue(accepted["accepted"])

    def test_top3_early_stop(self) -> None:
        group = pd.DataFrame([candidate(1, 0.8, 0.4), candidate(2, 0.7, 0.7), candidate(3, 0.6, 0.8)])
        result = pick_policy(group, "gicp_top3", 0.9)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["gicp_calls"], 2)
        self.assertEqual(int(result["SC_rank"]), 2)
        self.assertTrue(result["pose_used"])

    def test_selective_bypass_and_fallback(self) -> None:
        high = pd.DataFrame([candidate(1, 0.9, 0.1), candidate(2, 0.7, 0.9), candidate(3, 0.6, 0.9)])
        bypass = pick_policy(high, "selective_gicp", 0.8)
        self.assertEqual(bypass["gicp_calls"], 0)
        self.assertFalse(bypass["pose_available_for_mapping"])
        low = high.copy()
        low.loc[0, "SC_score"] = 0.7
        fallback = pick_policy(low, "selective_gicp", 0.8)
        self.assertEqual(fallback["gicp_calls"], 2)
        self.assertTrue(fallback["pose_available_for_mapping"])

    def test_temporal_event_clustering(self) -> None:
        frame = pd.DataFrame({
            "query_keyframe_id": [1, 3, 10],
            "candidate_keyframe_id": [100, 103, 200],
        })
        clusters = assign_clusters(frame, pd.Series([True, True, True]))
        self.assertEqual(int(clusters.nunique()), 2)


if __name__ == "__main__":
    unittest.main()
