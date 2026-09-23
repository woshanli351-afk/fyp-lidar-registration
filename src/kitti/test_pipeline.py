#!/usr/bin/env python3

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_candidates import evaluate, read_rows
from export_loop_edges import export_edges, parse_transform
from generate_cvtnet_inputs import candidate_frames, load_kitti_xyz, make_input
from geometric_verify_candidates import select_rows, transform_summary, yaw_initialization
from estimate_keyframe_odometry import agent_frames, sequential_edges
from optimize_pose_graph_se2 import compose as compose_se2, inverse as inverse_se2, relative as relative_se2
from rerank_candidates import rerank
from scan_context_baseline import aligned_similarity, frame_ranges, make_scan_context
from score_cvtnet_candidates import strip_module_prefix
from select_geometry_threshold import run_protocol, split_by_block


class ScanContextTests(unittest.TestCase):
    def test_shift_alignment(self):
        descriptor = np.zeros((3, 8), dtype=np.float32)
        descriptor[:, 1] = [1, 2, 4]
        descriptor[:, 5] = [3, 1, 2]
        shifted = np.roll(descriptor, 3, axis=1)
        score, _ = aligned_similarity(descriptor, shifted)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_descriptor_shape_and_height(self):
        xyz = np.asarray([[1.0, 0.0, 0.5], [1.0, 0.0, 1.5], [100.0, 0.0, 9.0]])
        descriptor = make_scan_context(xyz, 20, 60, 80.0, 2.0)
        self.assertEqual(descriptor.shape, (20, 60))
        self.assertAlmostEqual(float(descriptor.max()), 3.5)

    def test_thesis_frame_counts(self):
        queries, database = frame_ranges(4541, 2500, 100, 5)
        self.assertEqual((len(queries), len(database)), (481, 389))


class EvaluationTests(unittest.TestCase):
    def test_csv_metrics(self):
        rows = [
            {"query_frame": "0", "rank": "1", "scan_context_score": "0.9", "is_positive": "1", "query_has_positive_in_B": "1"},
            {"query_frame": "0", "rank": "2", "scan_context_score": "0.2", "is_positive": "0", "query_has_positive_in_B": "1"},
            {"query_frame": "1", "rank": "1", "scan_context_score": "0.6", "is_positive": "0", "query_has_positive_in_B": "1"},
            {"query_frame": "1", "rank": "2", "scan_context_score": "0.5", "is_positive": "1", "query_has_positive_in_B": "1"},
            {"query_frame": "2", "rank": "1", "scan_context_score": "0.4", "is_positive": "0", "query_has_positive_in_B": "0"},
            {"query_frame": "2", "rank": "2", "scan_context_score": "0.3", "is_positive": "0", "query_has_positive_in_B": "0"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            result = evaluate(read_rows(path), "scan_context_score", [1, 2], None)
        self.assertEqual(result["total_queries"], 3)
        self.assertEqual(result["valid_queries"], 2)
        self.assertAlmostEqual(result["ranking"]["recall_at_1_valid"], 0.5)
        self.assertAlmostEqual(result["ranking"]["recall_at_2_valid"], 1.0)
        self.assertAlmostEqual(result["mrr_within_candidate_list"], 0.75)

    def test_reranking_interface(self):
        rows = [
            {"query_frame": "0", "candidate_frame": "10", "rank": "1", "scan_context_score": "0.9", "is_positive": "0", "query_has_positive_in_B": "1"},
            {"query_frame": "0", "candidate_frame": "11", "rank": "2", "scan_context_score": "0.8", "is_positive": "1", "query_has_positive_in_B": "1"},
        ]
        learned = {(0, 10): 0.1, (0, 11): 0.9}
        result = rerank(rows, learned, alpha=0.3, normalization="none")
        self.assertEqual(int(result[0]["candidate_frame"]), 11)
        self.assertEqual(int(result[0]["old_rank"]), 2)
        self.assertEqual(int(result[0]["rank"]), 1)


class CVTNetAdapterTests(unittest.TestCase):
    def test_candidate_frames_and_kitti_reader(self):
        rows = [
            {"query_frame": "0", "candidate_frame": "20", "rank": "1"},
            {"query_frame": "0", "candidate_frame": "25", "rank": "2"},
            {"query_frame": "5", "candidate_frame": "30", "rank": "11"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "candidates.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            self.assertEqual(candidate_frames(csv_path, 10), [0, 20, 25])
            scan_path = Path(tmp) / "000000.bin"
            np.asarray([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], dtype=np.float32).tofile(scan_path)
            np.testing.assert_array_equal(load_kitti_xyz(scan_path), [[1, 2, 3], [4, 5, 6]])

    def test_ten_channel_projection_layout(self):
        calls = {"range": 0, "bev": 0}

        def range_projection(_xyz, **kwargs):
            calls["range"] += 1
            value = kwargs["upper_bound"]
            array = np.full((kwargs["proj_H"], kwargs["proj_W"]), value, dtype=np.float32)
            return array, np.empty(0), np.empty(0)

        def bev_projection(_xyz, **kwargs):
            calls["bev"] += 1
            return np.full((kwargs["proj_H"], kwargs["proj_W"]), kwargs["upper_bound"], dtype=np.float32)

        result = make_input(
            np.ones((2, 3)), range_projection, bev_projection,
            height=4, width=8, fov_up=3, fov_down=-25,
            range_thresholds=[0, 15, 30, 45, 80],
            height_thresholds=[-3, -1.5, 0, 1.5, 5],
        )
        self.assertEqual(result.shape, (10, 4, 8))
        self.assertEqual(calls, {"range": 5, "bev": 5})
        self.assertEqual(result.dtype, np.float32)

    def test_data_parallel_prefix_removal(self):
        self.assertEqual(strip_module_prefix({"module.a": 1, "module.b": 2}), {"a": 1, "b": 2})
        self.assertEqual(strip_module_prefix({"a": 1}), {"a": 1})


class GeometricVerificationTests(unittest.TestCase):
    def test_se2_group_operations(self):
        pose = np.asarray([2.0, -1.0, 0.4])
        np.testing.assert_allclose(compose_se2(pose, inverse_se2(pose)), np.zeros(3), atol=1e-12)
        other = np.asarray([5.0, 3.0, -0.2])
        np.testing.assert_allclose(compose_se2(pose, relative_se2(pose, other)), other, atol=1e-12)

    def test_sequential_odometry_edge_counts(self):
        frames = agent_frames()
        self.assertEqual((len(frames["robot_a"]), len(frames["robot_b"])), (481, 389))
        edges = sequential_edges(frames)
        self.assertEqual(len(edges), 868)
        self.assertEqual(edges[0], ("robot_a", 0, 5))
        self.assertEqual(edges[-1], ("robot_b", 4535, 4540))

    def test_scan_context_shift_initialization(self):
        transform = yaw_initialization(15, 60)
        np.testing.assert_allclose(transform[:2, :2], [[0, -1], [1, 0]], atol=1e-12)
        translation, yaw = transform_summary(transform)
        self.assertAlmostEqual(translation, 0.0)
        self.assertAlmostEqual(yaw, 90.0)

    def test_top_k_selection_order(self):
        rows = [
            {"query_frame": "5", "candidate_frame": "30", "rank": "2", "best_shift": "0"},
            {"query_frame": "0", "candidate_frame": "20", "rank": "1", "best_shift": "0"},
            {"query_frame": "0", "candidate_frame": "25", "rank": "4", "best_shift": "0"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            selected = select_rows(path, 3)
        self.assertEqual([(r["query_frame"], r["rank"]) for r in selected], [("0", "1"), ("5", "2")])


class GeometryThresholdTests(unittest.TestCase):
    def test_disjoint_block_split_and_validation_only_threshold(self):
        rows = [
            {"query_frame": "0", "rank": "1", "is_positive": "1", "gicp_quality": "0.90"},
            {"query_frame": "50", "rank": "1", "is_positive": "0", "gicp_quality": "0.20"},
            {"query_frame": "100", "rank": "1", "is_positive": "1", "gicp_quality": "0.80"},
            {"query_frame": "150", "rank": "1", "is_positive": "0", "gicp_quality": "0.30"},
        ]
        validation, test = split_by_block(rows, 100)
        self.assertEqual([int(row["query_frame"]) for row in validation], [100, 150])
        self.assertEqual([int(row["query_frame"]) for row in test], [0, 50])
        result, manifest = run_protocol(rows, "gicp_quality", 100)
        self.assertAlmostEqual(result["threshold"], 0.8)
        self.assertEqual(result["test"]["threshold_metrics"]["tp"], 1)
        self.assertEqual(result["test"]["threshold_metrics"]["fp"], 0)
        self.assertEqual(len(manifest), 4)

    def test_loop_edge_export_preserves_measurement_direction(self):
        transform = np.eye(4)
        transform[0, 3] = 1.25
        row = {
            "query_frame": "10", "candidate_frame": "200", "gicp_quality": "0.8",
            "gicp_fitness": "0.9", "gicp_inlier_rmse": "0.2",
            "evaluation_split": "test", "accepted_loop_closure": "1",
            "is_positive": "1", "gicp_transform": str(transform.reshape(-1).tolist()),
        }
        edges = export_edges([row], "test")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["measurement_convention"], "candidate_to_query_velodyne")
        parsed = parse_transform(edges[0]["transform_row_major"])
        self.assertAlmostEqual(parsed[0, 3], 1.25)


if __name__ == "__main__":
    unittest.main()
