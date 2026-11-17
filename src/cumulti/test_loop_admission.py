import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import audit_loop_admission as audit


class LoopAdmissionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = self.root / "outputs/cumulti_v1"
        self.stage9 = base / "09_sc_gicp_integration"
        self.stage10 = base / "10_2_pgo_policy_selection"
        self.stage12 = base / "12d_clean_backend"
        for path in (self.stage9, self.stage10, self.stage12):
            path.mkdir(parents=True)
        self._write_fixture()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _edge(query_id, candidate_id, cluster_id):
        return {
            "query_robot": "robot1",
            "query_keyframe_id": query_id,
            "query_timestamp": query_id * 1_000,
            "candidate_robot": "robot3",
            "candidate_keyframe_id": candidate_id,
            "candidate_timestamp": candidate_id * 1_000,
            "SC_rank": 1,
            "SC_score": 0.8,
            "best_shift": 3,
            "SC_yaw_initialization_deg": 18.0,
            "tx": 1.0,
            "ty": 2.0,
            "tz": 3.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "GICP_quality": 0.7,
            "transform_direction": "T_query_from_candidate: p_query = T * p_candidate",
            "temporal_cluster_id": cluster_id,
            "accepted_after_sanitation": True,
            # Present in the historical source, but deliberately never loaded by
            # the Rank-1 admission audit.
            "offline_true_pair": True,
        }

    def _write_fixture(self):
        pd.DataFrame(
            [
                [True, "TP", 2],
                [False, "FP", 1],
                [True, "FN", 1],
                [False, "TN", 0],
            ],
            columns=["overlap_valid_query", "outcome", "count"],
        ).to_csv(self.stage9 / "verification_confusion.csv", index=False)
        pd.DataFrame(
            [
                [0, 2, 1, 0.8, 10, 11],
                [1, 1, 1, 0.7, 20, 20],
            ],
            columns=["temporal_cluster_id", "edges_before", "kept_edges", "max_quality", "first_query", "last_query"],
        ).to_csv(self.stage9 / "duplicate_loop_analysis.csv", index=False)
        pd.DataFrame(
            [
                [1, 0.8, True, "T_query_from_candidate"],
                [2, 0.7, True, "T_query_from_candidate"],
            ],
            columns=["SC_rank", "GICP_quality", "accepted_after_sanitation", "transform_direction"],
        ).to_csv(self.stage9 / "sanitized_loop_edges.csv", index=False)

        (self.stage10 / "rank1_sanitation_summary.json").write_text(
            json.dumps(
                {
                    "quality_threshold": audit.QUALITY_THRESHOLD,
                    "raw_accepted_rank1_edge_count": 3,
                    "sanitized_rank1_edge_count": 2,
                    "temporal_duplicates_removed": 1,
                    "cluster_count": 2,
                    "rule": "query gap >3 OR absolute candidate gap >5 starts a new cluster; maximum GICP_quality represents each cluster",
                }
            )
        )
        rank1 = pd.DataFrame([self._edge(100, 200, 0), self._edge(160, 240, 1)])
        rank1_path = self.stage10 / "rank1_sanitized_edges.csv"
        rank1.to_csv(rank1_path, index=False)

        clean_row = self._edge(160, 240, 1)
        clean_row.pop("SC_rank")
        clean_row.pop("temporal_cluster_id")
        clean_row.pop("accepted_after_sanitation")
        clean_row.pop("offline_true_pair")
        clean_row.update({"arrival_index": 0, "frozen_loop_id": 1})
        pd.DataFrame([clean_row]).to_csv(self.stage12 / "ordered_rank1_clean_loops.csv", index=False)
        source_hash = hashlib.sha256(rank1_path.read_bytes()).hexdigest()
        (self.stage12 / "clean_loop_provenance_audit.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "original_rows": 2,
                    "retained_rows": 1,
                    "removed_only_by_endpoint_starts": 1,
                    "source_sha256": source_hash,
                    "transform_values_unchanged": True,
                    "arrival_order_GT_free": True,
                }
            )
        )

    def test_happy_path_reconciles_both_funnels(self):
        result, funnel = audit.build_audit(self.root, "a" * 40)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["top3_early_stop_historical"]["gicp_accepted"], 3)
        self.assertEqual(result["top3_early_stop_historical"]["temporal_cluster_representatives"], 2)
        self.assertEqual(result["rank1_current_backend"]["gicp_passed"], 3)
        self.assertEqual(result["rank1_current_backend"]["temporal_cluster_representatives"], 2)
        self.assertEqual(result["rank1_current_backend"]["stable_endpoint_admitted"], 1)
        self.assertAlmostEqual(result["rank1_current_backend"]["final_admission_rate_of_queries"], 0.25)
        self.assertEqual(len(funnel), 7)
        self.assertNotIn("offline_true_pair", json.dumps(result))

    def test_changed_transform_is_rejected(self):
        path = self.stage12 / "ordered_rank1_clean_loops.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "tx"] = 99.0
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "changed preserved field: tx"):
            audit.build_audit(self.root, "b" * 40)

    def test_provenance_hash_mismatch_is_rejected(self):
        path = self.stage12 / "clean_loop_provenance_audit.json"
        value = json.loads(path.read_text())
        value["source_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            audit.build_audit(self.root, "c" * 40)

    def test_pre_stable_endpoint_is_rejected(self):
        path = self.stage12 / "ordered_rank1_clean_loops.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "query_keyframe_id"] = audit.QUERY_START - 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "query before the frozen stable start"):
            audit.build_audit(self.root, "d" * 40)


if __name__ == "__main__":
    unittest.main()
