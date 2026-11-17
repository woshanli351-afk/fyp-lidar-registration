import json
import tempfile
import unittest
from pathlib import Path

import evaluate_pgo_backend_comparison as comparison


class BackendComparisonAggregationTests(unittest.TestCase):
    def test_distribution_requires_three_repetitions(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            comparison.distribution([1.0, 2.0])
        result = comparison.distribution([1.0, 2.0, 3.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median"], 2.0)

    def test_run_loader_rejects_cross_backend_input_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for backend_index, backend in enumerate(comparison.BACKENDS):
                for repetition in range(1, 4):
                    value = {
                        "mode": backend,
                        "repetition": repetition,
                        "warmup": False,
                        "host": "same-host",
                        "thread_environment": {"OMP_NUM_THREADS": "1"},
                        "inputs": {"local": "same", "loops": str(backend_index)},
                        "GT_used": False,
                    }
                    path = root / f"{backend}_{repetition}.json"
                    path.write_text(json.dumps(value))
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "identical frozen inputs"):
                comparison.load_runs(paths)

    def test_performance_summary_uses_independent_process_memory(self):
        runs = {}
        for backend_index, backend in enumerate(comparison.BACKENDS):
            runs[backend] = [
                {
                    "backend": {"backend_wall_s": float(value)},
                    "process_wall_s": float(value + 1),
                    "peak_rss_kib": 1024.0 * (100 + backend_index + value),
                }
                for value in (1, 2, 3)
            ]
        rows = comparison.performance_rows(runs)
        self.assertEqual([row["backend"] for row in rows], list(comparison.BACKENDS))
        self.assertEqual(rows[0]["backend_wall_median_s"], 2.0)
        self.assertEqual(rows[0]["peak_rss_median_mib"], 102.0)

    def test_frozen_isam2_gate_fails_translation(self):
        checks = {
            field: comparison.EXPECTED_TEAM_ISAM2[field] <= limit
            for field, limit in comparison.AGREEMENT_GATES.items()
        }
        self.assertFalse(checks["translation_median_m"])
        self.assertFalse(checks["translation_p95_m"])
        self.assertTrue(checks["rotation_median_deg"])
        self.assertTrue(checks["rotation_p95_deg"])


if __name__ == "__main__":
    unittest.main()
