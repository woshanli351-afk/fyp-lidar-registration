#!/usr/bin/env python3
from __future__ import annotations

import unittest

import pandas as pd

from audit_pgo_orientation_covariance import summarize_frame


class OrientationCovarianceAuditTests(unittest.TestCase):
    def test_summary_checks_trace_and_covariance_validity(self) -> None:
        frame = pd.DataFrame({
            "message_id": [10, 11],
            "decoded_from_raw": [True, True],
            "ekf_orientation_covariance_proxy": [0.06, 0.12],
            "roll_variance_rad2": [0.01, 0.02],
            "pitch_variance_rad2": [0.02, 0.04],
            "yaw_variance_rad2": [0.03, 0.06],
            "orientation_trace_rad2": [0.06, 0.12],
            "tilt_sigma_proxy_deg": [7.0, 8.0],
            "yaw_sigma_deg": [9.0, 10.0],
            "yaw_variance_share": [0.5, 0.5],
            "maximum_orientation_offdiagonal": [0.0, 0.0],
            "minimum_orientation_eigenvalue": [0.01, 0.02],
        })
        summary = summarize_frame(frame, trace_tolerance=1e-12)
        self.assertEqual(summary["stable_rows"], 2)
        self.assertEqual(summary["matched_stability_rows"], 2)
        self.assertEqual(summary["decoded_rows"], 2)
        self.assertEqual(summary["missing_decoded_rows"], 0)
        self.assertTrue(summary["saved_trace_reproduction_passed"])
        self.assertEqual(summary["negative_diagonal_rows"], 0)
        self.assertEqual(summary["non_psd_rows"], 0)
        self.assertAlmostEqual(summary["quantiles"]["yaw_variance_share"]["q50"], 0.5)

    def test_summary_rejects_negative_or_mismatched_covariance(self) -> None:
        frame = pd.DataFrame({
            "message_id": [10],
            "decoded_from_raw": [True],
            "ekf_orientation_covariance_proxy": [0.06],
            "roll_variance_rad2": [-0.01],
            "pitch_variance_rad2": [0.02],
            "yaw_variance_rad2": [0.03],
            "orientation_trace_rad2": [0.04],
            "tilt_sigma_proxy_deg": [1.0],
            "yaw_sigma_deg": [1.0],
            "yaw_variance_share": [0.75],
            "maximum_orientation_offdiagonal": [0.0],
            "minimum_orientation_eigenvalue": [-0.01],
        })
        summary = summarize_frame(frame, trace_tolerance=1e-12)
        self.assertFalse(summary["saved_trace_reproduction_passed"])
        self.assertEqual(summary["negative_diagonal_rows"], 1)
        self.assertEqual(summary["non_psd_rows"], 1)

    def test_summary_reports_unmatched_stability_row_as_missing_raw_covariance(self) -> None:
        frame = pd.DataFrame({
            "message_id": pd.array([10, pd.NA], dtype="Int64"),
            "decoded_from_raw": [True, pd.NA],
            "ekf_orientation_covariance_proxy": [0.06, float("nan")],
            "roll_variance_rad2": [0.01, float("nan")],
            "pitch_variance_rad2": [0.02, float("nan")],
            "yaw_variance_rad2": [0.03, float("nan")],
            "orientation_trace_rad2": [0.06, float("nan")],
            "tilt_sigma_proxy_deg": [7.0, float("nan")],
            "yaw_sigma_deg": [9.0, float("nan")],
            "yaw_variance_share": [0.5, float("nan")],
            "maximum_orientation_offdiagonal": [0.0, float("nan")],
            "minimum_orientation_eigenvalue": [0.01, float("nan")],
        })
        summary = summarize_frame(frame, trace_tolerance=1e-12)
        self.assertEqual(summary["stable_rows"], 2)
        self.assertEqual(summary["matched_stability_rows"], 1)
        self.assertEqual(summary["decoded_rows"], 1)
        self.assertEqual(summary["missing_decoded_rows"], 1)
        self.assertTrue(summary["saved_trace_reproduction_passed"])


if __name__ == "__main__":
    unittest.main()
