#!/usr/bin/env python3
"""Audit raw EKF orientation covariance for a frozen PGO stable segment.

The Stage 12C stability table stores only the trace of the orientation block.
This tool returns to the source nav_msgs/Odometry messages and separates the
fixed-axis roll, pitch and yaw covariance terms. It never reads GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.typesys import Stores, get_typestore


QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(series: pd.Series) -> dict[str, float] | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    return {f"q{int(q * 100):02d}": float(values.quantile(q)) for q in QUANTILES}


def summarize_frame(frame: pd.DataFrame, trace_tolerance: float) -> dict[str, object]:
    saved = pd.to_numeric(frame.ekf_orientation_covariance_proxy, errors="coerce")
    measured = pd.to_numeric(frame.orientation_trace_rad2, errors="coerce")
    trace_delta = np.abs(saved - measured)
    finite = np.isfinite(measured)
    decoded = frame.decoded_from_raw.astype("boolean").fillna(False)
    finite_delta = trace_delta[np.isfinite(trace_delta)]
    maximum_trace_delta = float(finite_delta.max()) if not finite_delta.empty else None
    summary = {
        "stable_rows": int(len(frame)),
        "matched_stability_rows": int(frame.message_id.notna().sum()),
        "decoded_rows": int(decoded.sum()),
        "decoded_coverage_fraction": float(decoded.mean()),
        "unique_message_ids": int(frame.message_id.dropna().nunique()),
        "missing_decoded_rows": int((~decoded).sum()),
        "finite_orientation_rows": int(finite.sum()),
        "zero_trace_rows": int((measured[finite] == 0).sum()),
        "negative_diagonal_rows": int((frame[[
            "roll_variance_rad2", "pitch_variance_rad2", "yaw_variance_rad2"
        ]] < 0).any(axis=1).sum()),
        "non_psd_rows": int((frame.minimum_orientation_eigenvalue < -1e-12).sum()),
        "maximum_saved_trace_delta": maximum_trace_delta,
        "saved_trace_reproduction_passed": bool(
            maximum_trace_delta is not None
            and maximum_trace_delta <= trace_tolerance
            and int(decoded.sum()) == int(frame.message_id.notna().sum())
        ),
        "quantiles": {
            column: quantiles(frame[column])
            for column in (
                "roll_variance_rad2",
                "pitch_variance_rad2",
                "yaw_variance_rad2",
                "orientation_trace_rad2",
                "tilt_sigma_proxy_deg",
                "yaw_sigma_deg",
                "yaw_variance_share",
                "maximum_orientation_offdiagonal",
                "minimum_orientation_eigenvalue",
            )
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--stable-start", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stability-csv", type=Path)
    source.add_argument(
        "--keyframes-csv",
        type=Path,
        help=(
            "Reconstruct the minimal GT-free Stage 12C EKF match table from "
            "robot_id/keyframe_id/lidar_timestamp_ns only."
        ),
    )
    parser.add_argument("--imu-gps-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def load_stable_rows(path: Path, robot: str, stable_start: int) -> pd.DataFrame:
    required = {
        "robot_id",
        "keyframe_id",
        "ekf_match_status",
        "ekf_message_sequence_index",
        "ekf_orientation_covariance_proxy",
    }
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"stability CSV missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=sorted(required))
    frame = frame[frame.robot_id.eq(robot) & frame.keyframe_id.ge(stable_start)].copy()
    if frame.empty:
        raise ValueError("stable selection is empty")
    if frame.keyframe_id.duplicated().any():
        raise ValueError("stable selection has duplicate keyframe IDs")
    matched = frame.ekf_match_status.eq("MATCHED")
    if frame.loc[matched, "ekf_message_sequence_index"].isna().any():
        raise ValueError("matched EKF rows are missing message IDs")
    frame["message_id"] = pd.array(frame.ekf_message_sequence_index, dtype="Int64")
    return frame


def load_keyframe_timestamps(path: Path, robot: str, stable_start: int) -> pd.DataFrame:
    """Load only GT-free identity/timestamp columns from processed keyframes."""
    required = {"robot_id", "keyframe_id", "lidar_timestamp_ns"}
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"keyframes CSV missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=sorted(required))
    frame = frame[frame.robot_id.eq(robot) & frame.keyframe_id.ge(stable_start)].copy()
    frame = frame.sort_values("keyframe_id").reset_index(drop=True)
    if frame.empty:
        raise ValueError("stable keyframe selection is empty")
    if frame.keyframe_id.duplicated().any():
        raise ValueError("stable keyframe selection has duplicate keyframe IDs")
    timestamps = pd.to_numeric(frame.lidar_timestamp_ns, errors="raise").astype("int64")
    if not np.all(np.diff(timestamps.to_numpy()) > 0):
        raise ValueError("stable keyframe timestamps are not strictly increasing")
    frame["lidar_timestamp_ns"] = timestamps
    return frame


def reconstruct_stability_rows(
    archive: Path,
    topic: str,
    keyframes: pd.DataFrame,
    tolerance_s: float = 0.05,
) -> pd.DataFrame:
    """Reproduce the frozen Stage 12C nearest-EKF timestamp association."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rows: list[dict[str, float | int | str | None]] = []
    tolerance_ns = int(round(tolerance_s * 1e9))
    with zipfile.ZipFile(archive) as zipped:
        db_members = [name for name in zipped.namelist() if name.endswith(".db3")]
        if len(db_members) != 1:
            raise ValueError(f"expected one DB3 member, found {len(db_members)}")
        with tempfile.TemporaryDirectory(prefix="gravity_stability_") as directory:
            database = Path(directory) / "source.db3"
            with zipped.open(db_members[0]) as source, database.open("wb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            try:
                topic_row = connection.execute(
                    "SELECT id,type FROM topics WHERE name=?", (topic,)
                ).fetchone()
                if topic_row is None:
                    raise ValueError(f"missing topic: {topic}")
                topic_id, message_type = topic_row
                messages = iter(connection.execute(
                    "SELECT id,timestamp,data FROM messages "
                    "WHERE topic_id=? ORDER BY timestamp,id",
                    (topic_id,),
                ))
                before = None
                after = next(messages, None)
                for target_ns in keyframes.lidar_timestamp_ns.astype("int64"):
                    target_ns = int(target_ns)
                    while after is not None and int(after[1]) < target_ns:
                        before, after = after, next(messages, None)
                    choices = [value for value in (before, after) if value is not None]
                    best = (
                        min(choices, key=lambda value: abs(int(value[1]) - target_ns))
                        if choices else None
                    )
                    if best is None or abs(int(best[1]) - target_ns) > tolerance_ns:
                        rows.append({
                            "ekf_match_status": "OUT_OF_WINDOW",
                            "ekf_message_sequence_index": None,
                            "ekf_orientation_covariance_proxy": None,
                            "ekf_bag_timestamp_ns": int(best[1]) if best is not None else None,
                            "ekf_header_timestamp_ns": None,
                            "ekf_bag_match_error_s": (
                                abs(int(best[1]) - target_ns) / 1e9
                                if best is not None else None
                            ),
                            "ekf_header_match_error_s": None,
                        })
                        continue
                    message = typestore.deserialize_cdr(best[2], message_type)
                    header_ns = (
                        int(message.header.stamp.sec) * 1_000_000_000
                        + int(message.header.stamp.nanosec)
                    )
                    if abs(header_ns - target_ns) > tolerance_ns:
                        rows.append({
                            "ekf_match_status": "HEADER_OUT_OF_WINDOW",
                            "ekf_message_sequence_index": None,
                            "ekf_orientation_covariance_proxy": None,
                            "ekf_bag_timestamp_ns": int(best[1]),
                            "ekf_header_timestamp_ns": header_ns,
                            "ekf_bag_match_error_s": abs(int(best[1]) - target_ns) / 1e9,
                            "ekf_header_match_error_s": abs(header_ns - target_ns) / 1e9,
                        })
                        continue
                    covariance = np.asarray(
                        message.pose.covariance, dtype=float
                    ).reshape(6, 6)
                    rows.append({
                        "ekf_match_status": "MATCHED",
                        "ekf_message_sequence_index": int(best[0]),
                        "ekf_orientation_covariance_proxy": float(
                            np.trace(covariance[3:, 3:])
                        ),
                        "ekf_bag_timestamp_ns": int(best[1]),
                        "ekf_header_timestamp_ns": header_ns,
                        "ekf_bag_match_error_s": abs(int(best[1]) - target_ns) / 1e9,
                        "ekf_header_match_error_s": abs(header_ns - target_ns) / 1e9,
                    })
            finally:
                connection.close()
    stable = pd.concat(
        [keyframes[["robot_id", "keyframe_id"]].reset_index(drop=True), pd.DataFrame(rows)],
        axis=1,
    )
    stable["message_id"] = pd.array(
        stable.ekf_message_sequence_index, dtype="Int64"
    )
    return stable


def decode_orientation_blocks(
    archive: Path,
    topic: str,
    message_ids: list[int],
) -> pd.DataFrame:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rows: list[dict[str, float | int]] = []
    with zipfile.ZipFile(archive) as zipped:
        db_members = [name for name in zipped.namelist() if name.endswith(".db3")]
        if len(db_members) != 1:
            raise ValueError(f"expected one DB3 member, found {len(db_members)}")
        with tempfile.TemporaryDirectory(prefix="gravity_covariance_") as directory:
            database = Path(directory) / "source.db3"
            with zipped.open(db_members[0]) as source, database.open("wb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            try:
                topic_row = connection.execute(
                    "SELECT id,type FROM topics WHERE name=?", (topic,)
                ).fetchone()
                if topic_row is None:
                    raise ValueError(f"missing topic: {topic}")
                topic_id, message_type = topic_row
                for start in range(0, len(message_ids), 800):
                    chunk = message_ids[start:start + 800]
                    placeholders = ",".join("?" for _ in chunk)
                    query = (
                        f"SELECT id,data FROM messages WHERE topic_id=? "
                        f"AND id IN ({placeholders})"
                    )
                    for message_id, data in connection.execute(query, [topic_id, *chunk]):
                        message = typestore.deserialize_cdr(data, message_type)
                        covariance = np.asarray(message.pose.covariance, dtype=float).reshape(6, 6)
                        orientation = covariance[3:, 3:]
                        diagonal = np.diag(orientation)
                        symmetric = (orientation + orientation.T) / 2
                        trace = float(np.trace(orientation))
                        tilt_variance = float((diagonal[0] + diagonal[1]) / 2)
                        rows.append({
                            "message_id": int(message_id),
                            "decoded_from_raw": True,
                            "roll_variance_rad2": float(diagonal[0]),
                            "pitch_variance_rad2": float(diagonal[1]),
                            "yaw_variance_rad2": float(diagonal[2]),
                            "orientation_trace_rad2": trace,
                            "tilt_sigma_proxy_deg": (
                                float(np.degrees(np.sqrt(tilt_variance)))
                                if tilt_variance >= 0 else np.nan
                            ),
                            "yaw_sigma_deg": (
                                float(np.degrees(np.sqrt(diagonal[2])))
                                if diagonal[2] >= 0 else np.nan
                            ),
                            "yaw_variance_share": float(diagonal[2] / trace) if trace > 0 else np.nan,
                            "maximum_orientation_offdiagonal": float(np.max(np.abs(
                                orientation - np.diag(diagonal)
                            ))),
                            "minimum_orientation_eigenvalue": float(np.linalg.eigvalsh(symmetric).min()),
                        })
            finally:
                connection.close()
    decoded = pd.DataFrame(rows)
    if decoded.message_id.duplicated().any():
        raise ValueError("decoded duplicate EKF message IDs")
    found = set(decoded.message_id.astype(int))
    expected = set(message_ids)
    if found != expected:
        raise ValueError(f"message coverage mismatch: missing={sorted(expected - found)[:10]}")
    return decoded


def run(args: argparse.Namespace) -> None:
    if args.stability_csv is not None:
        stable = load_stable_rows(args.stability_csv, args.robot, args.stable_start)
        source_mode = "stage12c_stability_csv"
        source_path = args.stability_csv
    else:
        keyframes = load_keyframe_timestamps(
            args.keyframes_csv, args.robot, args.stable_start
        )
        stable = reconstruct_stability_rows(
            args.imu_gps_archive,
            f"{args.robot}/ekf/odometry_map",
            keyframes,
        )
        source_mode = "reconstructed_from_gtfree_keyframe_timestamps"
        source_path = args.keyframes_csv
    message_ids = stable.message_id.dropna().astype(int).unique().tolist()
    decoded = decode_orientation_blocks(
        args.imu_gps_archive,
        f"{args.robot}/ekf/odometry_map",
        message_ids,
    )
    joined = stable.merge(decoded, on="message_id", how="left", validate="many_to_one")
    summary = summarize_frame(joined, args.trace_tolerance)
    summary.update({
        "robot": args.robot,
        "stable_start_keyframe": args.stable_start,
        "source_message_type": "nav_msgs/msg/Odometry",
        "orientation_basis": "fixed-axis rotation X/Y/Z covariance block from PoseWithCovariance",
        "source_mode": source_mode,
        "trace_reference": (
            "stage12c_saved_orientation_covariance_proxy"
            if source_mode == "stage12c_stability_csv"
            else "reconstructed_raw_message_orientation_covariance_trace"
        ),
        "trace_reproduction_independent_of_raw_decode": (
            source_mode == "stage12c_stability_csv"
        ),
        "source_timestamp_input_filename": source_path.name,
        "source_timestamp_input_sha256": sha256(source_path),
        "archive_filename": args.imu_gps_archive.name,
        "archive_size_bytes": args.imu_gps_archive.stat().st_size,
        "archive_sha256": sha256(args.imu_gps_archive),
        "GT_used": False,
        "decision": (
            "FULL_TILT_COVARIANCE_AVAILABLE"
            if summary["saved_trace_reproduction_passed"]
            and summary["missing_decoded_rows"] == 0
            and summary["negative_diagonal_rows"] == 0
            and summary["non_psd_rows"] == 0
            else "PARTIAL_TILT_COVARIANCE_AVAILABLE"
            if summary["saved_trace_reproduction_passed"]
            and summary["decoded_coverage_fraction"] >= 0.99
            and summary["negative_diagonal_rows"] == 0
            and summary["non_psd_rows"] == 0
            else "COVARIANCE_NOT_READY"
        ),
        "missing_covariance_policy": "omit gravity factor for unmatched node",
    })
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.keyframes_csv is not None:
        stable.drop(columns="message_id").to_csv(
            output / f"{args.robot}_reconstructed_stability_input.csv", index=False
        )
    joined.to_csv(output / f"{args.robot}_orientation_covariance_by_keyframe.csv", index=False)
    (output / f"{args.robot}_orientation_covariance_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
