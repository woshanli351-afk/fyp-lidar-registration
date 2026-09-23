#!/usr/bin/env python3
"""Stage 9A: read-only CU-Multi LiDAR/TF/SC frame-convention audit."""
from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import json
import math
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from prepare_stage1_validation import extract_member, load_gt

RAW = RAW_ROOT
OUT = (UPSTREAM_REPO / 'outputs/cumulti_v1/09_sc_gicp_integration')
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
SAMPLES = 20


def wrap_deg(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    return float(np.degrees(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))))


def urdf_sensor_to_base(robot: str) -> dict:
    archive = RAW / "calib/robot_description.zip"
    with zipfile.ZipFile(archive) as zf:
        root = ET.fromstring(zf.read(f"robot_description/urdf/{robot}.urdf"))
    joints: dict[str, list[dict]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        joints.setdefault(parent, []).append({"child": child, "xyz": xyz, "rpy": rpy, "joint": joint.attrib["name"]})
    # The mounting plate also has camera/IMU children, so find the unique route
    # that terminates at base_link rather than retaining one arbitrary child.
    target = f"{robot}_base_link"
    stack = [(f"{robot}_os_sensor", [])]
    paths = []
    while stack:
        current, chain = stack.pop()
        if current == target:
            paths.append(chain)
            continue
        for item in joints.get(current, []):
            stack.append((item["child"], chain + [item]))
    if len(paths) != 1:
        raise RuntimeError(f"URDF must have one os_sensor-to-base_link path; found {len(paths)}")
    chain = paths[0]
    translation, yaw = np.zeros(3), 0.0
    for item in chain:
        if any(abs(v) > 1e-9 for v in item["rpy"][:2]):
            raise RuntimeError("Non-planar URDF chain requires full SE(3) audit")
        translation += np.asarray(item["xyz"])
        yaw += item["rpy"][2]
    return {"chain_direction": "parent os_sensor -> child base_link", "joints": chain,
            "net_translation_m_approx": translation.tolist(), "net_yaw_deg": float(np.degrees(yaw))}


def frame_header(robot: str) -> str:
    # Reuse the frozen Stage-2/4 raw-message header audits rather than expanding
    # multi-gigabyte LiDAR archives a second time. These reports were produced by
    # the same rosbags deserialization path during the accepted data preparation.
    report = ((UPSTREAM_REPO / 'outputs/cumulti_v1/01_sc_baseline/protocol_validation.txt')
              if robot == "robot1" else
              (UPSTREAM_REPO / 'outputs/cumulti_v1/04_robot1_robot3_sc/protocol_validation.txt'))
    match = re.search(rf"(?:robot{robot[-1]}|Robot{robot[-1]}) PointCloud2 frame_id: (\S+)", report.read_text())
    if match is None:
        raise RuntimeError(f"Cannot recover frozen PointCloud2 frame_id for {robot} from {report}")
    return match.group(1)


def tf_vs_utm(robot: str) -> dict:
    archive = RAW / "main_campus" / robot / f"{robot}_main_campus_gt_rel_poses.zip"
    if not archive.is_file():
        return {"available": False, "reason": f"Missing required recorded /tf archive: {archive}"}
    gt = load_gt(RAW / "main_campus" / robot / f"{robot}_main_campus_gt_utm_poses.csv")
    timestamps = gt.timestamp_ns.to_numpy(np.int64)
    records = []
    with tempfile.TemporaryDirectory(prefix="stage9_tf_") as temp:
        work = Path(temp)
        extract_member(archive, work, "metadata.yaml")
        extract_member(archive, work, ".db3")
        with Reader(work) as reader:
            connection = next(c for c in reader.connections if c.topic == "/tf")
            count = connection.msgcount
            stride = max(1, (count - 1) // (SAMPLES - 1))
            for index, (_, timestamp, raw) in enumerate(reader.messages(connections=[connection])):
                if index % stride and index != count - 1:
                    continue
                message = TYPESTORE.deserialize_cdr(raw, connection.msgtype)
                transform = next((x for x in message.transforms if x.child_frame_id == f"{robot}_os_sensor"), None)
                if transform is None:
                    continue
                nearest = int(np.clip(np.searchsorted(timestamps, timestamp), 1, len(timestamps) - 1))
                if abs(timestamps[nearest - 1] - timestamp) < abs(timestamps[nearest] - timestamp):
                    nearest -= 1
                pose = gt.iloc[nearest]
                tf_yaw = yaw_deg(transform.transform.rotation.x, transform.transform.rotation.y,
                                  transform.transform.rotation.z, transform.transform.rotation.w)
                csv_yaw = yaw_deg(float(pose.qx), float(pose.qy), float(pose.qz), float(pose.qw))
                records.append({"tf_timestamp_ns": int(timestamp), "tf_parent_frame": transform.header.frame_id,
                                "tf_child_frame": transform.child_frame_id, "tf_yaw_deg": tf_yaw,
                                "utm_timestamp_ns": int(pose.timestamp_ns), "sync_abs_ms": abs(int(pose.timestamp_ns) - int(timestamp)) / 1e6,
                                "utm_csv_yaw_deg": csv_yaw,
                                "csv_minus_tf_yaw_deg": float(wrap_deg(csv_yaw - tf_yaw)),
                                "residual_after_documented_180_deg": float(wrap_deg(csv_yaw - tf_yaw - 180.0))})
                if len(records) >= SAMPLES:
                    break
    residual = np.abs(np.asarray([r["residual_after_documented_180_deg"] for r in records]))
    return {"available": True, "records": records, "sample_count": len(records), "max_abs_residual_deg": float(residual.max()),
            "mean_abs_residual_deg": float(residual.mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    robots = {}
    for robot in ("robot1", "robot3"):
        header = frame_header(robot)
        chain = urdf_sensor_to_base(robot)
        tf_check = tf_vs_utm(robot)
        robots[robot] = {"pointcloud_topic": f"{robot}/ouster/points", "pointcloud_frame_id": header,
                         "urdf_sensor_to_base": chain, "tf_vs_utm_orientation_check": tf_check}
    resolved = all(value["pointcloud_frame_id"] == f"{robot}_os_sensor" and
                   abs(value["urdf_sensor_to_base"]["net_yaw_deg"]) < 1e-6 and
                   value["tf_vs_utm_orientation_check"].get("available", False) and
                   value["tf_vs_utm_orientation_check"]["max_abs_residual_deg"] < 1e-3
                   for robot, value in robots.items())
    audit = {
        "status": "RESOLVED" if resolved else "UNRESOLVED",
        "transform_direction": "T_query_from_candidate: p_query = T_query_from_candidate * p_candidate",
        "gicp_source_target": "Open3D source=candidate cloud, target=query cloud; returned transform is T_query_from_candidate",
        "cloud_storage_frame": "PointCloud2 header os_sensor; cached XYZI is stored unchanged in that frame",
        "local_map_frame": "relative ROS /tf parent frame 'world', with child robot*_os_sensor",
        "gt_utm_frame": "CSV positions are UTM coordinates; its quaternion yaw is Rz(+180 deg) offset from recorded world->os_sensor TF",
        "previous_180_degree_discrepancy_cause": "recorded UTM CSV quaternion convention, not sensor mounting, SC tuning, or candidate inversion",
        "sc_shift_convention": {
            "source": "run_stage2_sc_baseline.py::scores_for_query",
            "formula": "score(s) = sum_k q[k] dot candidate[k-s]",
            "candidate_query_order": "query descriptor first, candidate/database descriptor second",
            "best_shift_to_gicp_yaw": "signed_shift = best_shift if best_shift <= 30 else best_shift - 60; yaw_SC = wrap(+signed_shift * 6 deg + 0 deg)",
            "reason": "a positive shift moves a candidate sector k-s into query sector k, therefore Rz(+shift*6) maps candidate os_sensor coordinates to query os_sensor coordinates",
        },
        "robots": robots,
        "gt_policy": "TF/UTM comparison is post-derivation validation only; GT is never read by retrieval or GICP initialization/quality.",
    }
    (OUT / "frame_convention_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not resolved:
        raise SystemExit("Stage 9A frame convention is unresolved; stopping before PGO-ready edges")


if __name__ == "__main__":
    main()
