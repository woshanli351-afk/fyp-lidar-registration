#!/usr/bin/env python3
"""Create the CU-Multi Main Campus 100-frame validation subset.

This tool is deliberately read-only with respect to ``raw_root``.  CU-Multi
stores ROS 2 SQLite bags as single members inside ZIP files.  SQLite requires
random access, so the required bag member is expanded temporarily under
``processed_root/.work`` for one robot at a time, decoded, then removed.  Only
the requested 100 LiDAR frames and their matched RGB images persist.
"""

from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import argparse
import csv
import json
import math
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
POINT_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


@dataclass
class BagInfo:
    topic: str
    msgtype: str
    count: int
    start_ns: int
    duration_ns: int


def extract_member(zip_path: Path, destination: Path, member_suffix: str) -> Path:
    """Expand exactly one archive member; never write beneath the raw archive."""
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [name for name in zf.namelist() if name.endswith(member_suffix)]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {member_suffix!r} in {zip_path}, found {candidates}")
        member = candidates[0]
        out = destination / Path(member).name
        destination.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as source, out.open("wb") as target:
            shutil.copyfileobj(source, target, length=32 * 1024 * 1024)
    return out


def read_metadata(zip_path: Path) -> dict[str, Any]:
    import yaml
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("metadata.yaml"))
        return yaml.safe_load(zf.read(name))["rosbag2_bagfile_information"]


def topic_info(zip_path: Path) -> list[BagInfo]:
    meta = read_metadata(zip_path)
    return [BagInfo(
        item["topic_metadata"]["name"], item["topic_metadata"]["type"],
        int(item["message_count"]), int(meta["starting_time"]["nanoseconds_since_epoch"]),
        int(meta["duration"]["nanoseconds"]),
    ) for item in meta["topics_with_message_count"]]


def messages_index(db3: Path, topic: str) -> tuple[list[tuple[int, Any]], Any]:
    with Reader(db3.parent) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if len(connections) != 1:
            raise RuntimeError(f"Expected one connection for {topic}, got {len(connections)}")
        connection = connections[0]
        rows = [(timestamp, raw) for _, timestamp, raw in reader.messages(connections=connections)]
    return rows, connection


def nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(target_ns, source_ns, side="left")
    right = np.clip(right, 0, len(target_ns) - 1)
    left = np.clip(right - 1, 0, len(target_ns) - 1)
    choose_left = np.abs(source_ns - target_ns[left]) <= np.abs(source_ns - target_ns[right])
    index = np.where(choose_left, left, right)
    return index, target_ns[index] - source_ns


def pointcloud_xyzi(msg: Any) -> tuple[np.ndarray, list[str], str]:
    fields = {field.name: field for field in msg.fields}
    missing = [name for name in ("x", "y", "z") if name not in fields]
    if missing:
        raise RuntimeError(f"PointCloud2 missing fields: {missing}; has {sorted(fields)}")
    intensity_name = "intensity" if "intensity" in fields else ("reflectivity" if "reflectivity" in fields else "")
    names = ["x", "y", "z"] + ([intensity_name] if intensity_name else [])
    max_item = max(fields[name].offset + np.dtype(POINT_TYPES[fields[name].datatype]).itemsize for name in names)
    if msg.point_step < max_item:
        raise RuntimeError(f"Invalid PointCloud2 point_step {msg.point_step}")
    dtype = np.dtype({"names": names, "formats": [POINT_TYPES[fields[name].datatype] for name in names],
                      "offsets": [fields[name].offset for name in names], "itemsize": msg.point_step})
    data = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    output = np.empty((len(data), 4), dtype=np.float32)
    for i, name in enumerate(("x", "y", "z")):
        output[:, i] = data[name]
    output[:, 3] = data[intensity_name] if intensity_name else 0.0
    good = np.isfinite(output[:, :3]).all(axis=1)
    return output[good], sorted(fields), intensity_name or "not present (zero-filled)"


def image_bgr(msg: Any) -> np.ndarray:
    encoding = msg.encoding.lower()
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        image = raw.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image
    if encoding in ("rgba8", "bgra8"):
        image = raw.reshape(msg.height, msg.step)[:, : msg.width * 4].reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        image = raw.reshape(msg.height, msg.step)[:, :msg.width]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported RGB encoding {msg.encoding!r}")


def plot_trajectory(poses: pd.DataFrame, robot: str, output: Path, others: pd.DataFrame | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=140)
    ax.plot(poses.x, poses.y, lw=1.0, label=robot)
    if others is not None:
        ax.plot(others.x, others.y, lw=1.0, label="robot2")
        ax.legend()
    ax.set_xlabel("UTM easting x (m)")
    ax.set_ylabel("UTM northing y (m)")
    ax.set_title("CU-Multi Main Campus GT trajectory" if others is not None else f"CU-Multi {robot} GT trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


def plot_cloud(points: np.ndarray, robot: str, frame_id: int, output: Path) -> None:
    sample = points[::max(1, len(points) // 30000)]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    scatter = ax.scatter(sample[:, 0], sample[:, 1], c=sample[:, 2], s=0.25, cmap="viridis")
    ax.set_title(f"{robot} LiDAR frame {frame_id:010d}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal")
    fig.colorbar(scatter, ax=ax, label="z (m)")
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


def load_gt(csv_path: Path) -> pd.DataFrame:
    fields = ["timestamp_s", "x", "y", "z", "qx", "qy", "qz", "qw"]
    df = pd.read_csv(csv_path, comment="#", header=None, names=fields)
    if len(df.columns) != 8 or df.empty:
        raise RuntimeError(f"Unexpected GT CSV schema in {csv_path}")
    df["timestamp_ns"] = np.rint(df.timestamp_s.to_numpy() * 1e9).astype("int64")
    qx, qy, qz, qw = (df[name].to_numpy() for name in ("qx", "qy", "qz", "qw"))
    df["yaw_rad"] = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    df["yaw_deg"] = np.degrees(df.yaw_rad)
    return df


def stats(values_ns: np.ndarray) -> dict[str, float]:
    values_ms = np.abs(values_ns.astype(np.float64)) / 1e6
    return {"mean_abs_ms": float(values_ms.mean()), "median_abs_ms": float(np.median(values_ms)),
            "p95_abs_ms": float(np.percentile(values_ms, 95)), "max_abs_ms": float(values_ms.max())}


def process_robot(raw_root: Path, processed_root: Path, robot: str, keyframes: int, manifest: dict[str, Any]) -> pd.DataFrame:
    raw_robot = raw_root / "main_campus" / robot
    out_robot = processed_root / robot
    lidar_dir, rgb_dir = out_robot / "lidar", out_robot / "rgb"
    for path in (lidar_dir, rgb_dir): path.mkdir(parents=True, exist_ok=True)
    work = processed_root / ".work" / robot
    work.mkdir(parents=True, exist_ok=True)
    lidar_zip = raw_robot / f"{robot}_main_campus_lidar.zip"
    rgb_zip = raw_robot / f"{robot}_main_campus_camera_rgb.zip"
    gt_path = raw_robot / f"{robot}_main_campus_gt_utm_poses.csv"
    lidar_topic = f"{robot}/ouster/points"
    rgb_topic = f"{robot}/camera/color/image_raw"
    # Temporarily extract LiDAR DB3; it is required because SQLite cannot read a ZIP stream.
    lidar_work = work / "lidar"
    extract_member(lidar_zip, lidar_work, "metadata.yaml")
    lidar_db3 = extract_member(lidar_zip, lidar_work, ".db3")
    lidar_rows, lidar_connection = messages_index(lidar_db3, lidar_topic)
    if len(lidar_rows) < keyframes: raise RuntimeError(f"Only {len(lidar_rows)} LiDAR messages")
    selected_indices = np.unique(np.linspace(0, len(lidar_rows) - 1, keyframes, dtype=int))
    if len(selected_indices) != keyframes: raise RuntimeError("Keyframe selection did not yield exactly requested count")
    selected = [(int(index), lidar_rows[int(index)][0], lidar_rows[int(index)][1]) for index in selected_indices]
    gt = load_gt(gt_path)
    gt_index, gt_delta_ns = nearest_indices(np.array([row[1] for row in selected], dtype=np.int64), gt.timestamp_ns.to_numpy())
    point_fields: list[str] | None = None; intensity_source = ""; cloud_counts: list[int] = []
    sample_ids = {0, keyframes // 2, keyframes - 1}
    for seq, (source_index, timestamp_ns, raw) in enumerate(selected):
        msg = TYPESTORE.deserialize_cdr(raw, lidar_connection.msgtype)
        cloud, fields, intensity = pointcloud_xyzi(msg)
        point_fields, intensity_source = fields, intensity
        cloud_counts.append(len(cloud))
        filename = f"{seq:06d}_lidar_{timestamp_ns}.npy"
        np.save(lidar_dir / filename, cloud)
        if seq in sample_ids:
            plot_cloud(cloud, robot, source_index, processed_root / "00_validation" / f"{robot}_lidar_{seq:03d}.png")
    del lidar_rows
    lidar_db3.unlink(); (lidar_work / "metadata.yaml").unlink(); lidar_work.rmdir()
    # Extract RGB DB3 only after LiDAR selection; decode only the matched 100 images.
    rgb_work = work / "rgb"
    extract_member(rgb_zip, rgb_work, "metadata.yaml")
    rgb_db3 = extract_member(rgb_zip, rgb_work, ".db3")
    rgb_rows, rgb_connection = messages_index(rgb_db3, rgb_topic)
    rgb_ts = np.array([row[0] for row in rgb_rows], dtype=np.int64)
    rgb_index, rgb_delta_ns = nearest_indices(np.array([row[1] for row in selected], dtype=np.int64), rgb_ts)
    rgb_filenames: list[str] = []
    for seq, index in enumerate(rgb_index):
        timestamp_ns, raw = rgb_rows[int(index)]
        image = image_bgr(TYPESTORE.deserialize_cdr(raw, rgb_connection.msgtype))
        filename = f"{seq:06d}_rgb_{timestamp_ns}.png"
        if not cv2.imwrite(str(rgb_dir / filename), image): raise RuntimeError(f"Unable to write {filename}")
        rgb_filenames.append(filename)
        if seq in sample_ids: shutil.copy2(rgb_dir / filename, processed_root / "00_validation" / f"{robot}_rgb_{seq:03d}.png")
    del rgb_rows
    rgb_db3.unlink(); (rgb_work / "metadata.yaml").unlink(); rgb_work.rmdir(); work.rmdir()
    rows: list[dict[str, Any]] = []
    for seq, (source_index, timestamp_ns, _) in enumerate(selected):
        pose = gt.iloc[int(gt_index[seq])]
        rows.append({"keyframe_id": seq, "source_lidar_message_index": source_index, "lidar_timestamp_ns": timestamp_ns,
                     "lidar_file": f"{seq:06d}_lidar_{timestamp_ns}.npy", "rgb_message_index": int(rgb_index[seq]),
                     "rgb_timestamp_ns": int(rgb_ts[rgb_index[seq]]), "rgb_file": rgb_filenames[seq],
                     "rgb_minus_lidar_ns": int(rgb_delta_ns[seq]), "gt_pose_index": int(gt_index[seq]),
                     "gt_timestamp_ns": int(pose.timestamp_ns), "gt_minus_lidar_ns": int(gt_delta_ns[seq]),
                     "x": float(pose.x), "y": float(pose.y), "z": float(pose.z), "qx": float(pose.qx), "qy": float(pose.qy),
                     "qz": float(pose.qz), "qw": float(pose.qw), "yaw_rad": float(pose.yaw_rad), "yaw_deg": float(pose.yaw_deg)})
    sync = pd.DataFrame(rows)
    sync.to_csv(out_robot / "sync_index.csv", index=False)
    sync.to_csv(processed_root / "00_validation" / f"{robot}_sync_index.csv", index=False)
    sync[["keyframe_id", "source_lidar_message_index", "lidar_timestamp_ns", "lidar_file"]].to_csv(out_robot / "timestamps.csv", index=False)
    sync[["gt_pose_index", "gt_timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw", "yaw_rad", "yaw_deg"]].to_csv(out_robot / "poses.csv", index=False)
    lidar_topic_count = next(item.count for item in topic_info(lidar_zip) if item.topic == lidar_topic)
    manifest[robot] = {"lidar_topic": lidar_topic, "rgb_topic": rgb_topic, "lidar_frames_total": lidar_topic_count,
                       "rgb_frames_total": len(rgb_ts), "extracted_keyframes": len(sync), "point_fields": point_fields,
                       "intensity_source": intensity_source, "points_per_frame": {"min": int(min(cloud_counts)), "median": float(np.median(cloud_counts)), "max": int(max(cloud_counts))},
                       "rgb_sync": stats(rgb_delta_ns), "gt_sync": stats(gt_delta_ns),
                       "lidar_metadata_topics": [asdict(x) for x in topic_info(lidar_zip)], "rgb_metadata_topics": [asdict(x) for x in topic_info(rgb_zip)]}
    return gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--keyframes", type=int, default=100)
    args = parser.parse_args()
    if args.keyframes != 100: raise SystemExit("This Stage 1 script is restricted to exactly 100 keyframes per robot.")
    validation = args.processed_root / "00_validation"; validation.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"stage": "CU-Multi Main Campus Stage 1", "raw_root": str(args.raw_root), "processed_root": str(args.processed_root),
                                "keyframes_per_robot": 100, "raw_modified": False, "notes": "Raw archives are read only; temporary DB3 files are deleted after decoding."}
    trajectories = {}
    for robot in ("robot1", "robot2"):
        trajectories[robot] = process_robot(args.raw_root, args.processed_root, robot, args.keyframes, manifest)
        plot_trajectory(trajectories[robot], robot, validation / f"{robot}_trajectory.png")
    plot_trajectory(trajectories["robot1"], "robot1", validation / "combined_trajectory.png", trajectories["robot2"])
    calib_zip = args.raw_root / "calib" / "robot_description.zip"
    summary = ["CU-Multi calibration summary", "", "Source: calib/robot_description.zip, robot1.urdf and robot2.urdf.",
               "Both robots define base_link, chassis, mounting_plate, os_sensor, imu_link, GNSS antennas, and camera_link.",
               "Fixed mounting_plate -> camera_link: xyz=(0.2, -0.02, -0.023), rpy=(0,0,0).",
               "Fixed os_sensor -> ouster_riser_plate: xyz=(0,0,-0.05), rpy=(0,0,0).",
               "Fixed ouster_riser_plate -> mounting_plate: xyz=(-0.02286,0,-0.0175), rpy=(0,0,0).",
               "Fixed mounting_plate -> chassis: xyz=(-0.35,0,-0.08), rpy=(0,0,0).",
               "URDF link visual origins are not substituted for sensor measurement coordinate frames.",
               "GT CSV positions are interpreted as common UTM easting/northing/elevation because both robots overlap in the same numerical coordinate range; this is validated geometrically but should be confirmed against dataset documentation before metric cross-robot alignment."]
    (validation / "calibration_summary.txt").write_text("\n".join(summary) + "\n")
    report = ["CU-Multi Main Campus Stage 1 validation summary", "", "Raw archive format: ZIP containers holding ROS2 rosbag2 SQLite (.db3) plus metadata.yaml; no archive was modified.",
              "LiDAR: sensor_msgs/msg/PointCloud2 on robot{1,2}/ouster/points (about 20 Hz). RGB: sensor_msgs/msg/Image on robot{1,2}/camera/color/image_raw (about 10 Hz).",
              "GT CSV fields: timestamp, x, y, z, qx, qy, qz, qw; yaw is analysis-only and computed with standard quaternion-to-Z yaw atan2 formula.",
              "Coordinate assumption: x/y/z are UTM easting/northing/elevation; frames remain in raw LiDAR sensor coordinates and are not transformed using GT.",
              "Robot1/robot2 common-world status: likely common UTM frame, based on shared coordinate range and simultaneous timestamps; dataset documentation confirmation remains an explicit caveat.", ""]
    for robot in ("robot1", "robot2"):
        item = manifest[robot]
        report += [f"{robot}: extracted LiDAR frames={item['extracted_keyframes']}; source LiDAR frames={item['lidar_frames_total']}; source RGB frames={item['rgb_frames_total']}.",
                   f"  Point fields={item['point_fields']}; intensity={item['intensity_source']}.",
                   f"  RGB nearest-frame abs error ms: {item['rgb_sync']}", f"  GT nearest-pose abs error ms: {item['gt_sync']}"]
    report += ["", "Unresolved issues: camera intrinsics are contained in CameraInfo messages but are not exported in this minimal adapter; exact UTM zone/vertical datum and measurement-frame origin need external dataset documentation confirmation."]
    (validation / "validation_summary.txt").write_text("\n".join(report) + "\n")
    (validation / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
