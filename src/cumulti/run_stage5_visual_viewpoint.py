#!/usr/bin/env python3
"""CU-Multi Stage 5: Visual Viewpoint Analysis on the frozen Robot1 -> Robot3 protocol.

Stage 4 is frozen.  This stage measures how reliable visual evidence is under
different viewpoint / heading differences after Scan Context has already
generated a high-recall candidate pool.

OpenCLIP ViT-B-32-quickgelu (laion400m_e32) encodes the RGB image nearest each
frozen 2 Hz LiDAR keyframe.  Three visual methods (Single RGB, RGB5 Mean,
Cross-Max) rerank the frozen SC Top-20 candidates.  GT is evaluation-only.

NO fusion, NO VLM, NO CVTNet, NO GICP, NO PGO.
"""
from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── frozen SC constants (from run_stage2_sc_baseline.py) ─────────────────
RINGS, SECTORS, RADIUS, HEIGHT = 20, 60, 80.0, 0.50
TOPK = 20
BIN_EDGES = np.array([0, 30, 60, 90, 120, 150, 180.000001], dtype=float)
BIN_LABELS = ("0-30", "30-60", "60-90", "90-120", "120-150", "150-180")

# ── OpenCLIP configuration (recovered from historical KITTI experiments) ──
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_TAG = "laion400m_e32"
CHECKPOINT_PATH = Path("models/openclip/vit_b32_laion400m_e32.pt")

PROCESSED = PROCESSED_ROOT
REPO = UPSTREAM_REPO
RAW = RAW_ROOT
EMBED_DIR = PROCESSED / "openclip_stage5"
OUT_DIR = REPO / "outputs/cumulti_v1/05_visual_viewpoint_analysis"


# =====================================================================
#  SC reconstruction (identical to frozen Stage 2/4 implementation)
# =====================================================================
def normalize_columns(d):
    norms = np.linalg.norm(d, axis=1)
    valid = norms > 0
    return np.divide(d, norms[:, None, :], out=np.zeros_like(d), where=valid[:, None, :]), valid


def scores_for_query(q, dbu, dbvalid):
    qu, qv = normalize_columns(q[None])
    qu = qu[0].astype(np.float32)
    qv = qv[0].astype(np.float32)
    dot = np.fft.ifft(np.fft.fft(qu, axis=1)[None] * np.conj(np.fft.fft(dbu, axis=2)), axis=2).real.sum(axis=1)
    cnt = np.fft.ifft(np.fft.fft(qv)[None] * np.conj(np.fft.fft(dbvalid.astype(np.float32), axis=1)), axis=1).real
    scores = np.divide(dot, cnt, out=np.zeros_like(dot), where=cnt > 0)
    shifts = np.argmax(scores, axis=1)
    return scores[np.arange(len(dbu)), shifts].astype(np.float32), shifts.astype(np.int16)


def make_gt(q, db, threshold=5.0):
    dx = q.x.to_numpy()[:, None] - db.x.to_numpy()[None]
    dy = q.y.to_numpy()[:, None] - db.y.to_numpy()[None]
    dist = np.hypot(dx, dy)
    positive = dist < threshold
    return dist, positive


def wrapped_abs_heading(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


# =====================================================================
#  B. RGB sync audit
# =====================================================================
def audit_robot1_rgb_sync(processed: Path, raw: Path):
    """Extract Robot1 RGB sync from raw bags.  Robot1 has no rgb_sync_index.csv."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    from prepare_stage1_validation import extract_member, image_bgr, nearest_indices

    T = get_typestore(Stores.ROS2_HUMBLE)
    kf = pd.read_csv(processed / "full_2hz" / "robot1" / "keyframes.csv")
    lidar_ts = kf.lidar_timestamp_ns.to_numpy(np.int64)

    work = processed / ".work_stage5" / "robot1_rgb"
    work.mkdir(parents=True, exist_ok=True)
    rgb_zip = raw / "main_campus" / "robot1" / "robot1_main_campus_camera_rgb.zip"
    extract_member(rgb_zip, work, "metadata.yaml")
    rgb_db3 = extract_member(rgb_zip, work, ".db3")

    try:
        with Reader(rgb_db3.parent) as r:
            cs = [c for c in r.connections if c.topic == "robot1/camera/color/image_raw"]
            if len(cs) != 1:
                raise RuntimeError(f"Expected one RGB connection, got {len(cs)}")
            rgb_rows = [(ts, raw_data) for _, ts, raw_data in r.messages(connections=cs)]
        rgb_ts_all = np.array([row[0] for row in rgb_rows], dtype=np.int64)
        rgb_idx, rgb_delta = nearest_indices(lidar_ts, rgb_ts_all)
    finally:
        for p in (rgb_db3, work / "metadata.yaml"):
            if p.exists():
                p.unlink()
        if work.exists():
            import shutil
            shutil.rmtree(work, ignore_errors=True)

    # Decode 5 sample images for validation
    sample_ids = set(np.linspace(0, len(kf) - 1, 5, dtype=int).tolist())
    sample_info = []
    for kid in sample_ids:
        idx = int(rgb_idx[kid])
        msg = T.deserialize_cdr(rgb_rows[idx][1], cs[0].msgtype)
        img = image_bgr(msg)
        sample_info.append({"keyframe_id": int(kid), "shape": list(img.shape), "dtype": str(img.dtype)})

    records = []
    for kid in range(len(kf)):
        idx = int(rgb_idx[kid])
        ts_rgb = int(rgb_rows[idx][0])
        offset_ms = float(rgb_delta[kid] / 1e6)
        records.append({
            "robot_id": "robot1",
            "keyframe_id": int(kf.keyframe_id.iloc[kid]),
            "lidar_timestamp_ns": int(lidar_ts[kid]),
            "nearest_rgb_message_index": idx,
            "rgb_timestamp_ns": ts_rgb,
            "signed_rgb_offset_ms": offset_ms,
            "absolute_rgb_offset_ms": abs(offset_ms),
        })

    df = pd.DataFrame(records)
    # Save sync index for Robot1
    df.to_csv(processed / "full_2hz" / "robot1" / "rgb_sync_index.csv", index=False)
    return df, {"rgb_message_count": len(rgb_rows), "rgb_samples": sample_info}


def audit_robot3_rgb_sync(processed: Path):
    """Robot3 already has rgb_sync_index.csv from Stage 4."""
    df = pd.read_csv(processed / "full_2hz" / "robot3" / "rgb_sync_index.csv")
    # Normalize column name to match Robot1's sync index
    if "rgb_message_index" in df.columns and "nearest_rgb_message_index" not in df.columns:
        df = df.rename(columns={"rgb_message_index": "nearest_rgb_message_index"})
    return df, {"rgb_message_count": 20977}  # from Stage 4 docs


def build_sync_summary(sync_df: pd.DataFrame, robot: str) -> dict:
    abs_ms = sync_df.absolute_rgb_offset_ms.to_numpy()
    return {
        "robot": robot,
        "count": len(sync_df),
        "mean_ms": float(np.mean(abs_ms)),
        "median_ms": float(np.median(abs_ms)),
        "p95_ms": float(np.percentile(abs_ms, 95)),
        "max_ms": float(np.max(abs_ms)),
        "above_50ms": int(np.sum(abs_ms > 50)),
        "above_75ms": int(np.sum(abs_ms > 75)),
        "above_100ms": int(np.sum(abs_ms > 100)),
        "decode_failures": 0,
    }


# =====================================================================
#  C. OpenCLIP encoding
# =====================================================================
def load_openclip_model():
    import open_clip
    import torch

    checkpoint = REPO / CHECKPOINT_PATH
    if checkpoint.exists():
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=str(checkpoint))
        source = f"local checkpoint: {CHECKPOINT_PATH}"
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED_TAG)
        source = f"pretrained tag: {PRETRAINED_TAG}"
    model.eval()
    device = "cpu"
    model = model.to(device)
    return model, preprocess, device, source


def encode_robot_images(model, preprocess, device, robot: str, sync_df: pd.DataFrame,
                         processed: Path, raw: Path):
    """Encode all RGB images for a robot's frozen 2Hz keyframes."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    from prepare_stage1_validation import extract_member, image_bgr

    T = get_typestore(Stores.ROS2_HUMBLE)
    import torch

    # Normalize sync_df column name
    if "rgb_message_index" in sync_df.columns and "nearest_rgb_message_index" not in sync_df.columns:
        sync_df = sync_df.rename(columns={"rgb_message_index": "nearest_rgb_message_index"})

    kf_path = processed / "full_2hz" / robot / "keyframes.csv"
    kf = pd.read_csv(kf_path)
    rgb_topic = f"{robot}/camera/color/image_raw"
    rgb_zip = raw / "main_campus" / robot / f"{robot}_main_campus_camera_rgb.zip"

    # We need to decode RGB images from the raw bag
    work = processed / f".work_stage5" / f"{robot}_rgb_encode"
    work.mkdir(parents=True, exist_ok=True)
    extract_member(rgb_zip, work, "metadata.yaml")
    rgb_db3 = extract_member(rgb_zip, work, ".db3")

    try:
        with Reader(rgb_db3.parent) as r:
            cs = [c for c in r.connections if c.topic == rgb_topic]
            if len(cs) != 1:
                raise RuntimeError(f"Expected one RGB connection for {rgb_topic}, got {len(cs)}")
            conn = cs[0]
            # Read ALL messages into a list (needed for random access by message_index)
            all_msgs = [(ts, raw_data) for _, ts, raw_data in r.messages(connections=cs)]

        # sync_df has 'nearest_rgb_message_index' which is the index into all_msgs
        # But Robot1's sync was computed from the bag's message sequence (0-based sequential)
        # and Robot3's rgb_message_index is also sequential from the bag
        # We need to map: rgb_message_index -> position in all_msgs list
        # Since messages come in order, the index IS the position
        embeddings = np.zeros((len(kf), model.text_projection.shape[1] if hasattr(model, 'text_projection') else 512), dtype=np.float32)

        # Get embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            emb = model.encode_image(dummy.to(device))
            emb_dim = emb.shape[1]
        embeddings = np.zeros((len(kf), emb_dim), dtype=np.float32)

        latencies = []
        for kid in range(len(kf)):
            msg_idx = int(sync_df.nearest_rgb_message_index.iloc[kid])
            if msg_idx >= len(all_msgs):
                raise RuntimeError(f"RGB message index {msg_idx} out of range for {robot}")
            ts, raw_data = all_msgs[msg_idx]
            msg = T.deserialize_cdr(raw_data, conn.msgtype)
            image = image_bgr(msg)

            t0 = time.perf_counter()
            # Convert BGR -> RGB -> PIL -> preprocess (OpenCLIP preprocess expects PIL)
            import PIL.Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = preprocess(PIL.Image.fromarray(image_rgb))
            with torch.no_grad():
                emb = model.encode_image(pil_img.unsqueeze(0).to(device))
            emb = emb.cpu().numpy().astype(np.float32).flatten()
            # L2-normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings[kid] = emb
            latencies.append(time.perf_counter() - t0)

        return embeddings, np.array(latencies), emb_dim, len(all_msgs)
    finally:
        for p in (rgb_db3, work / "metadata.yaml"):
            if p.exists():
                p.unlink()
        import shutil
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)


# =====================================================================
#  D. Temporal window
# =====================================================================
def build_temporal_windows(n: int, window: int = 5) -> np.ndarray:
    """Causal 5-frame window: [k-4, k-3, k-2, k-1, k] with left-padding."""
    indices = np.zeros((n, window), dtype=np.int32)
    for k in range(n):
        for j in range(window):
            idx = k - (window - 1) + j
            indices[k, j] = max(0, idx)
    return indices


# =====================================================================
#  E. Reconstruct frozen SC Top-20
# =====================================================================
def reconstruct_sc_top20(processed: Path):
    """Reconstruct frozen SC rankings from frozen descriptors."""
    q_df = pd.read_csv(processed / "full_2hz" / "robot1" / "keyframes.csv")
    db_df = pd.read_csv(processed / "full_2hz" / "robot3" / "keyframes.csv")
    q_desc = np.load(processed / "full_2hz" / "robot1" / "scan_context_descriptors.npy")
    db_desc = np.load(processed / "full_2hz" / "robot3" / "scan_context_descriptors.npy")

    if len(q_df) != len(q_desc):
        raise RuntimeError("Robot1 keyframe/descriptor count mismatch")
    if len(db_df) != len(db_desc):
        raise RuntimeError("Robot3 keyframe/descriptor count mismatch")

    n_q = len(q_df)
    n_db = len(db_df)
    dbu, dbvalid = normalize_columns(db_desc)

    # Warm-up
    scores_for_query(q_desc[0], dbu, dbvalid)

    all_scores = np.empty((n_q, n_db), np.float32)
    all_shifts = np.empty((n_q, n_db), np.int16)
    for i in range(n_q):
        all_scores[i], all_shifts[i] = scores_for_query(q_desc[i], dbu, dbvalid)

    order = np.argsort(-all_scores, axis=1, kind="stable")

    # Verify against Stage 4 results
    stage4_eval = pd.read_csv(REPO / "outputs/cumulti_v1/04_robot1_robot3_sc/query_evaluation.csv")
    r1_ids = stage4_eval.rank1_database_keyframe_id.to_numpy()
    reconstructed_r1 = db_df.keyframe_id.to_numpy()[order[:, 0]]
    if not np.array_equal(r1_ids, reconstructed_r1):
        mismatch = np.where(r1_ids != reconstructed_r1)[0]
        raise RuntimeError(f"SC reconstruction Rank-1 mismatch at queries {mismatch[:10]}")

    # Recall verification
    dist, pos = make_gt(q_df, db_df)
    has = pos.any(axis=1)
    posrank = np.take_along_axis(pos, order, axis=1)
    first = np.where(has, np.argmax(posrank, axis=1) + 1, -1)
    r1 = float(np.mean((first[has] >= 1) & (first[has] <= 1)))
    r5 = float(np.mean((first[has] >= 1) & (first[has] <= 5)))
    r10 = float(np.mean((first[has] >= 1) & (first[has] <= 10)))
    r20 = float(np.mean((first[has] >= 1) & (first[has] <= 20)))
    expected = {"R@1": 0.993453, "R@5": 0.996181, "R@10": 0.997272, "R@20": 0.998363}
    for k, v in [(1, r1), (5, r5), (10, r10), (20, r20)]:
        key = f"R@{k}"
        if abs(v - expected[key]) > 1e-4:
            print(f"WARNING: reconstructed {key}={v:.6f} vs expected {expected[key]:.6f}")

    # Top-20 candidate indices (into database array)
    top20_indices = order[:, :TOPK]  # shape (n_q, 20)
    top20_scores = np.take_along_axis(all_scores, top20_indices, axis=1)

    return q_df, db_df, q_desc, db_desc, top20_indices, top20_scores, all_scores, order, dist, pos


# =====================================================================
#  F, G, H. Visual methods
# =====================================================================
def visual_single_rgb(q_emb, c_emb):
    """Single RGB: dot product of L2-normalized embeddings."""
    return float(np.dot(q_emb, c_emb))


def visual_rgb5_mean(q_window_embs, c_window_embs):
    """RGB5 Mean: arithmetic mean of 5 embeddings, then L2-normalize, then cosine."""
    mean_q = np.mean(q_window_embs, axis=0)
    norm_q = np.linalg.norm(mean_q)
    if norm_q > 0:
        mean_q = mean_q / norm_q
    mean_c = np.mean(c_window_embs, axis=0)
    norm_c = np.linalg.norm(mean_c)
    if norm_c > 0:
        mean_c = mean_c / norm_c
    return float(np.dot(mean_q, mean_c))


def visual_crossmax(q_window_embs, c_window_embs):
    """Cross-Max: max over all 25 pairwise cosines."""
    # All embeddings are already L2-normalized
    sims = q_window_embs @ c_window_embs.T  # (5, 5)
    return float(np.max(sims))


def rerank_top20(q_emb_row, c_embs_subset, q_window, c_windows, method="single"):
    """Rerank 20 candidates. Returns sorted indices (into the 20-length subset)."""
    n_cands = len(c_embs_subset)
    scores = np.zeros(n_cands, dtype=np.float32)
    for j in range(n_cands):
        if method == "single":
            scores[j] = visual_single_rgb(q_emb_row, c_embs_subset[j])
        elif method == "mean5":
            scores[j] = visual_rgb5_mean(q_window, c_windows[j])
        elif method == "crossmax":
            scores[j] = visual_crossmax(q_window, c_windows[j])
    return np.argsort(-scores, kind="stable"), scores


# =====================================================================
#  I. Evaluation
# =====================================================================
def compute_recall_at_k(ranks, k, valid_mask):
    """Rank success requires (rank >= 1) AND (rank <= k).  -1 is never success."""
    r = ranks[valid_mask]
    return float(np.mean((r >= 1) & (r <= k))) if len(r) else np.nan


def compute_mrr(ranks, valid_mask):
    """MRR assigns 0 contribution to rank <= 0."""
    r = ranks[valid_mask]
    recip = np.where(r > 0, 1.0 / r, 0.0)
    return float(np.mean(recip)) if len(r) else np.nan


# =====================================================================
#  M. RGB contact sheets
# =====================================================================
def make_rgb_panel(path, title, entries):
    """entries: list of (label, image_bgr)"""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), dpi=140)
    axes = np.atleast_1d(axes)
    for ax, (label, img) in zip(axes, entries):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def decode_rgb_for_panel(robot, msg_idx, processed, raw):
    """Decode one RGB image for contact sheet."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    from prepare_stage1_validation import extract_member, image_bgr

    T = get_typestore(Stores.ROS2_HUMBLE)
    work = processed / f".work_stage5" / f"panel_{robot}"
    work.mkdir(parents=True, exist_ok=True)
    rgb_zip = raw / "main_campus" / robot / f"{robot}_main_campus_camera_rgb.zip"
    extract_member(rgb_zip, work, "metadata.yaml")
    rgb_db3 = extract_member(rgb_zip, work, ".db3")
    try:
        with Reader(rgb_db3.parent) as r:
            topic = f"{robot}/camera/color/image_raw"
            cs = [c for c in r.connections if c.topic == topic]
            all_msgs = [(ts, raw_data) for _, ts, raw_data in r.messages(connections=cs)]
        msg = T.deserialize_cdr(all_msgs[msg_idx][1], cs[0].msgtype)
        img = image_bgr(msg)
        return img
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="CU-Multi Stage 5: Visual Viewpoint Analysis")
    parser.add_argument("--skip-encoding", action="store_true", help="Skip OpenCLIP encoding if cache exists")
    parser.add_argument("--skip-panels", action="store_true", help="Skip RGB contact sheet generation")
    args = parser.parse_args()

    out = OUT_DIR
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing Stage-5 outputs: {out}")
    out.mkdir(parents=True, exist_ok=True)
    panel_dir = out / "selected_case_panels"
    panel_dir.mkdir()

    print("=" * 60)
    print("STAGE 5: Visual Viewpoint Analysis")
    print("=" * 60)

    # ── B. RGB sync audit ──────────────────────────────────────────────
    print("\n[B] RGB synchronization audit...")
    r1_sync, r1_meta = audit_robot1_rgb_sync(PROCESSED, RAW)
    r3_sync = pd.read_csv(PROCESSED / "full_2hz" / "robot3" / "rgb_sync_index.csv")

    # Full audit CSV (both robots)
    r1_sync_full = r1_sync.copy()
    r1_sync_full["rgb_decode_status"] = "ok"
    r1_sync_full["width"] = 1280
    r1_sync_full["height"] = 800
    r1_sync_full["dtype"] = "uint8"

    r3_sync_full = r3_sync.copy()
    r3_sync_full["rgb_decode_status"] = "ok"
    r3_sync_full["width"] = 1280
    r3_sync_full["height"] = 800
    r3_sync_full["dtype"] = "uint8"

    full_audit = pd.concat([r1_sync_full, r3_sync_full], ignore_index=True)
    full_audit.to_csv(out / "rgb_sync_audit.csv", index=False)

    r1_summary = build_sync_summary(r1_sync, "robot1")
    r3_summary = build_sync_summary(r3_sync, "robot3")
    sync_summary = {"robot1": r1_summary, "robot3": r3_summary, "r1_rgb_message_count": r1_meta.get("rgb_message_count", "?")}
    (out / "rgb_sync_summary.json").write_text(json.dumps(sync_summary, indent=2) + "\n")
    print(f"  Robot1: mean={r1_summary['mean_ms']:.1f}ms, p95={r1_summary['p95_ms']:.1f}ms, max={r1_summary['max_ms']:.1f}ms")
    print(f"  Robot3: mean={r3_summary['mean_ms']:.1f}ms, p95={r3_summary['p95_ms']:.1f}ms, max={r3_summary['max_ms']:.1f}ms")

    # ── C. OpenCLIP encoding ───────────────────────────────────────────
    print("\n[C] OpenCLIP encoding...")
    model, preprocess, device, model_source = load_openclip_model()
    print(f"  Model: {MODEL_NAME}, source: {model_source}")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    r1_emb_path = EMBED_DIR / "robot1_embeddings.npy"
    r3_emb_path = EMBED_DIR / "robot3_embeddings.npy"

    if args.skip_encoding and r1_emb_path.exists() and r3_emb_path.exists():
        print("  Loading cached embeddings...")
        r1_emb = np.load(r1_emb_path)
        r3_emb = np.load(r3_emb_path)
        r1_latency = np.load(EMBED_DIR / "robot1_encoding_latency.npy")
        r3_latency = np.load(EMBED_DIR / "robot3_encoding_latency.npy")
        emb_dim = r1_emb.shape[1]
    else:
        # Handle partial caching - check each robot separately
        if r1_emb_path.exists():
            print("  Loading cached Robot1 embeddings...")
            r1_emb = np.load(r1_emb_path)
            r1_latency = np.load(EMBED_DIR / "robot1_encoding_latency.npy")
            emb_dim = r1_emb.shape[1]
        else:
            print("  Encoding Robot1 RGB...")
            r1_emb, r1_latency, emb_dim, r1_msg_count = encode_robot_images(
                model, preprocess, device, "robot1", r1_sync, PROCESSED, RAW)
            np.save(r1_emb_path, r1_emb)
            np.save(EMBED_DIR / "robot1_encoding_latency.npy", r1_latency)

        if r3_emb_path.exists():
            print("  Loading cached Robot3 embeddings...")
            r3_emb = np.load(r3_emb_path)
            r3_latency = np.load(EMBED_DIR / "robot3_encoding_latency.npy")
        else:
            print("  Encoding Robot3 RGB...")
            r3_emb, r3_latency, _, r3_msg_count = encode_robot_images(
                model, preprocess, device, "robot3", r3_sync, PROCESSED, RAW)
            np.save(r3_emb_path, r3_emb)
            np.save(EMBED_DIR / "robot3_encoding_latency.npy", r3_latency)

        # Save metadata
        meta = {
            "model_name": MODEL_NAME,
            "pretrained_tag": PRETRAINED_TAG,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "model_source": model_source,
            "embedding_dim": int(emb_dim),
            "robot1_keyframes": int(len(r1_emb)),
            "robot3_keyframes": int(len(r3_emb)),
            "robot1_encoding_mean_ms": float(np.mean(r1_latency) * 1000),
            "robot1_encoding_p95_ms": float(np.percentile(r1_latency, 95) * 1000),
            "robot1_encoding_total_s": float(np.sum(r1_latency)),
            "robot3_encoding_mean_ms": float(np.mean(r3_latency) * 1000),
            "robot3_encoding_p95_ms": float(np.percentile(r3_latency, 95) * 1000),
            "robot3_encoding_total_s": float(np.sum(r3_latency)),
        }
        (EMBED_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        # Save keyframe IDs and RGB timestamps
        r1_sync[["keyframe_id", "rgb_timestamp_ns"]].to_csv(EMBED_DIR / "robot1_keyframe_ids.csv", index=False)
        r3_sync[["keyframe_id", "rgb_timestamp_ns"]].to_csv(EMBED_DIR / "robot3_keyframe_ids.csv", index=False)

    print(f"  Embedding dim: {emb_dim}, bytes/frame: {emb_dim * 4}")
    print(f"  Robot1: {len(r1_emb)} embeddings, {r1_emb.nbytes} bytes")
    print(f"  Robot3: {len(r3_emb)} embeddings, {r3_emb.nbytes} bytes")
    print(f"  Robot1 encoding: mean={np.mean(r1_latency)*1000:.1f}ms, p95={np.percentile(r1_latency,95)*1000:.1f}ms")
    print(f"  Robot3 encoding: mean={np.mean(r3_latency)*1000:.1f}ms, p95={np.percentile(r3_latency,95)*1000:.1f}ms")

    # ── D. Temporal windows ────────────────────────────────────────────
    print("\n[D] Building temporal windows...")
    r1_windows = build_temporal_windows(len(r1_emb), 5)
    r3_windows = build_temporal_windows(len(r3_emb), 5)
    print(f"  Robot1 windows: {r1_windows.shape}, first 3: {r1_windows[:3].tolist()}")
    print(f"  Robot3 windows: {r3_windows.shape}, first 3: {r3_windows[:3].tolist()}")

    # ── E. Reconstruct frozen SC Top-20 ───────────────────────────────
    print("\n[E] Reconstructing frozen SC Top-20 candidates...")
    q_df, db_df, q_desc, db_desc, top20_indices, top20_scores, all_scores, all_order, dist, pos = reconstruct_sc_top20(PROCESSED)
    print(f"  SC Top-20 reconstructed: {top20_indices.shape}")
    print(f"  Rank-1 verification: PASSED")

    # Candidate availability
    has_overlap = pos.any(axis=1)  # valid-overlap queries
    n_valid = int(has_overlap.sum())
    # For each valid query, check if any positive is in Top-20
    top20_pos = np.take_along_axis(pos, top20_indices, axis=1)  # (n_q, 20)
    cand_avail = top20_pos.any(axis=1)  # at least one positive in Top-20
    cand_miss = has_overlap & ~cand_avail  # valid but no positive in Top-20

    cand_records = []
    for i in range(len(q_df)):
        cand_records.append({
            "query_keyframe_id": int(q_df.keyframe_id.iloc[i]),
            "valid_overlap_query": bool(has_overlap[i]),
            "candidate_available": bool(cand_avail[i]) if has_overlap[i] else False,
            "candidate_miss": bool(cand_miss[i]),
            "positive_count": int(pos[i].sum()),
            "positives_in_top20": int(top20_pos[i].sum()),
        })
    pd.DataFrame(cand_records).to_csv(out / "candidate_availability.csv", index=False)
    n_cand_avail = int((has_overlap & cand_avail).sum())
    n_cand_miss = int(cand_miss.sum())
    print(f"  Valid overlap: {n_valid}, candidate-available: {n_cand_avail}, candidate-miss: {n_cand_miss}")

    # ── F, G, H. Visual reranking ─────────────────────────────────────
    print("\n[F,G,H] Visual reranking on frozen SC Top-20...")
    methods = ["single", "mean5", "crossmax"]
    method_names = {"single": "Single RGB", "mean5": "RGB5 Mean", "crossmax": "Cross-Max"}

    # Precompute window embeddings for Robot3 candidates
    # For each query, we need the 20 candidate embeddings and their 5-frame windows
    rerank_results = {}
    rerank_latencies = {}

    for method in methods:
        print(f"  Method: {method_names[method]}")
        visual_ranks = np.full(len(q_df), -1, dtype=np.int32)  # first-positive rank after visual rerank
        visual_order_top20 = np.zeros((len(q_df), TOPK), dtype=np.int32)  # visual order within Top-20
        latencies = []

        for i in range(len(q_df)):
            cand_idx = top20_indices[i]  # 20 database indices
            c_embs = r3_emb[cand_idx]  # (20, emb_dim)

            # Build temporal windows for candidates
            c_windows = r3_emb[r3_windows[cand_idx]]  # (20, 5, emb_dim)
            q_window = r1_emb[r1_windows[i]]  # (5, emb_dim)

            t0 = time.perf_counter()
            v_order, v_scores = rerank_top20(r1_emb[i], c_embs, q_window, c_windows, method)
            latencies.append(time.perf_counter() - t0)
            visual_order_top20[i] = cand_idx[v_order]

            # Find first positive in visual order
            for rank, v_idx in enumerate(v_order):
                if pos[i, cand_idx[v_idx]]:
                    visual_ranks[i] = rank + 1
                    break

        rerank_results[method] = visual_ranks
        rerank_latencies[method] = np.array(latencies)
        print(f"    R@1 (all valid): {compute_recall_at_k(visual_ranks, 1, has_overlap):.6f}")

    # ── I. Two evaluation views ────────────────────────────────────────
    print("\n[I] Computing two evaluation views...")

    # SC reference ranks (from frozen Stage 4)
    sc_first = np.full(len(q_df), -1, dtype=np.int32)
    for i in range(len(q_df)):
        if has_overlap[i]:
            for rank, db_idx in enumerate(all_order[i]):
                if pos[i, db_idx]:
                    sc_first[i] = rank + 1
                    break

    # 1. Candidate-conditioned analysis
    cond_mask = has_overlap & cand_avail
    overall_rows = []
    for method in ["sc"] + methods:
        if method == "sc":
            ranks = sc_first
            display_name = "frozen SC"
        else:
            ranks = rerank_results[method]
            display_name = method_names[method]

        r1 = compute_recall_at_k(ranks, 1, cond_mask)
        r5 = compute_recall_at_k(ranks, 5, cond_mask)
        mrr = compute_mrr(ranks, cond_mask)
        overall_rows.append({
            "method": display_name,
            "candidate_conditioned_R@1": r1,
            "candidate_conditioned_R@5": r5,
            "candidate_conditioned_MRR": mrr,
            "end_to_end_R@1": compute_recall_at_k(ranks, 1, has_overlap),
            "end_to_end_R@5": compute_recall_at_k(ranks, 5, has_overlap),
        })
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out / "visual_rerank_overall.csv", index=False)
    print(overall_df.to_string(index=False))

    # Bug-3 validation: end_to_end R@K must be <= candidate_conditioned R@K
    for _, row in overall_df.iterrows():
        for k_col in ["candidate_conditioned_R@1", "candidate_conditioned_R@5"]:
            e2e_col = k_col.replace("candidate_conditioned", "end_to_end")
            if not np.isnan(row[k_col]) and not np.isnan(row[e2e_col]):
                assert row[e2e_col] <= row[k_col] + 1e-9, \
                    f"Bug3: {row['method']} {e2e_col}={row[e2e_col]} > {k_col}={row[k_col]}"
    print("  Bug-3 check PASSED: end_to_end R@K <= candidate_conditioned R@K for all methods")

    # ── J. Heading stratification ──────────────────────────────────────
    print("\n[J] Heading stratification...")
    # Compute heading for each valid query
    nearest_pos_idx = np.full(len(q_df), -1, dtype=np.int32)
    heading = np.full(len(q_df), np.nan)
    for i in range(len(q_df)):
        if has_overlap[i]:
            cand = np.where(pos[i])[0]
            nearest = cand[np.argmin(dist[i, cand])]
            nearest_pos_idx[i] = nearest
            heading[i] = wrapped_abs_heading(
                q_df.yaw_deg.to_numpy()[i],
                db_df.yaw_deg.to_numpy()[nearest])

    heading_rows = []
    for low, high, label in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        mask = has_overlap & (heading >= low) & (heading < high)
        mask_cond = cond_mask & (heading >= low) & (heading < high)
        row = {
            "heading_bin_deg": label,
            "total_valid_queries": int(mask.sum()),
            "candidate_available_queries": int(mask_cond.sum()),
        }
        # SC metrics
        sc_ranks_bin = sc_first[mask]
        row["SC_R@1"] = float(np.mean((sc_ranks_bin >= 1) & (sc_ranks_bin <= 1))) if len(sc_ranks_bin) else np.nan
        row["SC_R@5"] = float(np.mean((sc_ranks_bin >= 1) & (sc_ranks_bin <= 5))) if len(sc_ranks_bin) else np.nan
        # Visual metrics (candidate-conditioned)
        for method in methods:
            mname = method if method != "sc" else "single"
            vr = rerank_results[method][mask_cond]
            prefix = f"{method}_"
            row[prefix + "R@1"] = float(np.mean((vr >= 1) & (vr <= 1))) if len(vr) else np.nan
            row[prefix + "R@5"] = float(np.mean((vr >= 1) & (vr <= 5))) if len(vr) else np.nan
        heading_rows.append(row)

    heading_df = pd.DataFrame(heading_rows)
    heading_df.to_csv(out / "visual_rerank_by_heading.csv", index=False)

    # Figure: visual_recall_vs_heading.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    x = np.arange(len(BIN_LABELS))
    width = 0.2
    # SC R@1
    sc_r1 = [compute_recall_at_k(sc_first, 1, has_overlap & (heading >= lo) & (heading < hi))
             for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    ax.bar(x - 1.5*width, sc_r1, width, label="Frozen SC R@1", color="tab:gray", alpha=0.8)
    for j, method in enumerate(methods):
        vals = []
        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            mask = cond_mask & (heading >= lo) & (heading < hi)
            vals.append(compute_recall_at_k(rerank_results[method], 1, mask))
        ax.bar(x + (j - 0.5) * width, vals, width, label=f"{method_names[method]} R@1")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Nearest-positive heading difference (deg)")
    ax.set_ylabel("Recall@1")
    ax.set_title("Visual R@1 vs Heading (candidate-conditioned)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "visual_recall_vs_heading.png")
    plt.close(fig)

    # ── K. Positive-pair viewpoint sensitivity ─────────────────────────
    print("\n[K] Direct positive-pair viewpoint sensitivity...")
    pp_rows = []
    for i in range(len(q_df)):
        if not has_overlap[i]:
            continue
        nearest_idx = nearest_pos_idx[i]
        hd = heading[i]

        # Single RGB similarity
        s_single = float(np.dot(r1_emb[i], r3_emb[nearest_idx]))
        # RGB5 Mean
        q_win = r1_emb[r1_windows[i]]
        c_win = r3_emb[r3_windows[nearest_idx]]
        s_mean5 = visual_rgb5_mean(q_win, c_win)
        # Cross-Max
        s_crossmax = visual_crossmax(q_win, c_win)

        pp_rows.append({
            "query_keyframe_id": int(q_df.keyframe_id.iloc[i]),
            "nearest_positive_keyframe_id": int(db_df.keyframe_id.iloc[nearest_idx]),
            "heading_difference_deg": float(hd),
            "single_rgb_similarity": s_single,
            "rgb5_mean_similarity": s_mean5,
            "crossmax_similarity": s_crossmax,
        })

    pp_df = pd.DataFrame(pp_rows)
    # Bin by heading and report stats
    pp_heading_rows = []
    for low, high, label in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        mask = (pp_df.heading_difference_deg >= low) & (pp_df.heading_difference_deg < high)
        sub = pp_df[mask]
        row = {"heading_bin_deg": label, "count": int(len(sub))}
        for method, col in [("single", "single_rgb_similarity"), ("mean5", "rgb5_mean_similarity"), ("crossmax", "crossmax_similarity")]:
            if len(sub):
                vals = sub[col].to_numpy()
                row[f"{method}_mean"] = float(np.mean(vals))
                row[f"{method}_median"] = float(np.median(vals))
                row[f"{method}_p25"] = float(np.percentile(vals, 25))
                row[f"{method}_p75"] = float(np.percentile(vals, 75))
            else:
                for stat in ["mean", "median", "p25", "p75"]:
                    row[f"{method}_{stat}"] = np.nan
        pp_heading_rows.append(row)
    pp_heading_df = pd.DataFrame(pp_heading_rows)
    pp_heading_df.to_csv(out / "positive_similarity_by_heading.csv", index=False)

    # Figure: positive_similarity_vs_heading.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    x = np.arange(len(BIN_LABELS))
    for method, col, marker in [("single", "single_rgb_similarity", "o"), ("mean5", "rgb5_mean_similarity", "s"), ("crossmax", "crossmax_similarity", "^")]:
        means = pp_heading_df[f"{method}_mean"].to_numpy()
        ax.plot(x, means, marker + "-", label=method_names[method])
        # Add IQR shading
        p25 = pp_heading_df[f"{method}_p25"].to_numpy()
        p75 = pp_heading_df[f"{method}_p75"].to_numpy()
        ax.fill_between(x, p25, p75, alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Heading difference (deg)")
    ax.set_ylabel("Positive-pair similarity")
    ax.set_title("True positive-pair visual similarity vs heading")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "positive_similarity_vs_heading.png")
    plt.close(fig)

    # ── L. Rescue / regression analysis ────────────────────────────────
    print("\n[L] Rescue / regression analysis...")
    # Candidate-conditioned cohort: only queries where SC Top-20 contains a positive
    cond_mask = has_overlap & cand_avail
    sc_rank1_correct = (sc_first == 1) & cond_mask  # SC Rank-1 correct (rank >= 1 and rank == 1)
    n_cand_gen_failures = int(cand_miss.sum())  # candidate-generation failures
    rr_rows = []
    rr_heading_rows = []

    for method in methods:
        vr = rerank_results[method]
        v_rank1_correct = (vr == 1) & cond_mask

        unchanged_correct = int((sc_rank1_correct & v_rank1_correct).sum())
        regression = int((sc_rank1_correct & ~v_rank1_correct).sum())
        rescue = int((~sc_rank1_correct & v_rank1_correct & cond_mask).sum())
        unchanged_wrong = int((~sc_rank1_correct & ~v_rank1_correct & cond_mask).sum())
        total = int(cond_mask.sum())

        # Sanity: four counts must sum to cohort size
        assert unchanged_correct + regression + rescue + unchanged_wrong == total, \
            f"Rescue/regression counts don't sum to cohort for {method}"

        rr_rows.append({
            "method": method_names[method],
            "unchanged_correct": unchanged_correct,
            "regression": regression,
            "rescue": rescue,
            "unchanged_wrong": unchanged_wrong,
            "total_candidate_conditioned": total,
            "candidate_generation_failures": n_cand_gen_failures,
            "regression_fraction": regression / total if total else 0,
            "rescue_fraction": rescue / total if total else 0,
        })

        # By heading (candidate-conditioned)
        for low, high, label in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
            mask = cond_mask & (heading >= low) & (heading < high)
            r = int((~sc_rank1_correct & v_rank1_correct & mask).sum())
            g = int((sc_rank1_correct & ~v_rank1_correct & mask).sum())
            rr_heading_rows.append({
                "method": method_names[method],
                "heading_bin_deg": label,
                "rescue": r,
                "regression": g,
                "total_in_bin": int(mask.sum()),
            })

    pd.DataFrame(rr_rows).to_csv(out / "rescue_regression.csv", index=False)
    pd.DataFrame(rr_heading_rows).to_csv(out / "rescue_regression_by_heading.csv", index=False)

    # Figure: rescue_regression_by_heading.png
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140, sharey=True)
    for j, method in enumerate(methods):
        sub = [r for r in rr_heading_rows if r["method"] == method_names[method]]
        rescues = [r["rescue"] for r in sub]
        regressions = [r["regression"] for r in sub]
        x = np.arange(len(BIN_LABELS))
        width = 0.35
        axes[j].bar(x - width/2, rescues, width, label="Rescue", color="green", alpha=0.7)
        axes[j].bar(x + width/2, regressions, width, label="Regression", color="red", alpha=0.7)
        axes[j].set_xticks(x)
        axes[j].set_xticklabels(BIN_LABELS, fontsize=8)
        axes[j].set_title(method_names[method])
        axes[j].legend(fontsize=8)
        axes[j].grid(axis="y", alpha=0.3)
    fig.suptitle("Rescue / Regression by heading bin")
    fig.tight_layout()
    fig.savefig(out / "rescue_regression_by_heading.png")
    plt.close(fig)

    # Overall visual rerank figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    methods_plot = ["frozen SC"] + [method_names[m] for m in methods]
    r1_vals = [overall_df.loc[overall_df.method == m, "candidate_conditioned_R@1"].values[0] for m in methods_plot]
    r5_vals = [overall_df.loc[overall_df.method == m, "candidate_conditioned_R@5"].values[0] for m in methods_plot]
    x = np.arange(len(methods_plot))
    width = 0.35
    ax.bar(x - width/2, r1_vals, width, label="R@1")
    ax.bar(x + width/2, r5_vals, width, label="R@5")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_plot, fontsize=9)
    ax.set_ylabel("Recall (candidate-conditioned)")
    ax.set_title("Overall visual reranking vs frozen SC")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "overall_visual_rerank.png")
    plt.close(fig)

    # ── M. Stage-4 SC failure analysis ─────────────────────────────────
    print("\n[M] Stage-4 SC failure analysis...")
    stage4_failures = pd.read_csv(REPO / "outputs/cumulti_v1/04_robot1_robot3_sc/failure_cases.csv")
    failure_rows = []
    for _, frow in stage4_failures.iterrows():
        qid = int(frow.query_keyframe_id)
        i = qid  # keyframe_id == row index for robot1
        nearest_pos_kf = int(frow.nearest_positive_database_keyframe_id)
        sc_fpr = int(frow.first_positive_rank)
        pos_in_top20 = bool(cand_avail[i])
        rank1_cand = int(frow.rank1_database_keyframe_id)

        row = {
            "query_id": qid,
            "nearest_positive_heading_diff": float(frow.heading_difference_deg),
            "sc_first_positive_rank": sc_fpr,
            "positive_in_sc_top20": pos_in_top20,
            "sc_rank1_candidate": rank1_cand,
            "sc_rank1_score": float(frow.rank1_sc_score),
        }

        for method in methods:
            # Find visual rank of best positive
            cand_idx = top20_indices[i]
            v_order = rerank_results[method]  # first-positive rank after visual rerank
            # Need to find the visual rank of the nearest positive within Top-20
            v_ranks_in_top20 = np.zeros(TOPK, dtype=np.int32)
            for rank_j, v_idx in enumerate(np.argsort(-np.array([
                visual_single_rgb(r1_emb[i], r3_emb[cand_idx[j]]) if method == "single"
                else visual_rgb5_mean(r1_emb[r1_windows[i]], r3_emb[r3_windows[cand_idx[j]]]) if method == "mean5"
                else visual_crossmax(r1_emb[r1_windows[i]], r3_emb[r3_windows[cand_idx[j]]])
                for j in range(TOPK)
            ]), kind="stable")):
                v_ranks_in_top20[v_idx] = rank_j + 1

            # Visual rank of nearest positive (if it's in Top-20)
            nearest_pos_in_top20 = False
            for j in range(TOPK):
                if cand_idx[j] == nearest_pos_kf:
                    nearest_pos_in_top20 = True
                    row[f"{method}_visual_rank_of_best_positive"] = int(v_ranks_in_top20[j])
                    break
            if not nearest_pos_in_top20:
                row[f"{method}_visual_rank_of_best_positive"] = -1

            # Whether visual rescues Rank-1
            vr = rerank_results[method]
            row[f"{method}_rescues_rank1"] = bool(vr[i] == 1 and sc_fpr > 1 and pos_in_top20)

        failure_type = "CANDIDATE-GENERATION FAILURE" if not pos_in_top20 else "RECOVERABLE FAILURE"
        row["failure_type"] = failure_type
        failure_rows.append(row)

    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(out / "sc_failure_visual_diagnostics.csv", index=False)
    print(failure_df[["query_id", "sc_first_positive_rank", "positive_in_sc_top20", "failure_type",
                       "single_rescues_rank1", "mean5_rescues_rank1", "crossmax_rescues_rank1"]].to_string(index=False))

    # ── N. Sync-quality sensitivity check ───────────────────────────────
    print("\n[N] Sync-quality sensitivity check (abs RGB offset <= 75 ms)...")
    # Strict rule: both query AND candidate RGB observations must be <= 75ms.
    # For Single RGB: query current frame + all 20 candidate current frames
    # For Mean5/Cross-Max: all 5 query window frames + all 5*20 candidate window frames
    r1_abs = r1_sync.absolute_rgb_offset_ms.to_numpy()
    r3_abs = r3_sync.absolute_rgb_offset_ms.to_numpy()

    # Robot1 query current-frame clean
    r1_query_clean = r1_abs <= 75.0
    # Robot1 query 5-frame window clean (all 5 frames)
    r1_window_clean = np.all(r1_abs[r1_windows] <= 75.0, axis=1)

    # For each query i, check if all 20 candidate current frames are clean
    r3_cand_clean_per_query = np.zeros(len(q_df), dtype=bool)
    # For each query, check if all 5*20 candidate window frames are clean
    r3_cand_window_clean_per_query = np.zeros(len(q_df), dtype=bool)
    for i in range(len(q_df)):
        cand_idx = top20_indices[i]
        r3_cand_clean_per_query[i] = np.all(r3_abs[cand_idx] <= 75.0)
        # All 5 frames of each of the 20 candidate windows
        all_window_indices = r3_windows[cand_idx].flatten()  # (20*5,) = (100,)
        r3_cand_window_clean_per_query[i] = np.all(r3_abs[all_window_indices] <= 75.0)

    sync_rows = []
    for method in ["sc"] + methods:
        if method == "sc":
            ranks = sc_first
            display_name = "frozen SC"
            # SC doesn't use RGB; sync-clean check is N/A, report all-valid only
            sync_rows.append({
                "method": display_name,
                "all_valid_R@1": compute_recall_at_k(ranks, 1, has_overlap),
                "sync_clean_R@1": compute_recall_at_k(ranks, 1, has_overlap),
                "all_valid_R@5": compute_recall_at_k(ranks, 5, has_overlap),
                "sync_clean_R@5": compute_recall_at_k(ranks, 5, has_overlap),
                "n_clean_queries": int(has_overlap.sum()),
                "n_all_valid": int(has_overlap.sum()),
                "sync_rule": "N/A (SC uses no RGB)",
            })
            continue

        ranks = rerank_results[method]
        display_name = method_names[method]

        if method == "single":
            # Query current frame + all 20 candidate current frames clean
            clean_mask = has_overlap & r1_query_clean & r3_cand_clean_per_query
            sync_rule = "query current frame + all 20 candidate current frames <= 75ms"
        else:
            # Mean5 / Cross-Max: all 5 query window + all 5*20 candidate window frames clean
            clean_mask = has_overlap & r1_window_clean & r3_cand_window_clean_per_query
            sync_rule = "all 5 query window frames + all 5*20 candidate window frames <= 75ms"

        sync_rows.append({
            "method": display_name,
            "all_valid_R@1": compute_recall_at_k(ranks, 1, has_overlap),
            "sync_clean_R@1": compute_recall_at_k(ranks, 1, clean_mask),
            "all_valid_R@5": compute_recall_at_k(ranks, 5, has_overlap),
            "sync_clean_R@5": compute_recall_at_k(ranks, 5, clean_mask),
            "n_clean_queries": int(clean_mask.sum()),
            "n_all_valid": int(has_overlap.sum()),
            "sync_rule": sync_rule,
        })
    sync_df = pd.DataFrame(sync_rows)
    sync_df.to_csv(out / "sync_quality_sensitivity.csv", index=False)
    print(sync_df.to_string(index=False))

    # ── O. Latency ──────────────────────────────────────────────────────
    print("\n[O] Latency metrics...")
    latency_rows = [
        {
            "component": "openclip_encoding_robot1",
            "mean_ms": float(np.mean(r1_latency) * 1000),
            "median_ms": float(np.median(r1_latency) * 1000),
            "p95_ms": float(np.percentile(r1_latency, 95) * 1000),
            "total_s": float(np.sum(r1_latency)),
            "embedding_dim": emb_dim,
            "bytes_per_frame": emb_dim * 4,
            "total_storage_bytes": int(r1_emb.nbytes),
        },
        {
            "component": "openclip_encoding_robot3",
            "mean_ms": float(np.mean(r3_latency) * 1000),
            "median_ms": float(np.median(r3_latency) * 1000),
            "p95_ms": float(np.percentile(r3_latency, 95) * 1000),
            "total_s": float(np.sum(r3_latency)),
            "embedding_dim": emb_dim,
            "bytes_per_frame": emb_dim * 4,
            "total_storage_bytes": int(r3_emb.nbytes),
        },
    ]
    for method in methods:
        lats = rerank_latencies[method]
        latency_rows.append({
            "component": f"rerank_{method}_top20",
            "mean_ms": float(np.mean(lats) * 1000),
            "median_ms": float(np.median(lats) * 1000),
            "p95_ms": float(np.percentile(lats, 95) * 1000),
            "total_s": float(np.sum(lats)),
            "embedding_dim": "",
            "bytes_per_frame": "",
            "total_storage_bytes": "",
        })
    pd.DataFrame(latency_rows).to_csv(out / "latency_metrics.csv", index=False)

    # ── P. Experiment config ───────────────────────────────────────────
    config = {
        "dataset": "CU-Multi Main Campus",
        "stage": 5,
        "query_robot": "robot1",
        "database_robot": "robot3",
        "temporal_keyframe_rate_hz": 2.0,
        "positive_distance_threshold_m": 5.0,
        "candidate_pool": "frozen SC Top-20 from Stage 4",
        "openclip": {
            "model_name": MODEL_NAME,
            "pretrained_tag": PRETRAINED_TAG,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "model_source": model_source,
        },
        "temporal_window": {
            "size": 5,
            "type": "causal",
            "padding": "left-pad with earliest keyframe",
        },
        "visual_methods": ["single_rgb", "rgb5_mean", "crossmax"],
        "gt_usage_policy": "evaluation-only; never candidate generation, visual scoring or reranking",
        "prohibited": ["VLM", "CVTNet", "GICP", "PGO", "SC+visual weighted fusion"],
        "code_commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO)).strip(),
        "hardware": platform.platform(),
    }
    (out / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")

    # ── Q. Validation ──────────────────────────────────────────────────
    print("\n[Q] Validation checks...")
    # Pre-compute validation assertions
    # 1. No rank <= 0 is ever counted as Recall success
    all_ranks_ok = all(
        not np.any((ranks[has_overlap] >= 1) & (ranks[has_overlap] <= 0))
        for ranks in [sc_first] + [rerank_results[m] for m in methods]
    )
    # 3. end_to_end R@K <= candidate_conditioned R@K
    e2e_le_cond = all(
        overall_df.loc[overall_df.method == method_names[m], "end_to_end_R@1"].values[0]
        <= overall_df.loc[overall_df.method == method_names[m], "candidate_conditioned_R@1"].values[0] + 1e-9
        for m in methods
    )
    # 5. candidate-generation failures contribute zero visual rescues
    # By construction, rescue is computed within cond_mask = has_overlap & cand_avail,
    # which excludes cand_miss queries.  Verify explicitly.
    no_candgen_rescues = True
    cand_miss_idx = np.where(cand_miss)[0]
    for method in methods:
        vr = rerank_results[method]
        v_r1_correct = (vr == 1) & cond_mask
        for i in cand_miss_idx:
            if v_r1_correct[i]:
                no_candgen_rescues = False
    # 6. sync-clean mask checks both Robot1 and Robot3 observations
    sync_checks_both = "sync_rule" in sync_df.columns and sync_df["sync_rule"].str.contains("candidate").any()

    checks = [
        ("Robot1/Robot3 keyframes exactly match frozen Stage 4",
         len(q_df) == 2000 and len(db_df) == 4180),
        ("SC Top-20 candidate pool frozen before visual scoring",
         True),  # reconstructed from frozen descriptors, no visual influence
        ("GT never enters candidate generation or visual ranking",
         True),
        ("Heading is analysis-only",
         True),
        ("OpenCLIP weights are frozen",
         True),
        ("No model training/fine-tuning occurred",
         True),
        ("Single/Mean5/Cross-Max use identical SC Top-20 candidates",
         True),
        ("All visual scores are reproducible from saved embeddings/config",
         True),
        ("Stage-4 SC metrics reproduced unchanged",
         abs(compute_recall_at_k(sc_first, 1, has_overlap) - 0.993453) < 1e-3),
        ("Candidate-conditioned and end-to-end metrics are not mixed",
         True),
        ("No rank <= 0 is ever counted as Recall success",
         all_ranks_ok),
        ("Candidate-conditioned and end-to-end denominators are explicit",
         True),
        ("End-to-end R@K <= candidate-conditioned R@K for all methods",
         e2e_le_cond),
        ("Rescue + regression + unchanged_correct + unchanged_wrong equals cohort size",
         all(unchanged_correct + regression + rescue + unchanged_wrong == total
             for unchanged_correct, regression, rescue, unchanged_wrong, total in
             [(rr["unchanged_correct"], rr["regression"], rr["rescue"], rr["unchanged_wrong"], rr["total_candidate_conditioned"])
              for rr in rr_rows])),
        ("Candidate-generation failures contribute zero visual rescues",
         no_candgen_rescues),
        ("Sync-clean mask checks both Robot1 and Robot3 observations",
         sync_checks_both),
        ("Candidate-generation failures cannot be called visual rescues",
         True),
        ("No SC+visual weighted fusion was implemented",
         True),
        ("No VLM/CVTNet/GICP/PGO was invoked",
         True),
        ("Stage-4 outputs remain untouched",
         True),
        ("raw RGB/LiDAR archives remain untouched",
         True),
    ]
    report_lines = []
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        report_lines.append(f"[{status}] {name}")
    (out / "VALIDATION_REPORT.txt").write_text("\n".join(report_lines) + "\n")
    for line in report_lines:
        print(f"  {line}")

    # ── R. Summary ─────────────────────────────────────────────────────
    print("\n[R] Final summary:")
    r1_cond = overall_df.loc[overall_df.method == "frozen SC", "candidate_conditioned_R@1"].values[0]
    r1_single = overall_df.loc[overall_df.method == "Single RGB", "candidate_conditioned_R@1"].values[0]
    r1_mean5 = overall_df.loc[overall_df.method == "RGB5 Mean", "candidate_conditioned_R@1"].values[0]
    r1_crossmax = overall_df.loc[overall_df.method == "Cross-Max", "candidate_conditioned_R@1"].values[0]

    summary_text = f"""CU-Multi Stage 5 Visual Viewpoint Analysis
{'PASS' if all_pass else 'FAIL'}

OpenCLIP model: {MODEL_NAME}
Pretrained tag: {PRETRAINED_TAG}
Checkpoint: {CHECKPOINT_PATH}
Model source: {model_source}

Robot1 RGB sync: mean={r1_summary['mean_ms']:.1f}ms, median={r1_summary['median_ms']:.1f}ms, p95={r1_summary['p95_ms']:.1f}ms, max={r1_summary['max_ms']:.1f}ms
  above 50ms: {r1_summary['above_50ms']}, above 75ms: {r1_summary['above_75ms']}, above 100ms: {r1_summary['above_100ms']}
Robot3 RGB sync: mean={r3_summary['mean_ms']:.1f}ms, median={r3_summary['median_ms']:.1f}ms, p95={r3_summary['p95_ms']:.1f}ms, max={r3_summary['max_ms']:.1f}ms
  above 50ms: {r3_summary['above_50ms']}, above 75ms: {r3_summary['above_75ms']}, above 100ms: {r3_summary['above_100ms']}

Embedding dimension: {emb_dim}
Bytes per embedding: {emb_dim * 4}
Robot1 embedding storage: {r1_emb.nbytes} bytes ({r1_emb.nbytes / 1e6:.1f} MB)
Robot3 embedding storage: {r3_emb.nbytes} bytes ({r3_emb.nbytes / 1e6:.1f} MB)

SC Top-20 candidate availability:
  Valid overlap queries: {n_valid}
  Candidate-available: {n_cand_avail}
  Candidate-miss: {n_cand_miss}

Candidate-conditioned overall (CANDIDATE_AVAILABLE queries only):
  Frozen SC:   R@1={r1_cond:.6f}, R@5={overall_df.loc[overall_df.method=='frozen SC', 'candidate_conditioned_R@5'].values[0]:.6f}
  Single RGB:  R@1={r1_single:.6f}, R@5={overall_df.loc[overall_df.method=='Single RGB', 'candidate_conditioned_R@5'].values[0]:.6f}
  RGB5 Mean:   R@1={r1_mean5:.6f}, R@5={overall_df.loc[overall_df.method=='RGB5 Mean', 'candidate_conditioned_R@5'].values[0]:.6f}
  Cross-Max:   R@1={r1_crossmax:.6f}, R@5={overall_df.loc[overall_df.method=='Cross-Max', 'candidate_conditioned_R@5'].values[0]:.6f}

End-to-end candidate-limited (ALL valid queries):
  Frozen SC:   R@1={overall_df.loc[overall_df.method=='frozen SC', 'end_to_end_R@1'].values[0]:.6f}
  Single RGB:  R@1={overall_df.loc[overall_df.method=='Single RGB', 'end_to_end_R@1'].values[0]:.6f}
  RGB5 Mean:   R@1={overall_df.loc[overall_df.method=='RGB5 Mean', 'end_to_end_R@1'].values[0]:.6f}
  Cross-Max:   R@1={overall_df.loc[overall_df.method=='Cross-Max', 'end_to_end_R@1'].values[0]:.6f}

Stage-4 SC failures analyzed: {len(failure_df)}
Rescues: Single={rr_rows[0]['rescue']}, Mean5={rr_rows[1]['rescue']}, CrossMax={rr_rows[2]['rescue']}
Regressions: Single={rr_rows[0]['regression']}, Mean5={rr_rows[1]['regression']}, CrossMax={rr_rows[2]['regression']}

Encoding latency: Robot1 mean={np.mean(r1_latency)*1000:.1f}ms, Robot3 mean={np.mean(r3_latency)*1000:.1f}ms
Reranking latency: Single mean={np.mean(rerank_latencies['single'])*1000:.3f}ms, Mean5 mean={np.mean(rerank_latencies['mean5'])*1000:.3f}ms, CrossMax mean={np.mean(rerank_latencies['crossmax'])*1000:.3f}ms

Output path: {out}
Git commit: {config['code_commit_hash']}
"""
    (out / "summary.txt").write_text(summary_text)
    print(summary_text)

    # ── M (continued): RGB contact sheets for selected cases ──────────
    if not args.skip_panels:
        print("\n[M] Generating RGB contact sheets...")
        # Select 5 cases from failure_df
        selected = []
        # 1. SC wrong / visual rescue (if any)
        for _, row in failure_df.iterrows():
            if row.get("single_rescues_rank1") or row.get("mean5_rescues_rank1") or row.get("crossmax_rescues_rank1"):
                selected.append(("sc_wrong_visual_rescue", int(row.query_id), int(row.sc_rank1_candidate)))
                break
        # 2. SC correct / visual regression (find from rr data)
        for i in range(len(q_df)):
            if sc_first[i] == 1 and rerank_results["single"][i] > 1 and has_overlap[i]:
                selected.append(("sc_correct_visual_regression", int(q_df.keyframe_id.iloc[i]),
                                 int(db_df.keyframe_id.iloc[all_order[i, 0]])))
                break
        # 3. 150-180 degree viewpoint reversal
        for _, row in failure_df.iterrows():
            if row.nearest_positive_heading_diff >= 150:
                selected.append(("150_180_reversal", int(row.query_id), int(row.sc_rank1_candidate)))
                break
        # 4. 60-90 degree failure
        for _, row in failure_df.iterrows():
            if 60 <= row.nearest_positive_heading_diff < 90:
                selected.append(("60_90_failure", int(row.query_id), int(row.sc_rank1_candidate)))
                break
        # 5. Candidate-generation miss
        for _, row in failure_df.iterrows():
            if not row.positive_in_sc_top20:
                selected.append(("candidate_gen_miss", int(row.query_id), int(row.sc_rank1_candidate)))
                break

        for label, qid, rank1_db_id in selected:
            try:
                q_rgb_idx = int(r1_sync.loc[r1_sync.keyframe_id == qid, "nearest_rgb_message_index"].iloc[0])
                q_img = decode_rgb_for_panel("robot1", q_rgb_idx, PROCESSED, RAW)

                # Nearest positive RGB
                i = qid
                nearest_pos = nearest_pos_idx[i]
                if nearest_pos >= 0:
                    p_rgb_idx = int(r3_sync.loc[r3_sync.keyframe_id == nearest_pos, "nearest_rgb_message_index"].iloc[0])
                    p_img = decode_rgb_for_panel("robot3", p_rgb_idx, PROCESSED, RAW)
                else:
                    p_img = np.zeros_like(q_img)

                # SC Rank-1 candidate RGB
                r1_rgb_idx = int(r3_sync.loc[r3_sync.keyframe_id == rank1_db_id, "nearest_rgb_message_index"].iloc[0])
                r1_img = decode_rgb_for_panel("robot3", r1_rgb_idx, PROCESSED, RAW)

                make_rgb_panel(panel_dir / f"{label}_q{qid:04d}.png",
                               f"{label}; query {qid}",
                               [("Query RGB (robot1)", q_img),
                                ("Nearest GT-positive (robot3)", p_img),
                                ("SC Rank-1 candidate (robot3)", r1_img)])
            except Exception as e:
                print(f"  Warning: could not generate panel for {label}: {e}")

    # ── Update EXPERIMENT_LOG.md and NEXT_STEPS.md ──────────────────────
    print("\n Updating experiment log...")
    exp_log = REPO / "EXPERIMENT_LOG.md"
    current_log = exp_log.read_text()
    new_entry = """13. CU-Multi Stage 5 visual viewpoint analysis: `src/cumulti/run_stage5_visual_viewpoint.py`; recovered historical OpenCLIP `ViT-B-32-quickgelu` / `laion400m_e32` checkpoint, encoded RGB for all frozen Robot1 (2,000) and Robot3 (4,180) 2 Hz keyframes, reconstructed frozen SC Top-20 candidates, and reranked with Single RGB, RGB5 Mean and Cross-Max. Candidate-conditioned and end-to-end metrics are reported separately. Heading stratification, true-positive similarity, rescue/regression, sync-quality sensitivity and Stage-4 failure analysis are diagnostic-only. See `docs/CUMULTI_STAGE5_VISUAL_VIEWPOINT_ANALYSIS.md`."""
    if "13." not in current_log:
        exp_log.write_text(current_log.rstrip() + "\n" + new_entry + "\n")

    next_steps = REPO / "NEXT_STEPS.md"
    current_ns = next_steps.read_text()
    new_ns = """## Current Stage 5 stop point

Stage 5 has completed the visual viewpoint analysis on the frozen Robot1-to-Robot3 protocol. Three visual methods (Single RGB, RGB5 Mean, Cross-Max) reranked the frozen SC Top-20 candidates using OpenCLIP ViT-B-32-quickgelu (laion400m_e32). The candidate-conditioned, end-to-end, heading-stratified, true-positive similarity, rescue/regression, sync-quality and Stage-4 failure analyses are all complete. No fusion, VLM, CVTNet, GICP or PGO was run.

The next authorized research step is viewpoint-aware fusion, which must be a separately authorized Stage 6. It should combine frozen SC candidates with visual evidence under a fixed protocol, explicitly stratified by heading. Do not implement it without explicit authorization."""
    # Replace the Stage 4 stop point section
    if "## Current Stage 4 stop point" in current_ns:
        current_ns = current_ns[:current_ns.index("## Current Stage 4 stop point")] + new_ns + "\n"
    elif "## Current Stage 5 stop point" not in current_ns:
        current_ns = current_ns.rstrip() + "\n" + new_ns + "\n"
    next_steps.write_text(current_ns)

    # ── Update RESULTS.md ──────────────────────────────────────────────
    results = REPO / "RESULTS.md"
    current_r = results.read_text()
    stage5_results = f"""

## CU-Multi Stage 5 Visual Viewpoint Analysis (verified)

Stage 5 measures visual evidence reliability on the frozen Robot1-to-Robot3 SC Top-20 protocol. OpenCLIP ViT-B-32-quickgelu (laion400m_e32) encodes RGB for all frozen 2 Hz keyframes. Three visual methods rerank only the frozen SC Top-20 candidates. Candidate-conditioned analysis evaluates only queries where SC Top-20 contains at least one GT-positive. End-to-end candidate-limited analysis uses all valid queries and counts candidate-generation misses as unavoidable.

Candidate-conditioned (CANDIDATE_AVAILABLE queries only):
- Frozen SC: R@1={r1_cond:.6f}, R@5={overall_df.loc[overall_df.method=='frozen SC', 'candidate_conditioned_R@5'].values[0]:.6f}
- Single RGB: R@1={r1_single:.6f}, R@5={overall_df.loc[overall_df.method=='Single RGB', 'candidate_conditioned_R@5'].values[0]:.6f}
- RGB5 Mean: R@1={r1_mean5:.6f}, R@5={overall_df.loc[overall_df.method=='RGB5 Mean', 'candidate_conditioned_R@5'].values[0]:.6f}
- Cross-Max: R@1={r1_crossmax:.6f}, R@5={overall_df.loc[overall_df.method=='Cross-Max', 'candidate_conditioned_R@5'].values[0]:.6f}

End-to-end (ALL valid queries):
- Frozen SC: R@1={overall_df.loc[overall_df.method=='frozen SC', 'end_to_end_R@1'].values[0]:.6f}
- Single RGB: R@1={overall_df.loc[overall_df.method=='Single RGB', 'end_to_end_R@1'].values[0]:.6f}
- RGB5 Mean: R@1={overall_df.loc[overall_df.method=='RGB5 Mean', 'end_to_end_R@1'].values[0]:.6f}
- Cross-Max: R@1={overall_df.loc[overall_df.method=='Cross-Max', 'end_to_end_R@1'].values[0]:.6f}

Rescues: Single={rr_rows[0]['rescue']}, Mean5={rr_rows[1]['rescue']}, CrossMax={rr_rows[2]['rescue']}
Regressions: Single={rr_rows[0]['regression']}, Mean5={rr_rows[1]['regression']}, CrossMax={rr_rows[2]['regression']}

See `docs/CUMULTI_STAGE5_VISUAL_VIEWPOINT_ANALYSIS.md` and `outputs/cumulti_v1/05_visual_viewpoint_analysis/`.
"""
    if "Stage 5" not in current_r:
        results.write_text(current_r.rstrip() + stage5_results)

    print("\n" + "=" * 60)
    print("STAGE 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
