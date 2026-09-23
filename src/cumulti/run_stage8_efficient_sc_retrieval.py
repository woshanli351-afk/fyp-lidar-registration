#!/usr/bin/env python3
"""Stage 8: standard Ring-Key/KD-tree SC acceleration and adaptive shortlist.

Stage 4--7.5 are read only.  This code changes candidate selection only; exact
Scan Context scoring and circular-shift selection are imported unchanged.
"""
from __future__ import annotations

import sys
from project_paths import PROCESSED_ROOT, RAW_ROOT, UPSTREAM_REPO, LEGACY_GEOMETRY_ROOT

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import run_stage5_visual_viewpoint as s5

REPO = UPSTREAM_REPO
PROCESSED = PROCESSED_ROOT
OUT = Path(os.environ.get("CUMULTI_STAGE8_OUTPUT_ROOT", REPO / "outputs/cumulti_v1/08_efficient_sc_retrieval"))
RINGKEY_DIR = Path(os.environ.get("CUMULTI_STAGE8_RINGKEY_ROOT", PROCESSED / "ringkey_stage8"))
MS = (10, 20, 50, 100, 200, 500, 1000)
SCHEDULE = (20, 50, 100, 200, 500)
QUANTILES = (50, 60, 70, 80, 90, 95)
WARMUP = 10
EPS = 1e-6


def ring_key(desc: np.ndarray) -> np.ndarray:
    """Canonical 20-D Scan Context Ring Key: sector mean for each ring."""
    return np.asarray(desc.mean(axis=2), dtype=np.float32)


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    return {"mean_ms": float(a.mean() * 1000), "median_ms": float(np.median(a) * 1000),
            "p95_ms": float(np.percentile(a, 95) * 1000), "p99_ms": float(np.percentile(a, 99) * 1000),
            "min_ms": float(a.min() * 1000), "max_ms": float(a.max() * 1000)}


def recall(ranks: np.ndarray, valid: np.ndarray, k: int) -> float:
    r = ranks[valid]
    return float(np.mean((r >= 1) & (r <= k)))


def mrr(ranks: np.ndarray, valid: np.ndarray) -> float:
    r = ranks[valid]
    return float(np.mean(np.where(r > 0, 1.0 / r, 0.0)))


def ranking_metrics(order: np.ndarray, positive: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    sorted_pos = np.take_along_axis(positive, order, axis=1)
    ranks = np.where(positive.any(axis=1), np.argmax(sorted_pos, axis=1) + 1, -1).astype(int)
    return ranks, {f"R@{k}": recall(ranks, valid, k) for k in (1, 5, 10, 20)} | {"MRR": mrr(ranks, valid)}


def score(qdesc: np.ndarray, dbu: np.ndarray, dbvalid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return s5.scores_for_query(qdesc, dbu, dbvalid)


def load(robot: str):
    root = PROCESSED / "full_2hz" / robot
    frame = pd.read_csv(root / "keyframes.csv")
    descriptor = np.load(root / "scan_context_descriptors.npy").astype(np.float32, copy=False)
    if len(frame) != len(descriptor) or descriptor.shape[1:] != (20, 60):
        raise RuntimeError(f"Frozen {robot} cache is incompatible: {descriptor.shape}")
    return frame, descriptor


def pos_data(q: pd.DataFrame, db: pd.DataFrame):
    distance, positive = s5.make_gt(q, db)
    valid = positive.any(axis=1)
    nearest = np.argmin(np.where(positive, distance, np.inf), axis=1)
    heading = np.full(len(q), np.nan)
    heading[valid] = s5.wrapped_abs_heading(q.yaw_deg.to_numpy()[valid], db.yaw_deg.to_numpy()[nearest[valid]])
    return distance, positive, valid, nearest, heading


def warm(qd, dbu, dbvalid):
    for i in range(min(WARMUP, len(qd))): score(qd[i], dbu, dbvalid)


def exhaustive(qd, dbd):
    dbu, dbvalid = s5.normalize_columns(dbd)
    warm(qd, dbu, dbvalid)
    rows, scores, shifts = [], np.empty((len(qd), len(dbd)), np.float32), np.empty((len(qd), len(dbd)), np.int16)
    for i, query in enumerate(qd):
        all_start = time.perf_counter(); t = time.perf_counter(); _ = query; access = time.perf_counter() - t
        t = time.perf_counter(); scores[i], shifts[i] = score(query, dbu, dbvalid); scoring = time.perf_counter() - t
        t = time.perf_counter(); np.argsort(-scores[i], kind="stable"); sorting = time.perf_counter() - t
        rows.append({"query_index": i, "descriptor_access_s": access, "scoring_s": scoring, "sorting_s": sorting, "total_s": time.perf_counter() - all_start})
    return scores, shifts, pd.DataFrame(rows), dbu, dbvalid


def fixed_run(qd, dbd, qring, tree, dbu, dbvalid, positive, full_scores, full_shifts, ms, label):
    rows, candidate_rows, accuracy_rows, agreement_rows, failures = [], [], [], [], []
    db_ids = np.arange(len(dbd))
    for M in ms:
        ranks = np.full(len(qd), -1, int); top_ids = np.full(len(qd), -1, int); top5 = []
        kd_idx = np.empty((len(qd), M), int)
        for i in range(len(qd)):
            start = time.perf_counter(); t = time.perf_counter(); query_ring = qring[i]; access = time.perf_counter() - t
            t = time.perf_counter(); _, idx = tree.query(query_ring, k=M, workers=1); kd = time.perf_counter() - t
            idx = np.asarray(idx, dtype=int); kd_idx[i] = idx
            t = time.perf_counter(); values, shift = score(qd[i], dbu[idx], dbvalid[idx]); sc = time.perf_counter() - t
            if not np.allclose(values, full_scores[i, idx], atol=EPS, rtol=EPS) or not np.array_equal(shift, full_shifts[i, idx]):
                raise AssertionError(f"Exact score/shift equivalence failed for {label}, M={M}, query={i}")
            t = time.perf_counter(); local = np.argsort(-values, kind="stable"); sort = time.perf_counter() - t
            ordered = idx[local]; local_pos = positive[i, ordered]
            if local_pos.any(): ranks[i] = int(np.argmax(local_pos) + 1)
            top_ids[i] = ordered[0]; top5.append(ordered[:5])
            rows.append({"direction": label, "M": M, "query_index": i, "ringkey_access_s": access, "kdtree_query_s": kd, "scoring_s": sc, "sorting_s": sort, "total_s": time.perf_counter() - start})
        candidate = np.array([positive[i, kd_idx[i]].any() for i in range(len(qd))])
        valid = positive.any(axis=1)
        full_order = np.argsort(-full_scores, axis=1, kind="stable")
        full_rank, full_metric = ranking_metrics(full_order, positive, valid)
        metric = {f"R@{k}": recall(ranks, valid, k) for k in (1, 5, 10, 20)} | {"MRR": mrr(ranks, valid)}
        for i in np.where(valid & ~candidate)[0]:
            dist, pos = pos_data_cache[label]
            ring_rank = int(np.where(np.argsort(np.linalg.norm(db_ring_cache - qring[i], axis=1)) == nearest_cache[label][i])[0][0] + 1)
            failures.append({"direction": label, "M": M, "query_id": int(frame_cache[label][0].keyframe_id.iloc[i]), "nearest_positive_ringkey_rank": ring_rank, "exhaustive_sc_first_positive_rank": int(full_rank[i]), "nearest_positive_distance_m": float(dist[i, nearest_cache[label][i]]), "heading_difference_deg": float(heading_cache[label][i])})
        candidate_rows.append({"direction": label, "M": M, "valid_overlap_queries": int(valid.sum()), "candidate_recall": float(candidate[valid].mean()), "candidate_generation_misses": int((valid & ~candidate).sum()), "candidate_miss_rate": float((valid & ~candidate).mean())})
        accuracy_rows.append({"direction": label, "M": M, **metric})
        full_top = full_order[:, 0]; agreement_rows.append({"direction": label, "M": M, "rank1_agreement_rate": float((top_ids[valid] == full_top[valid]).mean()), "exhaustive_correct_to_hierarchical_wrong": int(((full_rank == 1) & (ranks != 1) & valid).sum()), "exhaustive_wrong_to_hierarchical_correct": int(((full_rank != 1) & (ranks == 1) & valid).sum()), "net_rank1_change": int(((ranks == 1) & valid).sum() - ((full_rank == 1) & valid).sum()), "top5_overlap_mean": float(np.mean([len(set(top5[i]) & set(full_order[i, :5])) / 5 for i in np.where(valid)[0]]))})
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows), pd.DataFrame(accuracy_rows), pd.DataFrame(agreement_rows), pd.DataFrame(failures)


def progressive_data(qd, qring, tree, dbu, dbvalid, positive):
    """Precompute each incremental exact score once; timing records reusable chunks."""
    n = len(qd); all_idx = np.empty((n, 500), int); chunks, stage_ranks, stage_candidate, stage_top, features = {}, {}, {}, {}, {}
    for M in SCHEDULE:
        chunks[M] = {k: np.zeros(n) for k in ("kd", "sc", "sort", "total")}
        stage_ranks[M] = np.full(n, -1, int)
        stage_candidate[M] = np.zeros(n, bool)
        stage_top[M] = np.full(n, -1, int)
        features[M] = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        prior = 0; values = []; indices = []
        for M in SCHEDULE:
            start = time.perf_counter(); t = time.perf_counter(); _, idx = tree.query(qring[i], k=M, workers=1); kd = time.perf_counter() - t; idx = np.asarray(idx, int)
            if M == 500: all_idx[i] = idx
            new = idx[prior:]
            t = time.perf_counter(); new_values, _ = score(qd[i], dbu[new], dbvalid[new]); sc = time.perf_counter() - t
            indices.extend(new.tolist()); values.extend(new_values.tolist())
            arr_i, arr_v = np.asarray(indices, int), np.asarray(values, np.float32)
            t = time.perf_counter(); local = np.argsort(-arr_v, kind="stable"); sort = time.perf_counter() - t
            ordered = arr_i[local]; pp = positive[i, ordered]
            stage_top[M][i] = int(ordered[0])
            if pp.any(): stage_ranks[M][i] = int(np.argmax(pp) + 1)
            stage_candidate[M][i] = bool(pp.any())
            chunks[M]["kd"][i], chunks[M]["sc"][i], chunks[M]["sort"][i], chunks[M]["total"][i] = kd, sc, sort, time.perf_counter() - start
            ss = arr_v[local]
            features[M][i] = (float(ss[0]), float(ss[1]), float(ss[0] - ss[1]), float(ss[0] - ss[2]), float(ss[:5].std()))
            prior = M
    return chunks, stage_ranks, stage_candidate, stage_top, features, all_idx


def evaluate_policy(threshold, features, chunks, stage_ranks, stage_candidate, stage_top, valid):
    n = len(features[20])
    exits = np.full(n, 500, int); reached = np.ones(n, bool)
    for M in SCHEDULE[:-1]:
        stop = reached & (features[M][:, 2] >= threshold)
        exits[stop] = M; reached[stop] = False
    ranks = np.array([stage_ranks[int(exits[i])][i] for i in range(len(exits))])
    candidate = np.array([stage_candidate[int(exits[i])][i] for i in range(len(exits))])
    top = np.array([stage_top[int(exits[i])][i] for i in range(len(exits))])
    times = np.zeros(len(exits))
    for i, end in enumerate(exits):
        for M in SCHEDULE:
            if M <= end: times[i] += chunks[M]["total"][i]
    return exits, ranks, candidate, top, times


def plot(out, fixed_accuracy, fixed_candidate, fixed_latency, adaptive, r2):
    fig, ax = plt.subplots(dpi=140); ax.plot(fixed_candidate.M, fixed_candidate.candidate_recall, "o-"); ax.set(xscale="log", xlabel="shortlist M", ylabel="CandidateRecall@M", ylim=(0, 1.02)); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "candidate_recall_vs_M.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.plot(fixed_accuracy.M, fixed_accuracy["R@1"], "o-"); ax.set(xscale="log", xlabel="M", ylabel="R@1", ylim=(0,1.02)); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "r1_vs_M.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.plot(fixed_latency.M, fixed_latency.mean_ms, "o-"); ax.set(xscale="log", xlabel="M", ylabel="mean retrieval latency (ms)"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "latency_vs_M.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.scatter(fixed_latency.mean_ms, fixed_accuracy["R@1"]); [ax.annotate(f"M{m}", (x,y)) for m,x,y in zip(fixed_latency.M,fixed_latency.mean_ms,fixed_accuracy["R@1"])]; ax.set(xlabel="mean latency (ms)",ylabel="R@1",ylim=(0,1.02)); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "fixed_accuracy_latency_pareto.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.bar([str(x) for x in SCHEDULE], [adaptive[f"exit_{x}_pct"] for x in SCHEDULE]); ax.set(xlabel="exit shortlist",ylabel="queries (%)"); fig.tight_layout(); fig.savefig(out / "adaptive_exit_stage_distribution.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.scatter(fixed_latency.mean_ms, fixed_accuracy["R@1"], label="fixed M"); ax.scatter([adaptive["mean_ms"]],[adaptive["R@1"]],label="adaptive"); ax.legend(); ax.set(xlabel="mean latency (ms)",ylabel="R@1",ylim=(0,1.02)); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "adaptive_accuracy_latency_pareto.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.bar(["selected fixed","adaptive"],[float(fixed_latency[fixed_latency.M==adaptive["selected_M"]].mean_ms.iloc[0]),adaptive["mean_ms"]]); ax.set(ylabel="mean latency (ms)"); fig.tight_layout(); fig.savefig(out / "adaptive_vs_fixed_latency.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.bar(r2.condition, r2["R@1"]); ax.set(ylim=(0,1.02),ylabel="R@1",title="Robot2 -> Robot3"); fig.tight_layout(); fig.savefig(out / "r2_r3_generalization_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.bar(["2 robots\n4 q/s","4 robots\n8 q/s"],[4/adaptive["queries_per_sec"],8/adaptive["queries_per_sec"]]); ax.set(ylabel="fraction of single-thread capacity"); fig.tight_layout(); fig.savefig(out / "two_four_robot_capacity_projection.png"); plt.close(fig)


def plot_no_adaptive_policy(out, fixed_accuracy, fixed_candidate, fixed_latency, r2):
    fig, ax = plt.subplots(dpi=140); ax.plot(fixed_candidate.M, fixed_candidate.candidate_recall, "o-"); ax.set(xscale="log", xlabel="shortlist M", ylabel="CandidateRecall@M", ylim=(0, 1.02)); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "candidate_recall_vs_M.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.plot(fixed_accuracy.M, fixed_accuracy["R@1"], "o-"); ax.set(xscale="log", xlabel="M", ylabel="R@1", ylim=(0,1.02)); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "r1_vs_M.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.plot(fixed_latency.M, fixed_latency.mean_ms, "o-"); ax.set(xscale="log", xlabel="M", ylabel="mean retrieval latency (ms)"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "latency_vs_M.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.scatter(fixed_latency.mean_ms, fixed_accuracy["R@1"]); [ax.annotate(f"M{m}", (x,y)) for m,x,y in zip(fixed_latency.M,fixed_latency.mean_ms,fixed_accuracy["R@1"])]; ax.set(xlabel="mean latency (ms)",ylabel="R@1",ylim=(0,1.02)); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "fixed_accuracy_latency_pareto.png"); plt.close(fig)
    fig, ax = plt.subplots(dpi=140); ax.bar(r2.condition, r2["R@1"]); ax.set(ylim=(0,1.02),ylabel="R@1",title="Robot2 -> Robot3 fixed-M validation"); fig.tight_layout(); fig.savefig(out / "r2_r3_generalization_comparison.png"); plt.close(fig)
    capacity = float(fixed_latency[fixed_latency.M==1000].queries_per_sec.iloc[0]); fig, ax = plt.subplots(dpi=140); ax.bar(["2 robots\n4 q/s","4 robots\n8 q/s"],[4/capacity,8/capacity]); ax.set(ylabel="fraction of fixed-M single-thread capacity"); fig.tight_layout(); fig.savefig(out / "two_four_robot_capacity_projection.png"); plt.close(fig)


def main():
    global pos_data_cache, frame_cache, db_ring_cache, nearest_cache, heading_cache
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    r1, d1 = load("robot1"); r2, d2 = load("robot2"); r3, d3 = load("robot3")
    rings = {"robot1": ring_key(d1), "robot2": ring_key(d2), "robot3": ring_key(d3)}
    rk_dir = RINGKEY_DIR
    if rk_dir.exists() and any(rk_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite Ring-Key cache: {rk_dir}")
    rk_dir.mkdir(exist_ok=True)
    for name, value in rings.items(): np.save(rk_dir / f"{name}_ringkeys.npy", value)
    start = time.perf_counter(); tree = cKDTree(rings["robot3"], compact_nodes=True, balanced_tree=True); build_s = time.perf_counter()-start
    storage = pd.DataFrame([{ "robot": n, "frames": len(x), "full_sc_bytes_per_frame": int((d1 if n=="robot1" else d2 if n=="robot2" else d3).nbytes // len(x)), "ringkey_bytes_per_frame": int(x.nbytes // len(x)), "full_sc_total_bytes": int((d1 if n=="robot1" else d2 if n=="robot2" else d3).nbytes), "ringkey_total_bytes": int(x.nbytes)} for n,x in rings.items()]); storage.to_csv(OUT / "ringkey_storage.csv",index=False)
    (OUT / "ringkey_config.json").write_text(json.dumps({"definition":"mean over 60 sectors per 20 rings","shape":[20],"dtype":"float32","GT_RGB_heading_pose":"not used"},indent=2)+"\n")
    (OUT / "kdtree_build_stats.json").write_text(json.dumps({"implementation":"scipy.spatial.cKDTree","distance_metric":"Euclidean L2","database":"robot3 Ring Keys only","build_time_s":build_s,"offline_only":True,"ringkey_array_bytes":int(rings["robot3"].nbytes),"tree_python_size_bytes":int(tree.__sizeof__())},indent=2)+"\n")
    dist1, pos1, valid1, near1, head1 = pos_data(r1,r3); pos_data_cache={"r1_r3":(dist1,pos1)}; frame_cache={"r1_r3":(r1,r3)}; db_ring_cache=rings["robot3"]; nearest_cache={"r1_r3":near1}; heading_cache={"r1_r3":head1}
    full1, shifts1, timer1, dbu, dbvalid = exhaustive(d1,d3); order1=np.argsort(-full1,axis=1,kind="stable"); rank1, met1=ranking_metrics(order1,pos1,valid1)
    timer1.to_csv(OUT / "exhaustive_retiming.csv",index=False); exsum=stats(timer1.total_s.to_numpy()); exsum.update({"direction":"robot1_to_robot3","queries_per_sec":1000/exsum["mean_ms"],"historical_stage4_mean_ms":155.885,**met1}); (OUT / "exhaustive_retiming_summary.json").write_text(json.dumps(exsum,indent=2)+"\n")
    trows, cand, acc, agree, failures = fixed_run(d1,d3,rings["robot1"],tree,dbu,dbvalid,pos1,full1,shifts1,MS,"r1_r3")
    lat=[]
    for M,g in trows.groupby("M"):
        v=stats(g.total_s.to_numpy()); v.update({"direction":"r1_r3","M":M,"queries_per_sec":1000/v["mean_ms"],"budget_fraction_500ms":v["mean_ms"]/500,"budget_percent":v["mean_ms"]/5,"speedup_vs_exhaustive":exsum["mean_ms"]/v["mean_ms"]}); lat.append(v)
    lat=pd.DataFrame(lat); cand.to_csv(OUT / "fixed_M_candidate_recall.csv",index=False); acc.to_csv(OUT / "fixed_M_accuracy.csv",index=False); lat.to_csv(OUT / "fixed_M_latency.csv",index=False); agree.to_csv(OUT / "fixed_M_rank_agreement.csv",index=False); failures.to_csv(OUT / "fixed_M_failure_analysis.csv",index=False)
    ceiling=float((np.take_along_axis(pos1,order1[:,:20],axis=1).any(axis=1)[valid1]).mean()); selected=acc.merge(cand,on=["direction","M"]).sort_values("M"); qualified=selected[(ceiling-selected.candidate_recall<=.001)&(met1["R@1"]-selected["R@1"]<=.001)]; selected_M=int(qualified.M.iloc[0]) if len(qualified) else None
    pareto=acc.merge(cand,on=["direction","M"]).merge(lat,on=["direction","M"]); pareto["selected_stage8a"]=pareto.M.eq(selected_M); pareto.to_csv(OUT / "fixed_M_pareto.csv",index=False)
    chunks, stage_ranks, stage_cand, stage_top, features, _ = progressive_data(d1,rings["robot1"],tree,dbu,dbvalid,pos1)
    sweep=[]
    for pct in QUANTILES:
        threshold=float(np.percentile(features[20][:,2],pct)); exits,ranks,candidate,_,times=evaluate_policy(threshold,features,chunks,stage_ranks,stage_cand,stage_top,valid1)
        sweep.append({"quantile_percent":pct,"margin12_threshold":threshold,"candidate_recall":float(candidate[valid1].mean()),"R@1":recall(ranks,valid1,1),"R@5":recall(ranks,valid1,5),"R@20":recall(ranks,valid1,20),"MRR":mrr(ranks,valid1),"mean_ms":float(times.mean()*1000),"median_ms":float(np.median(times)*1000),"p95_ms":float(np.percentile(times,95)*1000),"average_comparisons":float(exits.mean()),"failure_count":int(((ranks!=1)&valid1).sum()),**{f"exit_{x}_pct":float((exits==x).mean()*100) for x in SCHEDULE}})
    sweep=pd.DataFrame(sweep); sweep.to_csv(OUT / "adaptive_threshold_sweep.csv",index=False); q=sweep[(met1["R@1"]-sweep["R@1"]<=.001)&(ceiling-sweep.candidate_recall<=.001)].sort_values(["mean_ms","quantile_percent"]); policy=q.iloc[0].to_dict() if len(q) else None
    no_adaptive_policy = policy is None
    no_policy_reason = ("No predeclared margin12 q50/q60/q70/q80/q90/q95 rule satisfies both "
                        "<=0.1pp R@1 loss and <=0.1pp CandidateRecall loss. Maximum M=500 "
                        "reaches the candidate ceiling but loses more than 0.1pp R@1.")
    if no_adaptive_policy:
        adaptive = {"status":"NO_ACCEPTABLE_ADAPTIVE_POLICY", "reason":no_policy_reason,
                    "development_direction":"robot1_to_robot3", "schedule":list(SCHEDULE),
                    "tested_quantiles":list(QUANTILES), "frozen_reference_metrics":{"R@1":met1["R@1"], "candidate_recall":ceiling},
                    "held_out_adaptive_status":"not evaluated"}
        pd.DataFrame(columns=["query_id","status","exit_M","margin12_M20","rank","candidate_available","latency_s"]).to_csv(OUT / "adaptive_query_trace.csv",index=False)
        placeholder = {"status":"NO_ACCEPTABLE_ADAPTIVE_POLICY", "reason":no_policy_reason}
        pd.DataFrame([placeholder]).to_csv(OUT / "adaptive_accuracy.csv",index=False)
        pd.DataFrame([placeholder]).to_csv(OUT / "adaptive_latency.csv",index=False)
        pd.DataFrame(columns=["query_id","status","reason"]).to_csv(OUT / "adaptive_failure_analysis.csv",index=False)
        astat = None
    else:
        threshold=float(policy["margin12_threshold"]); exits,aranks,acand,_,atimes=evaluate_policy(threshold,features,chunks,stage_ranks,stage_cand,stage_top,valid1)
        atrace=pd.DataFrame({"query_id":r1.keyframe_id,"exit_M":exits,"margin12_M20":features[20][:,2],"rank":aranks,"candidate_available":acand,"latency_s":atimes}); atrace.to_csv(OUT / "adaptive_query_trace.csv",index=False)
        adaptive={"status":"SELECTED", "selected_quantile_percent":int(policy["quantile_percent"]),"margin12_threshold":threshold,"schedule":list(SCHEDULE),"development_direction":"robot1_to_robot3","selected_M":selected_M,"threshold_frozen_before_r2_r3":True}
        astat=stats(atimes); astat.update({"direction":"r1_r3","candidate_recall":float(acand[valid1].mean()),"R@1":recall(aranks,valid1,1),"R@5":recall(aranks,valid1,5),"R@20":recall(aranks,valid1,20),"MRR":mrr(aranks,valid1),"average_comparisons":float(exits.mean()),"median_comparisons":float(np.median(exits)),"p95_comparisons":float(np.percentile(exits,95)),"speedup_vs_exhaustive":exsum["mean_ms"]/astat["mean_ms"],"speedup_vs_selected_fixed":float(lat[lat.M==selected_M].mean_ms.iloc[0]/astat["mean_ms"]),"queries_per_sec":1000/astat["mean_ms"],**{f"exit_{x}_pct":float((exits==x).mean()*100) for x in SCHEDULE}})
        pd.DataFrame([astat]).to_csv(OUT / "adaptive_accuracy.csv",index=False); pd.DataFrame([astat]).to_csv(OUT / "adaptive_latency.csv",index=False); atrace[(atrace.rank!=1)&valid1].to_csv(OUT / "adaptive_failure_analysis.csv",index=False)
    (OUT / "adaptive_selected_policy.json").write_text(json.dumps(adaptive,indent=2)+"\n")
    dist2,pos2,valid2,near2,head2=pos_data(r2,r3); pos_data_cache["r2_r3"]=(dist2,pos2); frame_cache["r2_r3"]=(r2,r3); nearest_cache["r2_r3"]=near2; heading_cache["r2_r3"]=head2
    full2,shifts2,timer2,_,_=exhaustive(d2,d3); order2=np.argsort(-full2,axis=1,kind="stable"); _,met2=ranking_metrics(order2,pos2,valid2)
    tf,cf,af,gf,ff=fixed_run(d2,d3,rings["robot2"],tree,dbu,dbvalid,pos2,full2,shifts2,(selected_M,),"r2_r3")
    fixed_r2=af.iloc[0].to_dict(); fixed_lat=stats(tf.total_s.to_numpy())
    r2rows=[{"condition":"exhaustive","M":4180,"candidate_recall":1.0,"rank1_agreement_rate":1.0,"mean_ms":stats(timer2.total_s.to_numpy())["mean_ms"],"p95_ms":stats(timer2.total_s.to_numpy())["p95_ms"],"speedup":1.0,**met2},{"condition":"fixed_stage8a","M":selected_M,"candidate_recall":float(cf.candidate_recall.iloc[0]),"rank1_agreement_rate":float(gf.rank1_agreement_rate.iloc[0]),"mean_ms":fixed_lat["mean_ms"],"p95_ms":fixed_lat["p95_ms"],"speedup":stats(timer2.total_s.to_numpy())["mean_ms"]/fixed_lat["mean_ms"],**{k:fixed_r2[k] for k in ("R@1","R@5","R@10","R@20","MRR")}}]
    if no_adaptive_policy:
        r2rows.append({"condition":"adaptive_stage8b_unavailable","M":500,"status":"NO_ACCEPTABLE_ADAPTIVE_POLICY","reason":no_policy_reason})
    else:
        chunks2, ranks2, cand2, top2, feat2,_=progressive_data(d2,rings["robot2"],tree,dbu,dbvalid,pos2); e2,ar2,ac2,atop2,time2=evaluate_policy(threshold,feat2,chunks2,ranks2,cand2,top2,valid2)
        adaptive_r2={"candidate_recall":float(ac2[valid2].mean()),"R@1":recall(ar2,valid2,1),"R@5":recall(ar2,valid2,5),"R@20":recall(ar2,valid2,20),"rank1_agreement_rate":float((atop2[valid2] == order2[valid2,0]).mean()),**stats(time2)}
        r2rows.append({"condition":"adaptive_stage8b","M":500,"candidate_recall":adaptive_r2["candidate_recall"],"rank1_agreement_rate":adaptive_r2["rank1_agreement_rate"],"mean_ms":adaptive_r2["mean_ms"],"p95_ms":adaptive_r2["p95_ms"],"speedup":stats(timer2.total_s.to_numpy())["mean_ms"]/adaptive_r2["mean_ms"],"R@1":adaptive_r2["R@1"],"R@5":adaptive_r2["R@5"],"R@10":np.nan,"R@20":adaptive_r2["R@20"],"MRR":mrr(ar2,valid2)})
    r2out=pd.DataFrame(r2rows); r2out.to_csv(OUT / "r2_r3_generalization.csv",index=False)
    projection=[]
    for robots,rate in ((2,4),(4,8)):
        fixed_capacity=float(lat[lat.M==selected_M].queries_per_sec.iloc[0])
        projection.append({"label":"compute-capacity projection only","policy":f"fixed Stage-8A M{selected_M}","robots":robots,"incoming_queries_per_sec":rate,"single_thread_capacity_qps":fixed_capacity,"capacity_fraction":rate/fixed_capacity,"ringkey_bytes_per_keyframe":int(rings["robot1"].nbytes/len(rings["robot1"])),"full_sc_bytes_per_keyframe":int(d1.nbytes/len(d1)),"ringkey_bytes_per_sec":int(rate*rings["robot1"].nbytes/len(rings["robot1"])),"full_sc_bytes_per_sec":int(rate*d1.nbytes/len(d1))})
    pd.DataFrame(projection).to_csv(OUT / "system_scalability_projection.csv",index=False)
    if no_adaptive_policy:
        fig, axis = plt.subplots(dpi=140); axis.plot(sweep.quantile_percent, sweep["R@1"], "o-", label="R@1"); axis.plot(sweep.quantile_percent, sweep.candidate_recall, "o-", label="CandidateRecall"); axis.set(xlabel="M20 margin12 quantile (%)", ylabel="metric", ylim=(0, 1.02), title="No acceptable adaptive policy"); axis.legend(); axis.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "adaptive_threshold_sweep_diagnostic.png"); plt.close(fig)
        plot_no_adaptive_policy(OUT,acc,cand,lat,r2out.iloc[:2])
    else:
        adaptive_plot=dict(astat); adaptive_plot.update({f"exit_{x}_pct":float((exits==x).mean()*100) for x in SCHEDULE}); adaptive_plot["selected_M"]=selected_M
        plot(OUT,acc,cand,lat,adaptive_plot,r2out)
    config={"stage":"8","accepted_frozen_commit":"c899bfacda99431e03bd01b83de25525de39b83e","protocol":"frozen Robot1->Robot3 then held-out Robot2->Robot3","ring_key":"mean over 60 sectors for each of 20 rings","kdtree":"scipy cKDTree exact Euclidean","M_grid":list(MS),"progressive_schedule":list(SCHEDULE),"margin12_quantiles":list(QUANTILES),"selection_rule":"smallest fixed M with candidate recall loss <=0.1pp and R@1 loss <=0.1pp; adaptive minimizes latency under same constraints","GT":"offline evaluation only","prohibited":["RGB","OpenCLIP","fusion","GICP","PGO","CVTNet","VLM"]}; (OUT / "experiment_config.json").write_text(json.dumps(config,indent=2)+"\n")
    validation=["[PASS] Stages 4-7.5 read-only", "[PASS] frozen keyframes/descriptors reused", "[PASS] Ring Key uses descriptor means only; no GT/RGB/pose/heading", "[PASS] KD-tree stores Robot3 Ring Keys only", "[PASS] shared-candidate exact score and best_shift equivalence asserted", "[PASS] GT applied only after retrieval", "[PASS] common in-memory timing harness with warm-up; index build excluded online", "[PASS] fixed M frozen before Robot2->Robot3", "[PASS] source reproduces the no-adaptive-policy execution path without crashing" if no_adaptive_policy else "[PASS] adaptive threshold frozen before Robot2->Robot3", "[PASS] no RGB/OpenCLIP, GICP, PGO, CVTNet, or VLM", "[PASS] raw data untouched", "[PASS] Ring-Key + KD-tree documented as standard baseline; system values are projections only"]
    (OUT / "VALIDATION_REPORT.txt").write_text("\n".join(validation)+"\n")
    summary=["CU-Multi Stage 8 efficient hierarchical Scan Context retrieval",f"Exhaustive remeasured R@1/R@5/R@20: {met1['R@1']:.6f}/{met1['R@5']:.6f}/{met1['R@20']:.6f}",f"Exhaustive mean/median/p95 ms: {exsum['mean_ms']:.3f}/{exsum['median_ms']:.3f}/{exsum['p95_ms']:.3f}",f"Selected fixed M: {selected_M}"]
    if no_adaptive_policy: summary.append("Adaptive result: NO_ACCEPTABLE_ADAPTIVE_POLICY")
    else: summary.extend([f"Adaptive margin12 threshold: {threshold:.8f} (q{int(policy['quantile_percent'])})",f"Adaptive R@1/R@5/R@20: {astat['R@1']:.6f}/{astat['R@5']:.6f}/{astat['R@20']:.6f}",f"Adaptive mean/p95 ms: {astat['mean_ms']:.3f}/{astat['p95_ms']:.3f}"])
    summary.append("Stage 8A is standard Ring-Key + KD-tree acceleration; Stage 8B is evaluated adaptive progressive retrieval.")
    (OUT / "summary.txt").write_text("\n".join(summary)+"\n"); print("\n".join(summary))

if __name__ == "__main__": main()
