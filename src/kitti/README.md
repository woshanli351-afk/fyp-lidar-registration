# Reproducible place-recognition pipeline

This directory turns the thesis baseline into a command-line workflow.

## Required KITTI layout

```text
KITTI_ROOT/
  sequences/00/velodyne/000000.bin ...
  poses/00.txt
```

Data and poses are not distributed in this public repository. Obtain KITTI
separately. These historical command examples describe the experiment stages;
use ../../docs/RUNNING.md for this repository's current paths and entrypoints.

## Run the baseline

```bash
python scan_context_baseline.py \
  --kitti-root ../../data_raw/kitti/dataset \
  --pose-file ../../data_raw/kitti/dataset/poses/00.txt \
  --output-dir ../outputs/reproduced
```

The defaults reproduce the thesis protocol: split frame 2500, boundary gap 100,
frame step 5, 5 m positive threshold, and Top-10 retrieval.

## Evaluate an existing candidate file

```bash
python evaluate_candidates.py \
  ../outputs/formal_split2500_gap100_step5_thr5_candidates.csv \
  --output ../outputs/formal_split2500_extended_metrics.json
```

In addition to thesis Recall@K and Precision@K, the evaluator reports MRR,
candidate-pair average precision, and all-query Top-1 rejection diagnostics.
Diagnostic thresholds are selected on the same data and must not be reported as
test performance; a separate validation split is required before deployment.

## Prepare learned re-ranking

Create the exact pair manifest that a CVTNet, BEV, or image encoder should read:

```bash
python prepare_candidate_manifest.py \
  ../outputs/formal_split2500_gap100_step5_thr5_candidates.csv \
  --sequence-dir ../../data_raw/kitti/dataset/sequences/00 \
  --modality velodyne \
  --output ../outputs/candidate_pair_manifest.csv
```

After a model writes `query_frame,candidate_frame,learned_score`, fuse it with
Scan Context without using ground truth during ranking:

```bash
python rerank_candidates.py BASELINE.csv LEARNED_SCORES.csv \
  --alpha 0.7 --output RERANKED.csv --metrics-output RERANKED_METRICS.json
```

## CVTNet transfer re-ranking

The official CVTNet repository publishes an NCLT checkpoint but does not
publish the KITTI-specific checkpoint referenced by its KITTI config.  The
adapter therefore treats the public checkpoint as a cross-dataset transfer
experiment and keeps the public model's required 32x900 input shape.  It also
corrects the official KITTI contribution's stale NCLT binary reader: KITTI
odometry scans are read as Nx4 float32.

Generate inputs only for unique frames in the Scan Context Top-10 list:

```bash
python generate_cvtnet_inputs.py BASELINE.csv \
  --velodyne-dir KITTI_ROOT/sequences/00/velodyne \
  --output-dir OUTPUTS/cvtnet_inputs \
  --cvtnet-root extern/CVTNet
```

Extract descriptors once per unique frame and score candidate pairs:

```bash
python score_cvtnet_candidates.py BASELINE.csv \
  --input-dir OUTPUTS/cvtnet_inputs \
  --weights extern/CVTNet/pretrained_model.pth.tar \
  --cvtnet-root extern/CVTNet \
  --output OUTPUTS/cvtnet_scores.csv
```

`run_cvtnet_autodl.sh` composes preprocessing, CVTNet-only re-ranking, and the
predeclared Scan Context/CVTNet hybrid with alpha 0.7.  Do not select alpha on
KITTI 00 test labels; tune any alternative fusion weight on a validation split.

## ICP/GICP geometric verification

Run the same coarse-to-fine protocol on every reranked Top-1/Top-3 pair.  The
candidate cloud is the source and the query cloud is the target.  Scan Context's
column shift supplies only the initial yaw; ground truth is not used by
registration.

```bash
python geometric_verify_candidates.py HYBRID.csv \
  --velodyne-dir KITTI_ROOT/sequences/00/velodyne \
  --top-k 3 --output OUTPUTS/hybrid_top3_geometry.csv --resume
```

The default protocol uses 0.75 m voxel downsampling, a 3 m coarse stage, and a
1 m fine stage with 30 iterations per stage.  The CSV is flushed after every
pair so an interrupted cloud run can resume safely.

After registration, use the common evaluator on `icp_quality` and
`gicp_quality` (`fitness / (1 + inlier_rmse)`).  Any threshold selected on
Sequence 00 remains diagnostic and cannot be reported as held-out performance.

Select a fixed gate on alternating, non-overlapping query-frame blocks and
evaluate it once on the held-out blocks:

```bash
python select_geometry_threshold.py OUTPUTS/hybrid_top3_geometry.csv \
  --score-column gicp_quality --block-size 250 \
  --output OUTPUTS/gicp_block_split_metrics.json \
  --manifest-output OUTPUTS/gicp_fixed_gate_manifest.csv
```

Odd-numbered 250-frame blocks are used only for threshold selection; even
blocks are used only for the reported test confusion matrix.  The manifest
records the split and fixed decision for each Hybrid rank-1 candidate.

Export accepted GICP measurements as explicit pose-graph loop edges.  The
stored transform maps points from the candidate Velodyne frame into the query
Velodyne frame; ground-truth labels are retained only for offline auditing.

```bash
python export_loop_edges.py OUTPUTS/gicp_fixed_gate_manifest.csv \
  --split test --output OUTPUTS/gicp_test_loop_edges.csv
```

Estimate the two agents' sequential keyframe odometry edges from the same
Velodyne subset before building the complete graph:

```bash
python estimate_keyframe_odometry.py --velodyne-dir KITTI_ROOT/sequences/00/velodyne \
  --output OUTPUTS/keyframe_gicp_odometry.csv --resume
```

Fuse the two odometry chains in a planar pose graph.  Run the held-out loop
edges for the primary evaluation and the complete edge set only as a system
demonstration:

```bash
python optimize_pose_graph_se2.py \
  --odometry OUTPUTS/keyframe_gicp_odometry.csv \
  --loop-edges OUTPUTS/gicp_test_loop_edges.csv \
  --poses KITTI_ROOT/poses/00.txt \
  --calib KITTI_ROOT/sequences/00/calib.txt \
  --output-dir OUTPUTS/pose_graph_test
```

The graph is initialized by one strongest verified inter-robot edge, then
jointly optimized with robust least squares. KITTI poses are loaded only after
optimization to calculate trajectory error; they are not graph constraints.

## Tests

```bash
python -m unittest test_pipeline.py
```
