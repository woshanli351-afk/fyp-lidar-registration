#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WEIGHTS="${2:-$PROJECT_ROOT/extern/CVTNet/pretrained_model.pth.tar}"
PIPELINE="$PROJECT_ROOT/src/kitti"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/kitti/cvtnet_transfer}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/outputs/kitti/baseline/split2500_gap100_step5_thr5_candidates.csv}"
VELODYNE="${VELODYNE:-$PROJECT_ROOT/data/kitti/sequences/00/velodyne}"
INPUTS="$RUN_ROOT/inputs_32x900"
SCORES="$RUN_ROOT/cvtnet_scores.csv"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" "$PIPELINE/generate_cvtnet_inputs.py" "$CANDIDATES" \
  --velodyne-dir "$VELODYNE" \
  --output-dir "$INPUTS" \
  --cvtnet-root "$PROJECT_ROOT/extern/CVTNet"

"$PYTHON_BIN" "$PIPELINE/score_cvtnet_candidates.py" "$CANDIDATES" \
  --input-dir "$INPUTS" \
  --weights "$WEIGHTS" \
  --cvtnet-root "$PROJECT_ROOT/extern/CVTNet" \
  --output "$SCORES" \
  --descriptor-output "$RUN_ROOT/descriptors.npz"

"$PYTHON_BIN" "$PIPELINE/rerank_candidates.py" "$CANDIDATES" "$SCORES" \
  --alpha 0.0 \
  --output "$RUN_ROOT/cvtnet_candidates.csv" \
  --metrics-output "$RUN_ROOT/cvtnet_metrics.json"

"$PYTHON_BIN" "$PIPELINE/rerank_candidates.py" "$CANDIDATES" "$SCORES" \
  --alpha 0.7 \
  --output "$RUN_ROOT/hybrid_a070_candidates.csv" \
  --metrics-output "$RUN_ROOT/hybrid_a070_metrics.json"

echo "CVTNet transfer run completed: $RUN_ROOT"
