#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE="$PROJECT_ROOT/src/kitti"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/outputs/kitti/cvtnet_transfer/hybrid_a070_candidates.csv}"
VELODYNE="${VELODYNE:-$PROJECT_ROOT/data/kitti/sequences/00/velodyne}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/kitti/geometric_verification}"
RESULTS="$RUN_ROOT/hybrid_top3_icp_gicp.csv"

mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" -c 'import open3d; print(open3d.__version__)'

"$PYTHON_BIN" "$PIPELINE/geometric_verify_candidates.py" "$CANDIDATES" \
  --velodyne-dir "$VELODYNE" \
  --top-k 3 \
  --output "$RESULTS" \
  --resume

"$PYTHON_BIN" "$PIPELINE/evaluate_candidates.py" "$RESULTS" \
  --score-column icp_quality --ks 1 3 \
  --output "$RUN_ROOT/icp_metrics.json"

"$PYTHON_BIN" "$PIPELINE/evaluate_candidates.py" "$RESULTS" \
  --score-column gicp_quality --ks 1 3 \
  --output "$RUN_ROOT/gicp_metrics.json"

echo "Geometric verification completed: $RUN_ROOT"
