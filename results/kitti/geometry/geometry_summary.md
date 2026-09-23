# ICP/GICP geometric-verification summary

## Protocol

- Dataset: KITTI Odometry Sequence 00.
- Candidate source: Hybrid reranking (`alpha=0.7`) of the Scan Context Top-10 list.
- Verified candidates: Top-3 for each of 481 queries (1,443 pairs).
- Registration direction: candidate scan to query scan.
- Initial yaw: Scan Context sector shift (`360 * best_shift / 60`).
- Preprocessing: 0.75 m voxel downsampling, 2--50 m radial crop, and -3--5 m height crop.
- Registration: point-to-point ICP and Generalized ICP (GICP), using 3 m coarse and 1 m fine correspondence thresholds with 30 iterations per stage.
- Geometric quality score: `fitness / (1 + inlier_rmse)`.
- Positive loop-closure threshold: ground-truth distance at most 5 m.

## Retrieval ranking before geometry

The geometry stage verifies the existing Hybrid Top-3 candidates; it does not
change whether a positive is present in the candidate set.

| Metric | Result |
|---|---:|
| Total queries | 481 |
| Queries with at least one ground-truth match | 158 |
| Queries without overlap | 323 |
| Recall@1 on valid queries | 94.30% |
| Recall@3 on valid queries | 96.20% |
| Valid queries with no positive in Top-3 | 6 |

## Geometric discrimination

| Method | Candidate-pair AP | Candidate precision | Candidate recall | Candidate FPR | All-query Top-1 AP | Top-1 precision | Top-1 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ICP | 98.07% | 99.25% | 93.31% | 0.17% | 99.97% | 100.00% | 99.33% |
| GICP | 99.04% | 95.17% | 97.18% | 1.21% | 100.00% | 100.00% | 100.00% |

At the diagnostic Top-1 thresholds, ICP accepts 148 true loop closures with
zero false positives and misses one; GICP accepts all 149 true loop closures
with zero false positives. GICP has higher pair-level AP and recall, whereas
ICP gives higher pair-level precision and a lower false-positive rate at its
selected threshold.

## Reporting limitation

The quality thresholds (ICP 0.59865 and GICP 0.59770 for Top-1) were selected
on the same evaluated data. They are useful for diagnosing separability, but
must not be reported as held-out deployment performance. A deployable gate
requires a separate validation split or a fixed threshold chosen before final
testing.

## Fixed-threshold block split

To remove same-data threshold selection, query frames were divided into
non-overlapping 250-frame blocks. Odd-numbered blocks were used to select the
GICP threshold and even-numbered blocks were held out for one final test. The
validation-only threshold was 0.597696. It achieved 76 TP, 0 FP, and 0 FN on
231 validation queries, followed by 73 TP, 0 FP, and 0 FN on 250 held-out test
queries (73 positive and 177 negative Top-1 candidates). This block split is
the result that should be used when reporting the fixed geometric gate.

## Artifacts

- `hybrid_top3_icp_gicp.csv`: per-pair retrieval scores, labels, ICP/GICP
  metrics, transformations, and runtime.
- `icp_metrics.json`: complete ICP evaluation.
- `gicp_metrics.json`: complete GICP evaluation.
- `smoke_top3.csv`: three-pair preflight run.
- `gicp_block_split_metrics.json`: validation-selected threshold and held-out
  block metrics.
- `gicp_fixed_gate_manifest.csv`: all 481 rank-1 decisions with explicit split.
- `gicp_test_loop_edges.csv`: 73 accepted loop edges from held-out test blocks.
- `gicp_all_loop_edges.csv`: 149 accepted edges for the complete system demo.

## Recommended next step

Use GICP as the primary geometric-verification method and ICP as an ablation.
Choose the acceptance threshold on a validation split, convert accepted GICP
transformations into loop-closure edges, add them to the odometry pose graph,
and report trajectory error before and after graph optimization.

## Pose-graph result

Sequential GICP on the 870 sampled keyframes produced 868 odometry edges. The
primary graph used the 73 loop closures selected only from the held-out test
blocks, for 870 nodes and 941 total constraints. One strongest test loop was
used to establish the initial Robot B-to-A frame alignment; all 73 test loops
were then jointly optimized with a robust SE(2) least-squares backend. KITTI
ground truth was loaded only after optimization for error calculation.

The optimizer converged in 153 function evaluations. Overall translation RMSE
decreased from 4.862 m to 3.975 m (18.24%), and rotation RMSE decreased from
1.286 degrees to 0.915 degrees (28.81%). Robot A translation RMSE decreased
from 4.442 m to 2.821 m, while Robot B decreased from 5.335 m to 5.050 m.

Using all 149 validation and test loop edges is retained only as an ablation:
after 2,000 evaluations it had not met the convergence criterion, translation
RMSE improved by 8.18%, but rotation RMSE worsened by 21.53%. This indicates
that accepting more true loop pairs does not guarantee better pose constraints;
registration uncertainty and mutually inconsistent transforms still require
robust weighting or edge selection.

Pose-graph artifacts are stored under `pose_graph_test/` (primary experiment)
and `pose_graph_all/` (all-edge ablation).
