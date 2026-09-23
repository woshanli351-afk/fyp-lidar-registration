# Scan Context / CVTNet / Hybrid comparison

## Protocol

- Dataset: KITTI Odometry Sequence 00.
- Pseudo-robot split: Robot A frames 0--2400, Robot B frames 2600--4540.
- Sampling step: 5 frames; boundary gap: 100 frames.
- Positive threshold: 5 m.
- Queries: 481 total, 158 with at least one positive in Robot B.
- Candidate set: Scan Context Top-10 (4,810 pairs).
- CVTNet: official NCLT pretrained checkpoint applied to KITTI as a
  cross-dataset transfer experiment; it is not a KITTI-trained checkpoint.
- Hybrid: per-query min-max normalization followed by
  `0.7 * Scan Context + 0.3 * CVTNet`.

## Results

| Method | R@1 | R@5 | R@10 | P@5 | MRR | Pair AP | All-query Top-1 AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scan Context | 94.30% | 96.84% | 97.47% | 40.13% | 95.28% | 63.18% | 97.92% |
| CVTNet transfer | 92.41% | 96.84% | 97.47% | 41.01% | 94.71% | 36.43% | 59.87% |
| Hybrid (alpha=0.7) | 94.30% | 96.84% | 97.47% | 41.14% | 95.40% | 49.05% | 87.47% |

## Interpretation

The public NCLT CVTNet checkpoint does not improve KITTI Top-1 retrieval.  The
fixed hybrid keeps Recall@1 unchanged, improves Precision@5 by 1.01 percentage
points and MRR by 0.11 percentage points, but reduces candidate-pair AP and
all-query Top-1 AP.  It should therefore be reported as a transfer-learning
ablation rather than as an overall improvement over Scan Context.

Top-5/Top-10 recall cannot increase in this experiment because CVTNet only
reorders the fixed Scan Context Top-10 candidate set.  Four valid queries have
no positive candidate in that set, so a second-stage reranker cannot recover
them.  Improving candidate generation and learning or fine-tuning on the
official KITTI train/validation split are separate follow-up experiments.

Thresholds labelled diagnostic in the JSON files were selected on the same
evaluated data and must not be presented as held-out deployment performance.
