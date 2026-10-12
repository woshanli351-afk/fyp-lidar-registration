# CU-Multi Progressive Mapping Replay V2

This static demo combines two frozen, independently generated result sets:

- Stage 11 aggregate SC/GICP accuracy and latency metrics from this repository.
- The team repository's Stage 12D stable-segment clean backend and Stage 13A
  batch K1/K10/K75 snapshots.

The replay is presentation software. It does not run Scan Context, GICP or
GTSAM in the browser, and it does not claim live sensing, a causal online
frontend, simultaneous robot acquisition, or validated incremental iSAM2.

## Build the source-backed bundle

```bash
python scripts/build_progressive_replay_demo.py \
  --team-root /path/to/James-5757/fyp-place-recognition \
  --map-root /path/to/server-exported/stage13a/maps
```

The map directory must contain the six binary XYZ PLY files for Robot1/Robot3
at K1, K10 and K75. The builder deterministically samples presentation points;
raw PLY files are not copied into the repository. It exports only aggregate
evaluation metrics and GT-free online event fields, records input hashes, and
never writes personal source paths into the bundle.

## Run locally

```bash
python scripts/serve_progressive_demo.py
```

The launcher validates the 75-event/10-policy bundle before opening
<http://127.0.0.1:8000>. Use `--no-browser` on a headless host. `file://`
loading is unsupported because the demo fetches its JSON bundle.

## Presenter controls

- `Space`: play or pause.
- `1`–`6`: jump directly to a replay stage.
- `R`: reset the replay.
- `Fullscreen`: enter or leave browser fullscreen.
- K1/K10/K75 buttons: inspect a frozen batch map without changing the timeline.

The SC stage now exposes the Rank-1 candidate before any map edge exists; the
GICP stage then shows the aligned cloud and measured quality against the frozen
threshold. Accuracy callouts quantify how many protocol false positives GICP
removes, while the efficiency panel reports the remaining 500 ms budget
headroom.
