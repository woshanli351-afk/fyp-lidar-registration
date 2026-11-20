# CU-Multi Progressive Mapping Replay V2

This static demo combines three frozen, independently generated result sets:

- Stage 11 aggregate SC/GICP accuracy and latency metrics from this repository.
- The team repository's Stage 12D stable-segment clean backend and Stage 13A
  batch K1/K10/K75 snapshots.
- The frozen Soft Gravity ×8 diagnostic, including aggregate evaluation metrics
  and a GT-free Free/Soft map reconstruction sampled from the original LiDAR.

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

Add the optional, presentation-only satellite overlay from the validated
keyframe/GNSS audit tables:

```bash
python scripts/add_geospatial_replay_data.py \
  --gnss-root /path/to/12c_dataset_stability
```

The script joins the midpoint of the two valid GNSS antennas by keyframe ID.
GNSS is used only to draw the reference trajectory over Esri World Imagery; it
is excluded from Scan Context, GICP acceptance and pose-graph optimisation.
Satellite tiles require network access and fall back to a dark tile grid when
unavailable.

Add or refresh the Gravity diagnostic after building the main bundle:

```bash
python src/cumulti/export_pgo_gravity_demo_map.py \
  --free-trajectory /path/to/free_pose3_replay_trajectory.csv \
  --soft-trajectory /path/to/ekf_gravity_soft_cov_x8_trajectory.csv \
  --robot1-keyframes /path/to/robot1/keyframes.csv \
  --robot3-keyframes /path/to/robot3/keyframes.csv \
  --robot1-lidar-root /path/to/robot1/lidar \
  --robot3-lidar-root /path/to/robot3/lidar \
  --output /path/to/gravity_demo_map.json

python scripts/add_gravity_replay_data.py \
  --map-export /path/to/gravity_demo_map.json
```

The exporter loads only `keyframe_id` and `lidar_file` from the processed
keyframe tables. It never loads their GT pose columns. The attachment script
checks the frozen Free/Soft trajectory hashes and the corrected offline-only
decision before updating the static demo bundle.

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

The default **System view** follows the spatial playback language used by
multi-robot SLAM demonstrations: a large common-frame map, colour-coded robot
trajectories, moving robot markers and accumulating inter-robot constraints.
The **Analysis** view retains the source-backed SC/GICP accuracy, latency and
registration diagnostics.

The **Gravity diagnostics** view compares the real Free Pose3 and Soft Gravity
×8 map reconstructions. A source-backed common-up direction gauge makes the
12.80° to 0.35° roll/pitch correction visible before the map comparison. The
map defaults to a clearly labelled `z×6` elevation lens because BEV discards
the vertical dimension; metric 1:1 elevation and BEV remain selectable. The
page separates GT-free selection from explicitly labelled offline GT
evaluation and reports the accepted-loop local map gate as a 35-improved versus
40-worsened balance. The conclusion is deliberately negative: tilt and joint
ATE improve, but the local map p95 rises by 9.3 mm and the current Python
diagnostic is about 152× slower. Gravity does not replace the frozen K75 Batch
PGO map or enter the final backend.

The Gravity view now auto-plays a synchronized Free/Soft reconstruction using
the original keyframe IDs attached to every displayed point. `Play/Pause`,
`Reset`, the progress slider and `0.5×`–`4×` speed controls operate on the same
Robot1/Robot3 acquisition progress. Faint points show the final footprint;
bright points are the keyframes accumulated so far. This is a recorded
keyframe replay, not a claim that the browser runs the optimizer online.

Use the **Satellite / LiDAR** control in System view to switch between the
dual-GNSS geographic reference and the frozen batch LiDAR snapshots.

The SC stage now exposes the Rank-1 candidate before any map edge exists; the
GICP stage then shows the aligned cloud and measured quality against the frozen
threshold. Accuracy callouts quantify how many protocol false positives GICP
removes, while the efficiency panel reports the remaining 500 ms budget
headroom.
