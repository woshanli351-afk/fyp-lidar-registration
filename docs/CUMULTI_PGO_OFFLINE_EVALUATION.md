# CU-Multi PGO 离线精度评价状态

日期：2026-11-15。状态：当前 75 回环自由/重力保持轨迹的真实 GT 与点云
地图评价已完成；自由 Stage 12D 基线复现通过。

## 为什么不能直接复用历史结果

本地 `gravity_preserving_20260929` 归档评价的是 Robot1 KF91、Robot3 KF86
之后的 6003 节点、88 回环图。当前实验固定为 Robot1 KF150、Robot3 KF234
之后的 5796 节点、75 回环 Stage 12D 图。两者节点、回环集合和初始化均不同，
所以历史硬重力 ATE `14.155 m` 不能作为当前结果。

## 离线 GT 与地图结果

评价读取 Robot1 KF150 之后 1850 个节点、Robot3 KF234 之后 3946 个节点，
两分支共用同一 5796 节点/75 回环图。求解、参数和 pre-GT 选择均未重跑。

| 指标 | 自由 Pose3 | 硬重力诊断 | 相对变化 |
| --- | ---: | ---: | ---: |
| joint ATE RMSE | 19.608 m | **13.737 m** | **-29.94%** |
| joint ATE p95 | 39.635 m | **24.327 m** | **-38.62%** |
| Robot1 scoped ATE RMSE | 12.108 m | **7.674 m** | **-36.62%** |
| Robot3 scoped ATE RMSE | 22.271 m | **15.798 m** | **-29.06%** |
| combined RPE translation RMSE | 10.5409 m | 10.5411 m | +0.002% |
| combined RPE rotation RMSE | 7.522° | 7.496° | -0.34% |

| 跨机器人地图 NN | 自由 Pose3 | 硬重力诊断 | 相对变化 |
| --- | ---: | ---: | ---: |
| median | 3.031 m | **3.016 m** | -0.47% |
| p90 | **92.693 m** | 100.950 m | +8.91% |
| p95 | **119.278 m** | 125.215 m | +4.98% |

自由分支的 GT 指标与地图指标均在 `1e-6` 容差内复现上游 Stage 12D，说明
比较没有受到数据或实现漂移影响。硬重力因子明显改善 joint/per-robot ATE，
同时保持固定间隔 RPE；但是地图 NN 尾部恶化，所以不能把 ATE 改善解释为
地图质量整体验收通过。硬因子仍是失效模式诊断上界。

公开聚合指标位于 `results/cumulti_pgo_gravity_validation/offline_gt_metrics.csv`
和 `map_consistency.csv`，论文主图为 `offline_gt_map_comparison.png`。

## 新评价器

`src/cumulti/evaluate_pgo_gravity_offline.py` 严格与求解器分离：

1. 先验证 `pre_gt_decision.json` 声明 `GT_used=false`；
2. 冻结 pre-GT 决定、自由轨迹、重力轨迹和 GT keyframe 文件的 SHA-256；
3. 再读取 GT 列并计算 joint rigid ATE、joint alignment 下的机器人 scoped
   ATE、每机器人独立 shape ATE，以及固定 10 keyframe RPE；
4. 如提供点云，则严格复用 Stage 12D 的 frame stride 10、point stride 16、
   1–60 m 距离过滤、0.5 m voxel 和双向最近邻指标；
5. 自由分支必须在 `1e-6` 容差内复现已有 Stage 12D GT/地图汇总，否则停止；
6. 评价过程不调用 GTSAM、不重新求解，也不修改 pre-GT 决定。

测试覆盖无尺度刚体对齐、全局刚体变换下的 RPE 不变性、joint/independent
ATE、双向地图 NN、冻结基线复现和完整 phase-order 写出。

## 数据与运行命令

需要原 Stage 12D processed 数据中的：

- `robot1/keyframes.csv`、`robot3/keyframes.csv`，含
  `keyframe_id,x,y,z,qx,qy,qz,qw,lidar_file`；
- `robot1/lidar/`、`robot3/lidar/` 下对应 `.npy` 点云。

GT-only 评价可不传 LiDAR root；完整评价命令为：

```bash
external/pgo-env/bin/python src/cumulti/evaluate_pgo_gravity_offline.py \
  --pre-gt-decision outputs/cumulti_pgo_gravity_validation_final/pre_gt_decision.json \
  --free-trajectory outputs/cumulti_pgo_gravity_validation_final/free_pose3_replay_trajectory.csv \
  --gravity-trajectory outputs/cumulti_pgo_gravity_validation_final/ekf_gravity_hard_diagnostic_trajectory.csv \
  --robot1-keyframes /path/to/robot1/keyframes.csv \
  --robot3-keyframes /path/to/robot3/keyframes.csv \
  --robot1-lidar-root /path/to/robot1/lidar \
  --robot3-lidar-root /path/to/robot3/lidar \
  --expected-free-metrics /path/to/clean_offline_gt_evaluation.csv \
  --expected-free-map /path/to/clean_map_consistency.csv \
  --output-dir outputs/cumulti_pgo_gravity_offline_evaluation
```

本次完整评价在独立实验目录中只读服务器 processed 数据，未修改团队仓库或
冻结结果。公开仓库不保存原始 GT、点云、逐帧轨迹、服务器绝对路径或凭据；
只保存聚合指标、输入哈希、复现判定和图表。
