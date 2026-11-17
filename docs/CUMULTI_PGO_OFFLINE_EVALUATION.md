# CU-Multi PGO 离线精度评价状态

日期：2026-11-17。状态：当前 75 回环自由、硬重力和协方差软重力轨迹的真实
GT 与点云地图评价已完成；自由 Stage 12D 基线复现通过，地图尾部来源已复核。

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
同时保持固定间隔 RPE。后续 GT-free 诊断确认，全地图 NN 尾部主要来自不等
覆盖而不是闭环重叠区域；因此不能把 ATE 改善或该单一尾部指标分别解释为地图
质量整体验收通过。硬因子仍是失效模式诊断上界。

公开聚合指标位于 `results/cumulti_pgo_gravity_validation/offline_gt_metrics.csv`
和 `map_consistency.csv`，论文主图为 `offline_gt_map_comparison.png`。

## 协方差软重力追加评价

软因子候选 `ekf_gravity_soft_cov_x8` 在读取 GT 前按预声明规则冻结；离线评价器
通过 `--candidate-label` 复用同一协议，并确认 pre-GT 决定未改变、求解器未重跑。

| 指标 | 自由 Pose3 | soft ×8 | 相对变化 |
| --- | ---: | ---: | ---: |
| joint ATE RMSE | 19.608 m | **13.725 m** | **-30.00%** |
| Robot3 scoped ATE RMSE | 22.271 m | **15.785 m** | **-29.12%** |
| combined RPE translation RMSE | 10.5409 m | 10.5411 m | +0.002% |
| map NN median | 3.031 m | **3.018 m** | -0.40% |
| map NN p90 | **92.693 m** | 100.945 m | **+8.90%** |
| map NN p95 | **119.278 m** | 125.198 m | **+4.96%** |

软因子的 ATE 改善与硬诊断近乎相同。追加诊断显示原全地图尾部回退主要是覆盖
指标伪影；但 75 个闭环局部 p95 在五组设置下仍增加 9–22 mm，最终系统选择
仍为离线诊断。公开追加结果使用 `soft_covariance_` 前缀，完整判定见
[协方差软重力验证](CUMULTI_PGO_SOFT_GRAVITY_VALIDATION.md)。

## 全地图 NN 的适用边界

Robot1→Robot3 方向有约 94% 的点在 5 m 内找到对应点，soft p95 从 5.682 m
改善到 5.521 m；Robot3→Robot1 只有约 40% 在 5 m 内找到对应点，并主导
100–125 m 尾部。超过自由全地图 p90 的 soft 点全部来自 Robot3，且全部距
最近闭环端点超过 300 keyframe。

因此，对于不等覆盖路线，本文的全地图对称 NN p90/p95 只能作为覆盖敏感的
描述指标，不能单独作为地图对齐 gate。后续应至少同时报告方向覆盖率和闭环
局部 NN。完整分解见 [PGO 地图尾部诊断](CUMULTI_PGO_MAP_TAIL_DIAGNOSTIC.md)。

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

评价软候选时把 `--gravity-trajectory` 指向冻结的 soft 轨迹，并增加：

```bash
--candidate-label ekf_gravity_soft_cov_x8
```

本次完整评价在独立实验目录中只读服务器 processed 数据，未修改团队仓库或
冻结结果。公开仓库不保存原始 GT、点云、逐帧轨迹、服务器绝对路径或凭据；
只保存聚合指标、输入哈希、复现判定和图表。
