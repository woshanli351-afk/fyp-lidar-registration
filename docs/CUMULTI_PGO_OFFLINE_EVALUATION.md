# CU-Multi PGO 离线精度评价状态

日期：2026-11-11。状态：评价代码与测试完成；当前 75 回环重力保持轨迹的
真实 GT/点云评价等待原始 processed 数据。

## 为什么不能直接复用历史结果

本地 `gravity_preserving_20260929` 归档评价的是 Robot1 KF91、Robot3 KF86
之后的 6003 节点、88 回环图。当前实验固定为 Robot1 KF150、Robot3 KF234
之后的 5796 节点、75 回环 Stage 12D 图。两者节点、回环集合和初始化均不同，
所以历史硬重力 ATE `14.155 m` 不能作为当前结果。

## 当前可确认的自由基线

本地 Stage 12D 归档保存了自由 Pose3 的离线汇总，但没有逐帧 GT 和点云：

| 指标 | 自由 Stage 12D |
| --- | ---: |
| joint ATE RMSE | 19.608 m |
| Robot1 ATE（joint alignment 下） | 12.108 m |
| Robot3 ATE（joint alignment 下） | 22.271 m |
| combined 10-keyframe RPE translation RMSE | 10.541 m |
| combined 10-keyframe RPE rotation RMSE | 7.522° |
| 跨机器人地图 NN median | 3.031 m |
| 跨机器人地图 NN p95 | 119.278 m |

这些是上游既有自由基线，不是本轮新计算的重力保持结果。当前硬重力轨迹
已经冻结，但只凭上述汇总无法推导其 ATE、RPE 或地图 NN。

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

## 所需数据与运行命令

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

当前仓库与本地归档中没有这四项原始 processed 输入。继续实际评价需要用户
明确授权从服务器读取，或把数据复制到本机；交接文档里的凭据不会被代码、
日志、命令或 Git 记录。
