# CU-Multi PGO 重力保持诊断

日期：2026-11-10。状态：GT-free batch 诊断完成；离线 GT 与点云地图评价待做。

## 问题

团队 Stage 12D 在 Robot1 KF150、Robot3 KF234 之后的稳定片段上构建了
5796 节点 Pose3 图，包含 5794 条顺序里程计边和 75 条 Rank-1 GICP
跨机器人边。其 batch 结果能显著压低回环残差，但此前重力审计发现优化后
轨迹的 up 方向出现约十几度偏移。这里检验一个具体假设：**自由 6DoF PGO
是否在满足回环的同时破坏了 EKF 已估计的重力方向。**

## 固定输入与图约定

- 输入是 Stage 12D 导出的 `clean_local_trajectory.csv` 和
  `ordered_rank1_clean_loops.csv`，不重新选择起点、回环或阈值；
- candidate 为 source，query 为 target，回环测量保持
  `T_query_from_candidate = X_query^-1 X_candidate`；
- 里程计与回环噪声保持旋转 5°、平移 1 m，回环使用 Huber 1.345；
- Robot1 首节点使用 1e-6 的 Pose3 prior；
- 首条到达回环只用于把 Robot3 局部轨迹放入 Robot1 公共坐标系；
- 所有选择、求解和下述判定都不读取 GT。

自由分支逐节点复现了团队保存的 Stage 12D 轨迹：5796 个节点的平移差
最大值为 `4.74e-9 m`，旋转差最大值为 `8.39e-10°`，且没有额外对齐。
因此对照组不是近似重写，而是同一 batch 基线。

## 对照设计

`free_pose3_replay` 完全重放 Stage 12D。`ekf_gravity_hard_diagnostic`
在同一图上为除锚点外的每个节点增加一个 up-vector 因子：

```text
u_sensor = R_local^T [0, 0, 1]
e(X) = R_X u_sensor - u_first-loop-initialized
```

该三维残差直接约束 EKF 重力向量。绕该向量的旋转和全部平移位于因子
零空间，因此 yaw 与位置仍由里程计和回环决定。小角度 sigma 固定为 0.01°。
这是用于定位失效模式的硬诊断上界，不是已经完成协方差建模的生产因子。

## GT-free 结果

| 指标 | 自由 Pose3 | 保持 EKF 重力 |
| --- | ---: | ---: |
| 公共 up 方向偏差 p95 | 12.800° | **0.352°** |
| 相对初始 EKF 重力偏差 p95 | 12.678° | **0.000°** |
| 回环平移残差 p95 | 0.1962 m | **0.1940 m** |
| 回环旋转残差 p95 | **1.384°** | 1.683° |
| 新增相邻 >5 m 跳变 | 0 | 0 |
| 最大相邻平移 | 1.707 m | 1.702 m |
| LM 迭代 | 7 | 7 |
| 本次 Python 实现求解时间 | 0.41 s | 62.76 s |

硬约束分支的公共 up 偏差没有到零，是因为第一条回环建立的两个机器人
公共坐标系本身保留约 0.352° 的重力方向差；每条轨迹相对其 EKF 初始化
重力方向的 p95 偏差为数值上的 0°（最大 `1.03e-5°`）。

结果支持“自由 6DoF 后端会把 EKF 重力方向拉歪”这一失效模式：重力偏差
降低约 97%，回环平移残差没有恶化，也没有产生新的大步长。不过回环旋转
残差 p95 上升约 0.30°，说明部分回环姿态与固定重力之间存在小冲突。
当前重力因子由 Python `CustomFactor` 回调实现，完整图约慢两个数量级；若进入
在线或重复实验，应改写为 C++/原生 GTSAM factor，再重新测量运行成本。

## 结论边界与下一步

这次通过的是 **GT-free 数值诊断**，不是全局地图精度验收。不同因子集合的
原始图目标值不能直接横向比较；也不能仅凭回环残差断言地图更准。

下一步应在不改当前判定的前提下进行离线 GT 与点云地图评价：比较 joint
ATE、每机器人 ATE、固定间隔 RPE、跨机器人地图 NN median/p95，并检查
Robot3 原先的 ATE 恶化是否缓解。若结果一致，再把硬因子替换为由 IMU/EKF
姿态协方差支持的软重力因子，并做灵敏度与 batch/iSAM2 一致性实验。

2026-11-11 已加入独立评价器，详见 `CUMULTI_PGO_OFFLINE_EVALUATION.md`。
本地缺少当前图的逐帧 GT-bearing keyframes 和点云，尚未生成硬重力轨迹的
ATE/RPE/地图指标；历史 88 回环实验不能替代本次 75 回环评价。

## 复现

GTSAM 4.2 的 macOS Python wheel 需要 NumPy 1.x；使用 Python 3.10：

```bash
python3.10 -m venv external/pgo-env
external/pgo-env/bin/pip install -r requirements-pgo.txt
external/pgo-env/bin/python src/cumulti/run_pgo_gravity_validation.py \
  --local-trajectory /path/to/clean_local_trajectory.csv \
  --loop-edges /path/to/ordered_rank1_clean_loops.csv \
  --baseline-trajectory /path/to/clean_all75_gtsam_trajectory.csv \
  --output-dir outputs/cumulti_pgo_gravity_validation
```

完整逐节点输出保留在 Git 忽略的 `outputs/`；可公开的冻结摘要位于
`results/cumulti_pgo_gravity_validation/pre_gt_summary.json`。
