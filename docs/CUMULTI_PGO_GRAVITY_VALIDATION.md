# CU-Multi PGO 重力保持诊断

日期：2026-11-15。状态：GT-free batch 诊断与离线 GT/点云地图评价均已完成；
硬重力因子保留为诊断上界，不作为最终后端。

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

## 冻结后的离线 GT 与地图评价

在上述 pre-GT 决定和两条轨迹完全冻结后，独立评价器读取原 Stage 12D
processed keyframes 和点云。自由分支必须先在 `1e-6` 容差内复现上游
GT/地图汇总，否则评价停止；本次两个复现检查均通过。

| 指标 | 自由 Pose3 | 硬重力诊断 |
| --- | ---: | ---: |
| joint ATE RMSE | 19.608 m | **13.737 m** |
| joint ATE p95 | 39.635 m | **24.327 m** |
| Robot1 scoped ATE RMSE | 12.108 m | **7.674 m** |
| Robot3 scoped ATE RMSE | 22.271 m | **15.798 m** |
| combined RPE translation RMSE | 10.5409 m | 10.5411 m |
| combined RPE rotation RMSE | 7.522° | **7.496°** |
| map NN median | 3.031 m | **3.016 m** |
| map NN p90 | **92.693 m** | 100.950 m |
| map NN p95 | **119.278 m** | 125.215 m |

硬重力使 joint ATE RMSE 降低 29.94%，Robot3 scoped ATE 降低 29.06%，
但地图 NN p90/p95 分别恶化 8.91%/4.98%。这说明限制错误 roll/pitch 能改善
全局轨迹与 GT 的一致性，却不足以保证跨机器人点云尾部重叠更好。可能原因
包括局部回环姿态与固定重力的小冲突、硬约束过强，以及地图 NN 对少量远端
错位的高敏感性；这些是后续软因子实验要区分的假设，不是本轮已证明的因果结论。

## 结论边界与下一步

GT-free 诊断和离线 GT 评价共同支持“自由 6DoF 倾斜是一个真实失效模式”，
但地图尾部指标否定了“硬重力已经解决地图拼接”的更强结论。不同因子集合的
原始图目标值仍不能直接横向比较，也不能仅凭 ATE 或回环残差断言地图更准。

后续输入审计已经证明 Stage 12C 三轴姿态 trace 主要来自 yaw，不能直接给重力
因子定权重；Robot3 分轴协方差可用，但 Robot1 原始归档仍缺失。因此先补齐
Robot1 同粒度输入，再预先冻结软因子和有限灵敏度范围。采用门槛必须同时要求
joint/Robot3 ATE 改善、固定间隔 RPE 不恶化、地图 NN p90/p95 不恶化；通过后
再做原生因子和 batch/iSAM2 一致性。输入审计见
`CUMULTI_PGO_SOFT_GRAVITY_INPUT_AUDIT.md`，完整评价协议与公开聚合结果见
`CUMULTI_PGO_OFFLINE_EVALUATION.md`。

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
