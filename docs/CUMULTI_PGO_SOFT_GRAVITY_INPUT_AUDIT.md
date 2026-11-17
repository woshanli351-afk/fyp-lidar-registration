# CU-Multi 软重力因子输入审计

日期：2026-11-16。状态：Robot1/Robot3 原始 EKF 姿态协方差均已完成 GT-free
审计；Robot1 有 1 个节点超过冻结的 50 ms header 时间门槛，该节点不添加
重力因子，其余 5795/5796 个图节点可用于按节点软约束。

## 为什么不能使用 Stage 12C 的 trace

Stage 12C 只保存 `ekf_orientation_covariance_proxy`，它是
`nav_msgs/msg/Odometry.pose.covariance` 旋转 X/Y/Z 三轴块的 trace，即
roll、pitch、yaw 方差之和。重力因子只能约束 roll/pitch；把 yaw 混入噪声会
错误地改变权重。

本审计从原始 ROS 2 DB3 解码完整 3×3 旋转协方差，并定义

```text
tilt_sigma_proxy = sqrt((roll_variance + pitch_variance) / 2)
```

该量只来自 EKF 协方差，不读取 GT。Robot1 KF150、Robot3 KF234 的稳定段起点、
50 ms 时间关联门槛和原始消息字段均保持冻结。

## 输入和覆盖率

原 Stage 12C 稳定性 CSV 在当前服务器上已经不存在，因此审计器增加了等价的
GT-free 重建入口：它只读取 processed `keyframes.csv` 中的
`robot_id,keyframe_id,lidar_timestamp_ns`，先找最近 EKF bag timestamp，再检查
header timestamp，两者都必须在 0.05 s 内。它不会加载 keyframe 的位置、姿态
或任何 GT 列。

| 检查 | Robot1 | Robot3 |
| --- | ---: | ---: |
| 冻结稳定段节点 | 1850 | 3946 |
| 成功解码分轴协方差 | 1849 | 3946 |
| 覆盖率 | 99.946% | 100% |
| 零 trace / 负对角 / 非 PSD | 0 / 0 / 0 | 0 / 0 / 0 |
| 原始归档 SHA-256 | `43cd171d...4790a3b4` | `da7fb9fe...43e50ed` |

Robot1 唯一缺失的是 KF1199。最近 bag timestamp 只差 1.618 ms，但 EKF header
timestamp 差 52.034 ms，略高于冻结的 50 ms 门槛。实验没有为补齐这一行放宽
门槛，也没有插值或使用 GT；运行时直接对该节点省略重力因子。Robot1 首节点
本来就是固定锚点，因此最终 5796 节点图加入 5794 个软重力因子。

重建路径中的 trace 是从同一条原始消息再次计算的自一致性检查，不应称为独立
复现旧 Stage 12C CSV。Robot3 先前使用原 Stage 12C 表完成的独立 trace 复现仍
保留：最大绝对误差为 `9.996e-17 rad²`。

## 分轴结果

| 机器人 | tilt sigma min | median | p95 | max | yaw 方差占比 median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Robot1 | 0.01473° | **0.04137°** | **0.07214°** | 0.09704° | **99.41%** |
| Robot3 | 0.01596° | **0.04735°** | **0.09290°** | 0.15532° | **98.68%** |
| 合并 | 0.01473° | **0.04493°** | **0.08796°** | 0.15532° | — |

两个机器人都证明三轴 trace 被 yaw 主导，因此仍然**否决**直接使用 trace 或
`sqrt(trace/3)` 给重力因子定权重。原硬诊断的固定 `0.01°` 也低于两个机器人
观测到的最小倾斜代理，只能视为失效模式上界。

## 审计决定

1. 使用每个已解码节点的 roll/pitch 方差计算基础倾斜 sigma；
2. Robot1 KF1199 按预声明策略省略因子，不修改时间阈值；
3. 允许进入有限、预声明的协方差膨胀实验，但选择尺度时不得读取 GT；
4. 只有冻结候选后才能运行离线 GT/地图评价。后续 2026-11-17 诊断已证明全地图
   NN p90/p95 受不等覆盖主导；最终判定改用方向覆盖率与闭环局部 NN 复核，见
   `CUMULTI_PGO_MAP_TAIL_DIAGNOSTIC.md`。

软因子实验和最终否决结论见
[协方差软重力验证](CUMULTI_PGO_SOFT_GRAVITY_VALIDATION.md)。公开聚合输入判定为
`results/cumulti_pgo_gravity_validation/orientation_covariance_audit_summary.json`。

## 复现

如有 Stage 12C 稳定表，可继续使用原入口：

```bash
python src/cumulti/audit_pgo_orientation_covariance.py \
  --robot robot3 --stable-start 234 \
  --stability-csv /path/to/robot3_full_ekf_stability.csv \
  --imu-gps-archive /path/to/robot3_main_campus_imu_gps.zip \
  --output-dir outputs/robot3_covariance_audit
```

如稳定表缺失，则用冻结 keyframe 时间戳重建：

```bash
python src/cumulti/audit_pgo_orientation_covariance.py \
  --robot robot1 --stable-start 150 \
  --keyframes-csv /path/to/robot1/keyframes.csv \
  --imu-gps-archive /path/to/robot1_main_campus_imu_gps.zip \
  --output-dir outputs/robot1_covariance_audit
```

逐节点协方差、重建关联表和原始归档只保留在 Git 忽略的实验目录；公开仓库只
保存聚合分位数、覆盖率、哈希、缺失策略和判定。
