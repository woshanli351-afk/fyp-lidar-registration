# CU-Multi 软重力因子输入审计

日期：2026-11-16。状态：Robot3 分轴姿态协方差已通过 GT-free 审计；Robot1
原始 IMU/GPS 归档缺失，因此尚未开始软重力因子的权重选择或 GT 评价。

## 为什么先审计协方差

Stage 12C 的稳定性表只保存 `ekf_orientation_covariance_proxy`。源代码定义它为
`nav_msgs/msg/Odometry.pose.covariance` 中旋转 `X/Y/Z` 三轴块的 trace，也就是
roll、pitch、yaw 方差之和。重力因子只应限制倾斜方向（roll/pitch），不能把
yaw 不确定度混入权重，否则会把重力观测错误地设得过松。

审计工具 `src/cumulti/audit_pgo_orientation_covariance.py` 回到原始 ROS 2 DB3，
按冻结的 EKF 消息序号解码每个稳定段 keyframe 的完整 3×3 旋转协方差。它不
读取 GT，也不改变 Robot1 KF150、Robot3 KF234 的稳定段起点。

## 完整性检查

| 检查 | Robot3 结果 |
| --- | ---: |
| 冻结稳定段 keyframes | 3946 |
| 成功定位并解码原始消息 | 3946 / 3946 |
| Stage 12C trace 最大复现误差 | `9.996e-17 rad²` |
| 零 trace / 负对角 / 非半正定行 | 0 / 0 / 0 |
| 旋转协方差最大非对角元素 | 0 |

因此 Robot3 的分轴协方差可用。逐 keyframe 输出仍保留在 Git 忽略的实验目录，
公开仓库只保存聚合结果和判定。

Robot1 稳定段有 1850 个 keyframes，其中 1849 行在 Stage 12C 中匹配到 EKF
消息；但是当前本地工作区与服务器项目数据目录中都没有 Robot1 的原始
`imu_gps` DB3。现有 CSV 只有三轴 trace，无法恢复 roll/pitch 与 yaw 的分量，
所以 Robot1 尚不满足同样的输入验收。

## Robot3 分轴结果

这里的倾斜 sigma 代理定义为
`sqrt((roll_variance + pitch_variance) / 2)`；它是预声明软因子量级的输入，
不是经过 GT 调优后的最终噪声模型。

| 分位数 | tilt sigma proxy | yaw sigma | yaw 占三轴方差比例 |
| --- | ---: | ---: | ---: |
| min | 0.01596° | 0.10929° | 82.73% |
| p05 | 0.02730° | 0.16010° | 93.36% |
| median | **0.04735°** | 0.57397° | **98.68%** |
| p95 | **0.09290°** | 2.77663° | 99.88% |
| p99 | 0.11001° | 3.56200° | 99.94% |
| max | 0.15532° | 4.97329° | 99.96% |

这证明 Stage 12C 三轴 trace 不能作为倾斜噪声。Robot3 如果把 trace 平均到三轴，
得到的 sigma 中位数为约 0.336°、p95 为约 1.604°，远大于从 roll/pitch 分量
得到的 0.047°/0.093°；差异主要来自 yaw。原硬诊断使用的 `0.01°` 又低于
Robot3 观测到的最小倾斜代理 0.01596°，进一步说明它只是失效模式上界。

## 当前决定

1. **否决**直接使用 `ekf_orientation_covariance_proxy` 或
   `sqrt(trace/3)` 给重力因子定权重；它主要描述 yaw。
2. **不运行**只基于 Robot3 分布的双机器人软因子，也不根据当前 ATE/地图 GT
   反推 sigma。
3. 先取得 Robot1 原始 `imu_gps` 归档，或导出同一 EKF 消息序号对应的
   roll/pitch/yaw 协方差；然后用相同脚本完成覆盖率、trace 复现和 PSD 检查。
4. 两个机器人输入都通过后，先冻结按节点协方差映射及有限灵敏度范围，再运行
   离线评价。采用门槛仍为 joint/Robot3 ATE 改善、RPE 不恶化、地图 NN
   p90/p95 不恶化，并检查没有新增大步长。

这个结论是一个输入数据阻塞，不是算法失败。Robot3 已证明原始 EKF 数据足以
支持正确建模；缺少的是 Robot1 同粒度证据。

## 复现

安装 `requirements-pgo.txt` 后运行：

```bash
python src/cumulti/audit_pgo_orientation_covariance.py \
  --robot robot3 --stable-start 234 \
  --stability-csv /path/to/robot3_full_ekf_stability.csv \
  --imu-gps-archive /path/to/robot3_main_campus_imu_gps.zip \
  --output-dir outputs/cumulti_pgo_orientation_covariance_audit
```

公开聚合判定位于
`results/cumulti_pgo_gravity_validation/orientation_covariance_audit_summary.json`。
