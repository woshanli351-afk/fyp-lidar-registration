# 候选与几何配准接口

这里描述现有脚本接口，作为后续 Demo 的文件交换协议；尚未实现 HTTP 服务。

## CU-Multi 在线候选

CSV 必需列：`query_robot, query_keyframe_id, candidate_robot, candidate_keyframe_id, SC_rank, SC_score, best_shift`。
robot 值为 `robot1`、`robot2`、`robot3`，query 与 candidate 不同。
`SC_rank` 从 1 开始；`best_shift` 为 60 个 sector 中的整数 0–59。
signed shift 为 `shift <= 30 ? shift : shift-60`，yaw 初始化为 signed shift × 6°。
方向字符串示例：`robot2_to_robot3`，前者 query，后者 candidate/database。

检索还输出 LiDAR 时间戳（ns）、点云路径、second score 和 margin；margin 不参与本轮策略调参。
backend 按机器人与 keyframe ID 从 `full_2hz/<robot>/keyframes.csv` 的 `lidar_file` 定位点云。
路径应在对应机器人目录内，不能是绝对路径或逃逸该目录。
点云为 NumPy 数组，前三列 x/y/z（米），其余列不用于配准。

## GICP 输出

保留候选列，并追加以下字段：

| 字段 | 含义 |
| --- | --- |
| `transform_direction` | 固定 `T_query_from_candidate` |
| `gicp_transform` | 4×4 齐次矩阵按行展开的 16 元素 JSON 数组 |
| `init_yaw_deg` | SC 初始化角度，度 |
| `gicp_fitness`, `gicp_inlier_rmse` | Open3D 指标；RMSE 单位米 |
| `gicp_quality` | fitness / (1 + inlier RMSE) |
| `source_load_s`, `target_load_s` | 点云加载耗时，秒 |
| `source_preprocess_s`, `target_preprocess_s` | 裁剪/降采样耗时，秒 |
| `coarse_registration_s`, `fine_registration_s` | 两级配准耗时，秒 |
| `registration_s`, `pair_wall_s` | 配准合计与该次调用总耗时，秒 |
| `source_cache_hit`, `target_cache_hit` | 缓存命中标记 |

candidate 为 source，query 为 target：`p_query = T_query_from_candidate @ p_candidate`（齐次列向量）。
向位姿图传递约束前必须确认求解器采用同一方向；若方向相反，应显式求逆。
SC-only 接受决定没有米制变换，不能填单位矩阵来冒充闭环约束。

## 冻结策略与离线字段

SC 直接接受门槛 `>= 0.7502294182777406`，严格超过开发集最高错误分数。
GICP 接受门槛 `>= 0.6091`。Top-3 按 SC rank 顺序，接受首个通过者。
高置信 bypass 只用于地点识别策略比较；位姿图边仍需实际配准。

在线 manifest/配准输出不得携带 GT 或离线评价标签。
`evaluate` 才合并 GT，产出的 `decisions_*.csv` 和 `all_policy_decisions.csv` 是**离线评价文件**，不是可直接发布给前端的在线流。
Demo 应只投影 query/candidate ID、接受状态、quality 和实际变换，去除所有 GT/误差列。

Protocol FP 为 GT 水平距离 >=5 m；当前归档实现的 mapping hazard 使用平移误差 >3 m
或 SO(3) 旋转角误差 >15°。原计划用语曾写 yaw，但归档代码及指标列实际为 `rotation_error_deg`；
本次发布保持历史统计不变，后续若补 yaw 指标须另记版本，不能把当前数字直接改名为 yaw。

## KITTI 适配

CVTNet 打分接口为 `query_frame,candidate_frame,learned_score`，与固定 SC 候选按 ID 关联。
KITTI 历史评估 CSV 可能含 GT 列；这些研究文件不应直接接入 CU-Multi GT-free backend。
详细命令见 RUNNING.md，历史协议与迁移结果见 results/kitti。
