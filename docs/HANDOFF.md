# FYP 项目交接文档

更新时间：2026-10-20。

本文用于在聊天记录缺失时恢复项目状态。接手人应先阅读本页，再按文中链接查看证据和运行说明。

## 1. 项目目标与仓库分工

项目研究多机器人 LiDAR 地点识别、几何验证和地图拼接。当前主线是：

1. 用 Scan Context（SC）从另一台机器人的历史关键帧中检索候选地点；
2. 用 GICP 验证候选，并估计米制相对位姿；
3. 将通过验证的跨机器人约束交给位姿图优化（PGO）；
4. 用渐进式离线 Demo 展示候选、验证、约束和地图融合过程。

仓库分工：

- 本仓库：[woshanli351-afk/fyp-lidar-registration](https://github.com/woshanli351-afk/fyp-lidar-registration)，维护 GICP 价值研究、数据质量分析和渐进式 Demo。
- 团队仓库：[James-5757/fyp-place-recognition](https://github.com/James-5757/fyp-place-recognition)，维护共享数据处理、地点检索和地图融合集成。

接手工作前应同时拉取两个仓库的最新提交。不要直接覆盖团队仓库中尚未提交的实验文件；跨仓库复用代码时保留来源和提交号。

## 2. 当前完成状态

| 模块 | 状态 | 可安全声称的结论 |
| --- | --- | --- |
| Scan Context 检索 | 已完成冻结评估 | 原始 Rank-1 召回高，但在大量无重叠查询下会产生较多错误候选。 |
| GICP 几何验证 | Stage 11 已完成 | Rank-1 GICP 显著减少假闭环，并输出可用的相对位姿；应保留。 |
| Rank-1/Top-3/选择性策略 | 已比较 | Rank-1 GICP 为默认；Top-3 仅作离线或失败恢复；选择性策略不作为统一默认。 |
| Robot1/Robot3 数据质量 | 已完成专项复核 | 完整 EKF 启动段不可用于建图；必须使用冻结稳定起点。 |
| Progressive Demo V2 | 已完成并可本地运行 | 是使用冻结结果的静态离线重放，不是浏览器内实时运行 SC/GICP/PGO。 |
| CU-Multi batch PGO | GT-free 诊断已完成 | 自由 Stage 12D 已精确复现，重力保持方向有效；离线 GT、点云地图和增量后端仍未验收。 |
| 校园自主采集 | 尚未开始 | 下一步先确认 LiDAR/IMU 型号、记录字段和时间同步，再做小规模试采。 |

本文建立时，本仓库上一项已验收工作为提交 `a72d660`（Robot1/Robot3 数据质量评估）。后续接手时以远端最新提交为准，不要把该哈希当作永久版本号。

## 3. 冻结的 SC/GICP 结论

### 3.1 默认策略

地图拼接默认使用 **M=1000 SC Rank-1 + GICP quality threshold 0.6091**。

冻结参数：

- voxel：0.75 m；
- coarse/fine correspondence：3.0/1.0 m；
- coarse/fine iterations：各 30；
- 初始旋转：SC signed yaw；
- quality：`fitness / (1 + inlier RMSE)`；
- GICP 接受门槛：0.6091；
- SC 轻量门槛：0.7502294182777406，只从 Robot1→Robot3 开发集冻结。

SC 只给候选和粗 yaw，不能生成米制平移。SC-only 地点识别决定不得用单位矩阵冒充位姿图边。

### 3.2 Held-out 结果

| 方向 | Rank-1 GICP precision | Correct-pair recall | p95 延迟 | 预定义 mapping-hazard events |
| --- | ---: | ---: | ---: | ---: |
| Robot2→Robot3 | 99.65% | 91.15% | 83.80 ms | 0 |
| Robot3→Robot2 | 99.81% | 86.00% | 146.06 ms | 0 |

这两个方向均低于 500 ms 预算。0 个危险事件只描述本次冻结离线评价，不能外推为跨数据集安全证明；仍存在 protocol false positives。

完整统计、五种策略和延迟协议见 [Stage 11 GICP 报告](CUMULTI_STAGE11_GICP_VALUE_STUDY.md)。

### 3.3 变换方向和数据隔离

GICP 固定使用 candidate 为 source、query 为 target：

```text
p_query = T_query_from_candidate @ p_candidate
```

输出方向必须写成 `T_query_from_candidate`。向 PGO 传边时先确认求解器的边方向；需要反向时显式求逆。

在线候选 manifest、GICP 输出和前端事件不得包含 GT。GT 只能在候选和配准结果冻结后，用于离线评价。接口字段见 [候选与几何配准接口](INTERFACES.md)。

## 4. Robot1/Robot3 数据异常与使用边界

Robot1 和 Robot3 的关键帧 ID、时间戳与顺序本身正常，但完整 EKF 轨迹的启动阶段存在多秒级、数百米级错误位移：

| Robot | 关键帧数 | 最大相邻 XY 跳变 | 冻结稳定起点 | 保留比例 |
| --- | ---: | ---: | ---: | ---: |
| Robot1 | 2,000 | 291.454 m | KF150 / 75.034 s | 92.5% |
| Robot3 | 4,180 | 333.575 m | KF234 / 117.003 s | 94.4% |

原始约 100 Hz EKF 也包含这些异常，因此它们不是 CSV 降采样、浏览器绘图或 GTSAM 制造的。同期 GNSS 几乎静止，当前证据支持“早期 EKF 初始化/收敛失败”这一根因类别，但尚不能证明具体触发器。

操作规则：

1. 不得把 Robot1/Robot3 完整 EKF 轨迹直接用于地图融合；
2. 当前离线研究保持 Robot1 KF150、Robot3 KF234 两个冻结起点；
3. 不得根据 GT 重新挑切点；
4. 后段协方差告警和航向语义问题必须继续保留；
5. 稳定段适合当前离线 Demo，不足以证明全局地图准确。

完整证据与待查问题见 [Robot1/Robot3 数据质量评估](CUMULTI_ROBOT13_DATA_QUALITY_ASSESSMENT.md)。

## 5. Demo 当前内容与运行方法

Demo 位于 `demo/progressive_replay_v2/`，包含：

- Robot2→Robot3 / Robot3→Robot2 的准确率和效率面板；
- SC Rank-1 候选、GICP 验证和接受闭环的分阶段展示；
- Robot1/Robot3 六阶段轨迹与地图重放；
- K1/K10/K75 冻结 batch 地图快照；
- 系统视图和分析视图；
- 可选卫星/GNSS 参考底图；
- 播放、暂停、阶段跳转、重置和全屏控制。

从仓库根目录运行：

```bash
python scripts/serve_progressive_demo.py
```

浏览器打开 <http://127.0.0.1:8000>。常用操作：

- `Space`：播放/暂停；
- `1`–`6`：跳转阶段；
- `R`：重置；
- `Fullscreen`：全屏；
- K1/K10/K75：查看对应冻结地图快照。

Demo 由本机 HTTP 服务提供。`127.0.0.1` 只能被运行服务的电脑访问；其他人可克隆仓库后本地启动。若要公开访问，应另行部署静态文件或通过受控网络入口转发，不能公开服务器凭据。

卫星底图需要联网；离线时仍可使用 LiDAR 视图。更完整的构建和声明边界见 [Demo README](../demo/progressive_replay_v2/README.md) 与 [运行指南](RUNNING.md)。

演示时必须明确：浏览器播放的是预计算结果；它不在浏览器内运行 SC、GICP 或 GTSAM，也不代表实时同步采集或已验收的增量 iSAM2。

## 6. 文件和证据位置

| 内容 | 位置 |
| --- | --- |
| GICP 最终报告 | `docs/CUMULTI_STAGE11_GICP_VALUE_STUDY.md` |
| 数据异常报告 | `docs/CUMULTI_ROBOT13_DATA_QUALITY_ASSESSMENT.md` |
| 接口约定 | `docs/INTERFACES.md` |
| 完整运行指南 | `docs/RUNNING.md` |
| 来源和归属 | `docs/PROVENANCE.md` |
| Stage 11 公开摘要 | `results/cumulti_stage11/` |
| 数据质量公开摘要 | `results/cumulti_robot13_data_quality/` |
| Demo | `demo/progressive_replay_v2/` |
| Demo 构建/启动脚本 | `scripts/build_progressive_replay_demo.py`、`scripts/serve_progressive_demo.py` |
| SC/GICP 通用 backend | `src/cumulti/gicp_value_backend.py` |

完整查询级 CSV、原始点云、GT、模型权重、服务器路径和登录凭据不在 Git 中。它们必须通过私密渠道交接，并在实验 manifest 中记录哈希；不要写入公开文档或提交历史。

组会英文演示文稿 `CU_Multi_Two_Week_Update_Demo_Data_Audit_EN_v3.pptx` 是本地会议材料，目前不是仓库的一部分；若仍需使用，应通过项目共享盘单独归档。

## 7. 接手后的最小验证

无需原始点云即可完成以下检查：

```bash
python -m unittest discover -s src/cumulti -p test_stage11_gicp_value.py -v
python src/cumulti/gicp_value_backend.py --help
python src/cumulti/run_stage11_gicp_value_study.py --help
python scripts/serve_progressive_demo.py --no-browser
```

启动 Demo 后确认：

1. 校验提示包含 75 个事件和 10 组策略指标；
2. Robot1/Robot3 轨迹、SC 候选、GICP 阶段和 K1/K10/K75 可切换；
3. 前端数据不含 GT、距离误差或真值标签字段；
4. 页面声明为 offline replay；
5. 卫星瓦片不可用时 LiDAR 视图仍正常。

完整 GICP 复现还需要外部 CU-Multi 数据和冻结上游产物。不要因为缺少数据而绕过哈希、完整查询数或 GT 隔离验收。

## 8. 尚未完成的问题

优先级从高到低：

1. **PGO 后续验收：** batch 收敛、变换方向、鲁棒核和重力失效模式已完成
   GT-free 检查；下一步补 joint/per-robot ATE、RPE、点云地图一致性及
   batch/incremental 数值一致性。当前仍不能把 Demo 地图当作全局精度证明。
2. **Robot1/Robot3 传感器根因：** 取得采集时 MicroStrain 消息定义、启动配置和 map/earth TF，解码异常窗口的 EKF、双天线与 filter 状态。
3. **团队集成：** 对齐候选 ID、时间戳、点云帧和 `T_query_from_candidate`，再重放冻结 Rank-1 GICP 边。
4. **校园试采：** 用自有设备完成小规模数据记录、LIO 和重复路线闭环验证。

## 9. 校园建图建议

### 9.1 最小设备

先准备一套可重复使用的采集平台：

- 一台 3D LiDAR；
- 一台 IMU，优先使用 LiDAR 内置或厂家已提供时间同步的组合；
- 一台能稳定记录原始数据的计算机；
- 稳定电源、电池和备用线材；
- 刚性安装支架，确保 LiDAR 与 IMU 外参不变。

可选设备：GNSS/RTK 用于地理参考和独立评估，相机用于展示和人工核验。它们不是第一轮 SC/GICP/LIO 闭环试验的硬性条件。

采购或确定算法前，先记录 LiDAR 与 IMU 的准确型号、接口、时间戳来源、点频、视场和驱动输出字段。尤其确认点云是否提供每点时间和 ring，以及 LiDAR/IMU 是否共享硬件时钟。

### 9.2 必须记录的数据

- 原始点云及其 frame ID、时间戳、每点时间/ring（若设备提供）；
- 原始 IMU；
- 静态 TF 和 LiDAR–IMU 外参；
- 驱动、URDF、launch/config 与软件版本；
- 录包起止时间、消息频率、丢帧和时钟诊断；
- 若有 GNSS/RTK，保存原始定位、状态、协方差和天线配置。

### 9.3 先做小试采

1. 做一段 2–5 分钟的单回环，检查时间同步、点云畸变和 LIO 轨迹；
2. 做两段 5–10 分钟的重叠路线，模拟 Robot1/Robot3；
3. 让两条路线有约 30%–50% 共同区域，并包含明显建筑边缘、路口和开阔区；
4. 第二次从不同方向经过共同区域，验证视角变化下的 SC/GICP；
5. 小试采通过后再扩大到校园级路线和不同时间/天气。

### 9.4 进入完整实验的验收门槛

- topic、频率、frame 和时间戳完整且单调；
- 无明显丢帧或时钟跳变；
- LiDAR–IMU 外参固定并有记录；
- 单次 LIO 地图无严重重影、断裂或数百米跳变；
- 重复路线能检出闭环，GICP 叠加可人工核验；
- 若声明绝对精度，必须有独立 RTK、测量控制点或其他可信真值。

新数据应沿用“原始传感器 → LIO 局部轨迹/地图 → 关键帧与 SC → GICP → PGO → Demo”的顺序。不要先做网页，再补数据链路。

## 10. 交接检查表

- [ ] 已拉取两个仓库的最新提交并记录 commit；
- [ ] 已通过私密渠道取得所需数据路径和访问方式；
- [ ] 已确认没有把凭据、个人绝对路径或原始大文件放入 Git；
- [ ] 已阅读接口中的变换方向和 GT 隔离要求；
- [ ] 已在本地启动并完整走一遍 Demo；
- [ ] 已理解 Robot1 KF150 / Robot3 KF234 的冻结裁剪规则；
- [ ] 已把 PGO 验收与 SC/GICP 精度结论分开；
- [ ] 已阅读 PGO 重力诊断，并区分 GT-free 失效模式定位与离线精度验收；
- [ ] 开始校园采集前已确认 LiDAR、IMU、时钟与外参方案。

任何新实验都应保存配置、输入 commit、文件哈希、运行命令和指标摘要；完成后更新本页相关状态、`PROJECT_CONTEXT.md`、`NEXT_STEPS.md` 和 `CHANGELOG.md`，再提交并推送。
