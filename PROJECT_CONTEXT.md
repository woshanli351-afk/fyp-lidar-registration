# 项目上下文

更新时间：2026-11-10。

## 目的与分工

导师要求评估 Scan Context 已经很强时 GICP 的作用与性价比，并逐步实现双轨迹地图融合 Demo。
本仓库记录 woshanli351-afk 的 CVTNet/几何验证工作，以及为复现所需的共享检索模块。
团队主仓库为 https://github.com/James-5757/fyp-place-recognition 。不把复用的团队模块归为个人原创。

## 当前事实

- KITTI：已有 SC、CVTNet NCLT→KITTI 迁移、固定 hybrid、几何验证及历史 PGO 实验代码和指标。
- CU-Multi：已完成 Stage 11 GICP 价值研究；来源提交 `2ff22b8e5aa661d36f25414cdcac5de8e4ed87b1`。
- 来源代码基于团队提交 `198b3ac`；完整 SHA 见 docs/source_files.json。
- 发布版补齐 Stage 11 依赖、参数化存储路径、使用当前 Python 调用 backend；没有改变 GICP 数值参数。
- 完整注册/评价 CSV 留在研究归档，不随 Git 发布。公开 JSON 中机器路径使用占位符，数值保持原样。
- 报告推荐 Rank-1 GICP 默认策略；Top-3 保留离线恢复，SC score gate 仅用于地点识别。
- CU-Multi batch PGO 已完成第一轮独立 GT-free 验证：自由 Pose3 分支精确
  复现团队 Stage 12D；EKF up-vector 硬诊断将公共重力偏差 p95 从
  12.800° 降至 0.352°，且没有恶化回环平移残差或产生 >5 m 新跳变。
  这定位了 6DoF 倾斜失效模式，但离线 GT、点云地图一致性和增量后端仍未验收。
- Robot1/Robot3 数据质量复核确认完整运行的 EKF 启动段不可用于建图：
  Robot1/Robot3 最大相邻 XY 跳变为 291.454/333.575 m。当前离线链路必须
  使用冻结起点 Robot1 KF150、Robot3 KF234，并保留后段协方差与航向告警。
- 已增加渐进式离线 Demo V2：准确率/效率面板读取 Stage 11 汇总，地图
  重放使用团队仓库 Stage 12D/13A 的稳定片段 batch K1/K10/K75 结果。
  Demo 不将 Stage 13B iSAM2 描述为已验收在线系统。
- 已增加 `docs/HANDOFF.md`，集中记录当前结论、Demo 操作、数据使用边界、
  外部数据交接规则和校园试采的最小设备/验收流程。下一阶段先确认实际
  LiDAR、IMU、时钟与外参条件，再决定 LIO 和大规模采集方案。
- 新增 `src/cumulti/run_pgo_gravity_validation.py`，固定读取 Stage 12D 导出，
  在求解前冻结输入哈希和 GT-free 判定，并输出自由/重力保持两条完整轨迹。

## 冻结条件

M=1000；voxel 0.75 m；coarse/fine 对应距离 3/1 m；各 30 iterations；SC yaw 初始化；
quality = fitness/(1+inlier RMSE)；GICP threshold=0.6091。
SC 安全门槛为 0.7502294182777406，仅从 Robot1→Robot3 开发集冻结。
Robot2→Robot3（2,230 查询）与 Robot3→Robot2（4,180 查询）为 held-out 方向，禁止用其 GT 回调参数。

## AI 恢复工作顺序

先读 README、docs/HANDOFF、NEXT_STEPS、CHANGELOG、docs/INTERFACES，再查最新 commit。
读取报告中的统计时区分 query/pair/event、protocol FP 与 mapping hazard。
在线 manifest/GICP 输出无 GT；离线评价决策含 GT 字段，不能直接作为线上接口使用。
每次修改记录相关测试及未能完成的真实数据验证，并提交、推送该任务变更。
