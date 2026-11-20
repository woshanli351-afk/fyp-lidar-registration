# 项目上下文

更新时间：2026-11-20。

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
- Demo V2 新增独立 Gravity diagnostics 视图。页面加载由冻结 Free Pose3 与
  soft ×8 轨迹重建的真实 GT-free 展示点云，提供同尺度 BEV/高程切换，并把
  倾斜/ATE 改善、闭环局部 p95 +9.3 mm、当前约 152× 运行成本并列展示。
  展示点同时保留原始 keyframe ID，页面可按 Robot1/Robot3 采集顺序同步播放、
  暂停、拖动和调速。该页不改变主六阶段重放、K75 地图或最终 Batch 后端选择。
- 已增加 `docs/HANDOFF.md`，集中记录当前结论、Demo 操作、数据使用边界、
  外部数据交接规则和校园试采的最小设备/验收流程。下一阶段先确认实际
  LiDAR、IMU、时钟与外参条件，再决定 LIO 和大规模采集方案。
- 新增 `src/cumulti/run_pgo_gravity_validation.py`，固定读取 Stage 12D 导出，
  在求解前冻结输入哈希和 GT-free 判定，并输出自由/重力保持两条完整轨迹。
- 新增独立 offline evaluator，固定在 pre-GT 决定之后计算 joint/scoped/
  independent ATE、10-keyframe RPE 和点云地图 NN，并强制自由分支复现上游
  汇总。当前 5796 节点/75 回环图已完成实际评价：hard-gravity 将 joint ATE
  RMSE 从 19.608 m 降到 13.737 m，Robot3 scoped ATE 从 22.271 m 降到
  15.798 m；combined RPE 基本不变，原全地图 NN p90/p95 上升 8.91%/4.98%。
  后续已确认该尾部受覆盖不对称主导。因此硬因子仍是诊断上界，历史 6003
  节点/88 回环结果也不得替代当前图。
- 软因子输入审计已从 Robot1/Robot3 原始 ROS 2 Odometry 解码 1849/1850、
  3946/3946 个稳定段分轴协方差。Robot1 KF1199 的 header 差 52.034 ms，
  超过冻结门槛，未放宽阈值而是省略该节点因子；总覆盖率 5795/5796。
- GT-free 软因子实验预声明 `1/2/4/8` 协方差膨胀尺度并冻结最弱的 ×8。
  soft ×8 将 joint ATE RMSE 从 19.608 m 降到 13.725 m，Robot3 scoped ATE
  降到 15.785 m。GT-free 尾部定位随后证明，全地图 NN p90/p95 的
  8.90%/4.96% 回退由 Robot3 独有、远离闭环端点的覆盖区域主导；该指标不能
  单独作为不等覆盖路线的地图对齐门槛。
- 75 个已接受闭环的端点扫描局部复核显示，soft ×8 的 pooled p95 只上升
  9.3 mm；五组采样设置均为正，范围 9.3–21.7 mm。系统结论仍是不进入最终
  后端/iSAM2/Demo，但理由修正为小幅一致局部回退与约 62 s 运行成本，而非
  灾难性的全地图尾部错位。
- Loop Admission 真实统计已在团队提交
  `c09f9feb002947174d032971d5028487b9a35c85` 上核对：当前 Rank-1 后端为
  `2000 查询 -> 1233 GICP pass -> 120 时序事件 -> 75 稳定段最终边`。
  最后 75 条是 120 条的精确子集，变换值和方向未改变，clean 输出无 GT。
  历史 Top-3 的 `2000 -> 1403 -> 139` 必须单独报告。
- odometry/Batch/iSAM2 完整对比已闭合。本仓库复用了团队 Stage 13B/13C 的
  冻结事件流，在同一 5796 节点/75 回环图上新增 odometry-only，并统一测量
  独立/joint ATE、RPE、同 gauge 差异、运行时间和峰值内存。odometry-only
  的 Robot1/Robot3 独立 ATE RMSE 为 0.559/0.701 m，但不能合图；Batch/
  iSAM2 joint ATE 为 19.608/19.632 m。默认 iSAM2 复现上游结果，但平移
  0.124 m median、0.564 m p95 未通过冻结门槛。最终决定为
  `KEEP_BATCH_FOR_OFFLINE_DEMO`；这只选择可复现的离线后端，不代表 PGO
  已提高真实轨迹精度或完成实时系统验收。

## 冻结条件

M=1000；voxel 0.75 m；coarse/fine 对应距离 3/1 m；各 30 iterations；SC yaw 初始化；
quality = fitness/(1+inlier RMSE)；GICP threshold=0.6091。
SC 安全门槛为 0.7502294182777406，仅从 Robot1→Robot3 开发集冻结。
Robot2→Robot3（2,230 查询）与 Robot3→Robot2（4,180 查询）为 held-out 方向，禁止用其 GT 回调参数。

## AI 恢复工作顺序

先读 README、docs/HANDOFF、NEXT_STEPS、CHANGELOG、docs/INTERFACES，再查最新 commit。
读取报告中的统计时区分 query/pair/event、protocol FP 与 mapping hazard。
在线 manifest/GICP 输出无 GT；离线评价决策含 GT 字段，不能直接作为线上接口使用。
Gravity 的尺度选择同样必须先在 GT-free 阶段冻结；当前 soft ×8 仍未通过
冻结的严格不恶化规则，不得根据 GT 或地图结果回头调尺度。评价部分重叠路线时
必须把方向覆盖率、闭环局部 NN 和全地图 NN 分开解释。
每次修改记录相关测试及未能完成的真实数据验证，并提交、推送该任务变更。
