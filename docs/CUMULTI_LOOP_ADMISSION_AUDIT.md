# CU-Multi Loop Admission 真实统计审计

日期：2026-11-17

状态：**PASS（带明确证据边界）**

上游仓库：`James-5757/fyp-place-recognition`

上游提交：`c09f9feb002947174d032971d5028487b9a35c85`

## 1. 这次回答什么

GICP 通过不等于一条约束最终进入位姿图。当前系统依次执行：

```text
SC Rank-1 候选
  -> GICP quality >= 0.6091
  -> 相邻重复闭环聚类，每个事件只保留最高 quality
  -> Robot1 KF150 / Robot3 KF234 稳定起点过滤
  -> 最终 PGO 闭环边
```

本审计把候选级、重复事件级和最终图边级分开统计，避免把 `GICP_pass`
与 `loop_admitted` 当成同一个布尔值。

## 2. 当前后端：Rank-1 漏斗

| 阶段 | 数量 | 占 2,000 个查询 | 相对上一阶段保留 |
| --- | ---: | ---: | ---: |
| Robot1→Robot3 查询 / Rank-1 候选 | 2,000 | 100.00% | 100.00% |
| GICP 通过 | 1,233 | 61.65% | 61.65% |
| 时序聚类代表边 | 120 | 6.00% | 9.73% |
| 稳定起点后的最终接纳边 | **75** | **3.75%** | **62.50%** |

解释：

- 767 个 Rank-1 候选没有达到冻结 GICP 门槛；
- 1,113 条通过边是同一地点在相邻关键帧中的重复观测，占通过边的
  90.27%，去重后成为 120 个独立时序事件；
- 其余 45 条并非再次配准失败，而是端点位于冻结稳定起点之前；
- 最终 75 条才是 Stage 12D clean backend、当前 Demo 和后续 PGO 使用的
  `loop_admitted` 边。

时序聚类规则保持上游冻结定义：相邻通过边只有在 query gap `>3` 或
`abs(candidate gap) >5` 时才开始新事件；每个事件保留最高 GICP quality。

## 3. 历史 Top-3 路线不能与 75 条混算

Stage 9 的 Top-3 rank-ordered early stop 是另一条历史策略：

| 阶段 | 数量 | 占 2,000 个查询 | 相对上一阶段保留 |
| --- | ---: | ---: | ---: |
| 查询 | 2,000 | 100.00% | 100.00% |
| 首个通过 GICP 的 Top-3 候选 | 1,403 | 70.15% | 70.15% |
| 时序聚类代表边 | 139 | 6.95% | 9.91% |

其离线混淆统计为 TP 1,374、FP 29、FN 459、TN 138，即接受 precision
97.93%、correct-pair recall 74.96%。139 条是 Top-3 历史策略的去重结果，
不是当前 Rank-1 clean backend 的输入；不能把 `1403 -> 139 -> 75` 写成一条漏斗。

## 4. 数据质量与 GT 隔离

本次独立核对了：

- Top-3 1,403 条接受记录与 139 个聚类完全对账，移除 1,264 条重复；
- Rank-1 代表文件有 120 行、120 个唯一事件，全部为 Rank-1、quality
  不低于 0.6091；
- clean 文件的 75 条均是上述 120 条的精确子集；
- query/candidate ID、时间戳、SC 字段、GICP quality、变换和
  `T_query_from_candidate` 方向在过滤前后逐值不变；
- clean 输出不包含 GT、真值距离或误差标签；到达顺序声明为 GT-free；
- Rank-1 输入文件 SHA-256 与 Stage 12D provenance 记录完全一致。

Stage 9 的四行 confusion table 是冻结决策后的**聚合离线评价**，只用于报告
TP/FP/FN/TN，不参与候选、配准、聚类或稳定起点过滤。

## 5. 证据边界

当前上游提交没有发布完整的 pre-sanitation Rank-1 逐查询表
`rank1_raw_edges.csv`，服务器普通归档中也没有找到它。因此 1,233 这个数字
来自冻结的 `rank1_sanitation_summary.json`，不能在本次从 2,000 行原表重新
计算。下游 120 行、75 行、输入哈希、子集关系和所有保留值已独立复核。

这足以确认当前 Demo/PGO 的真实输入数量和 admission 语义，但若以后要做
逐查询 forensic audit，仍应从原实验归档补回 `rank1_raw_edges.csv`；补文件
不能改变冻结门槛或已有结果。

## 6. 可复现入口

公开聚合结果：

- `results/cumulti_loop_admission/admission_summary.json`
- `results/cumulti_loop_admission/admission_funnel.csv`

将团队仓库检出到任意本地目录后运行：

```bash
python src/cumulti/audit_loop_admission.py \
  --team-root /path/to/fyp-place-recognition \
  --source-commit c09f9feb002947174d032971d5028487b9a35c85 \
  --output-dir outputs/cumulti_loop_admission
```

脚本只写聚合摘要，不复制逐查询 GT 或原始数据。
