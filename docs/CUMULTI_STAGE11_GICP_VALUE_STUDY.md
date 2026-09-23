# CU-Multi Stage 11 — GICP 作用与性价比研究

## 研究问题与冻结协议

本阶段回答两个独立问题：GICP 作为几何验证器能否显著减少错误闭环，以及其输出的相对位姿是否足以作为后续位姿图约束。Stage 8 的 M=1000 Ring-Key/KD-tree Scan Context、Stage 9 的 SC yaw 初始化、0.75 m voxel、3/1 m coarse/fine correspondence、每级 30 次迭代、`fitness/(1+inlier_RMSE)` 质量分数和 0.6091 门槛全部保持不变。

Robot1→Robot3 只用于冻结 SC 安全门槛。开发集中错误 Rank-1 的最高 SC score 为 0.7502294182777405，因此部署门槛固定为其下一浮点数 0.7502294182777406。Robot2→Robot3 和 Robot3→Robot2 是未调参迁移测试；所有 Top-3 候选和 GICP 结果先在不含 GT 字段的文件中冻结，之后才加入 GT 做离线评价。

通用 backend 固定使用 candidate 为 source、query 为 target，输出 `T_query_from_candidate`。Robot2 的点云帧、URDF 和 TF/UTM 四元数约定已独立核对：点云为 `robot2_os_sensor`，URDF 净 yaw 为 0°，20 个 `/tf` 样本在记录的 UTM 180°约定修正后最大方向残差为 2.84e-14°。

## 五种策略

1. `sc_rank1`：直接接受 SC Rank-1；
2. `sc_score_gate`：只接受达到冻结安全门槛的 SC Rank-1；
3. `gicp_rank1`：Rank-1 通过固定 GICP 门槛才接受；
4. `gicp_top3`：按 SC 名次依次运行 GICP，接受第一个通过的候选；
5. `selective_gicp`：高置信 SC Rank-1 直接接受，其余运行 Top-3 early stop。

SC 直接接受只产生地点识别决定，没有米制平移变换；若要向位姿图添加边，仍须运行 GICP 或其他几何配准器。

## Held-out 准确率、安全性与相对位姿

### Robot2→Robot3

| 策略 | Precision | Correct-pair recall | 无重叠拒绝率 | Protocol FP | Mapping-hazard events | 平均 GICP calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SC Rank-1 | 98.03% | 99.73% | 0.00% | 44 | 不适用 | 0.000 |
| SC score gate | 99.94% | 80.02% | 97.37% | 1 | 不适用 | 0.000 |
| Rank-1 GICP | 99.65% | 91.15% | 81.58% | 7 | 0 | 1.000 |
| Top-3 GICP | 99.56% | 93.25% | 76.32% | 9 | 0 | 1.188 |
| Selective GICP | 99.57% | 94.89% | 76.32% | 9 | 0 | 0.353 |

### Robot3→Robot2

| 策略 | Precision | Correct-pair recall | 无重叠拒绝率 | Protocol FP | Mapping-hazard events | 平均 GICP calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SC Rank-1 | 44.76% | 99.57% | 0.00% | 2,309 | 不适用 | 0.000 |
| SC score gate | 99.93% | 80.52% | 99.96% | 1 | 不适用 | 0.000 |
| Rank-1 GICP | 99.81% | 86.00% | 99.87% | 3 | 0 | 1.000 |
| Top-3 GICP | 99.70% | 89.09% | 99.78% | 5 | 0 | 2.216 |
| Selective GICP | 99.71% | 90.95% | 99.78% | 5 | 0 | 1.828 |

Protocol FP 按严格 `GT d_xy >= 5 m` 定义。Mapping hazard 按 Stage 10 图权重的 3σ 定义，即平移误差大于 3 m 或旋转误差大于 15°。所有带 GICP 变换的策略在两个方向上均为 0 个危险事件，但这不等于跨数据集安全证明。

Rank-1 GICP 接受真边的平移误差中位数/p95，在 Robot2→Robot3 为 0.355/1.148 m，在 Robot3→Robot2 为 0.356/1.133 m；旋转误差中位数/p95 分别为 1.018/6.185°和 0.920/5.187°。这说明固定 GICP 变换在当前机器人组合上具有可用的相对位姿精度，但后续 PGO 仍需独立收敛验收。

## 端到端延迟

每个方向、每种策略使用 20 个 warm-up 和 200 个固定测量查询，重复三次；`OMP_NUM_THREADS=8`，检索和持久化 GICP backend 在同一顺序进程中执行。

| 方向 | 策略 | Mean | p95 | 测量样本平均 calls |
| --- | --- | ---: | ---: | ---: |
| Robot2→Robot3 | SC Rank-1 | 19.80 ms | 20.01 ms | 0.000 |
| Robot2→Robot3 | Rank-1 GICP | 59.32 ms | 83.80 ms | 1.000 |
| Robot2→Robot3 | Top-3 GICP | 67.72 ms | 144.24 ms | 1.210 |
| Robot2→Robot3 | Selective GICP | 37.87 ms | 138.26 ms | 0.405 |
| Robot3→Robot2 | SC Rank-1 | 19.19 ms | 19.41 ms | 0.000 |
| Robot3→Robot2 | Rank-1 GICP | 87.26 ms | 146.06 ms | 1.000 |
| Robot3→Robot2 | Top-3 GICP | 164.88 ms | 354.51 ms | 2.285 |
| Robot3→Robot2 | Selective GICP | 132.89 ms | 286.35 ms | 1.910 |

全部 p95 均低于 500 ms。与 Stage 9 的 538/1,301 ms p95 不同，本阶段复用同一进程中的描述子索引、点云 LRU 和 backend；结果表明旧延迟主要受到反复 backend 初始化和点云访问方式影响，不能直接归因于 GICP 数值优化本身。

## 结论与系统选择

- **纯地点识别：** raw SC Rank-1 在包含大量无重叠查询时不安全。SC score gate 可在约 19 ms 达到约 99.9% precision，但 held-out recall 只有约 80%，适合允许 abstain 的轻量模式。
- **地图拼接：** SC 只能给候选和粗 yaw，不能生成米制相对位姿。Rank-1 GICP 在两个 held-out 方向均提供高 precision、0 个预定义危险约束事件以及小于 150 ms 的 p95，因此值得保留。
- **Top-3：** 可增加约 2–3 个百分点 recall，但会增加 protocol FP 和尾延迟，只适合作为离线或 Rank-1 失败后的可选恢复路径。
- **选择性策略：** 正向能减少调用并提高 recall，但在无重叠占多数的反向方向平均需 1.91 次调用，p95 高于 Rank-1 GICP。它没有跨方向实现预先要求的一致成本下降，因此不作为统一在线默认策略。

最终默认策略为 **M=1000 SC Rank-1 + 固定 GICP 门槛**。SC score gate 保留为只做地点识别的轻量模式；Top-3 只作离线恢复。Stage 11 不运行 PGO，也不把本结果包装成地图融合成功。

## 复现与验收

代码位于 `src/cumulti/gicp_value_backend.py`、`run_stage11_gicp_value_study.py` 和 `run_stage11_latency.py`。结果位于 `outputs/cumulti_v1/11_gicp_value_study/`。完整验收、输入 SHA-256、冻结策略和最终决策分别记录在 `VALIDATION_REPORT.txt`、`run_manifest.json`、`frozen_policy.json` 和 `strategy_decision.json`。

通用 backend 在 Robot1→Robot3 的 30 对冻结样本上复现 Stage 9：最大质量差 1.11e-16，最大变换矩阵元素差 2.77e-15。七项纯逻辑单元测试通过。
