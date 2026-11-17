# 首次发布验证

日期：2026-09-23。验证针对公开仓库整理版；不替代原始服务器实验验收。

| 检查 | 结果 |
| --- | --- |
| KITTI 原有纯逻辑单元测试 | 14/14 通过 |
| CU-Multi Stage 11 原有单元测试 | 7/7 通过 |
| Python 源文件 AST 解析 | 22 个文件通过 |
| 支持的命令行入口 `--help` | 14/14 通过 |
| 两个 KITTI shell 入口 `bash -n` | 通过 |
| CU-Multi 环境变量路径传递 | 通过 |
| Stage 11 子进程使用当前 Python | 通过 |
| 所有公开结果 JSON 解析 | 通过 |
| 凭据/个人服务器路径扫描 | 待发布内容未检出实际凭据或个人服务器路径 |
| `.gitignore` 对环境、原始数据、权重与运行输出的排除 | 通过 |

本地逻辑验证环境：macOS arm64，Python 3.13；项目独立 venv 复用了已安装的科学计算包，
补装 rosbags 和 OpenCV。完整 Open3D/CVTNet 运行仍推荐 Python 3.10/3.11，并按目标 CUDA/平台安装依赖。

未执行：全量 KITTI/CU-Multi 重跑、真实 Open3D 配准、GPU CVTNet 推理、PGO 收敛、地图融合 Demo。
公开历史指标来自原研究归档；原服务器验收见 `results/cumulti_stage11/VALIDATION_REPORT.txt`。

> 上段是 2026-09-23 首次发布时的历史状态。2026-11-10 已补做完整
> 5796 节点 batch PGO 的 GT-free 重放与重力保持诊断；2026-11-15 已完成
> 独立离线 GT/点云地图评价；2026-11-16 又完成 Robot1/Robot3 分轴协方差
> 审计和 soft ×8 的冻结后评价；2026-11-17 完成 GT-free 地图尾部来源与
> 闭环局部敏感性复核。详见 `docs/CUMULTI_PGO_GRAVITY_VALIDATION.md`、
> `docs/CUMULTI_PGO_SOFT_GRAVITY_INPUT_AUDIT.md` 和
> `docs/CUMULTI_PGO_SOFT_GRAVITY_VALIDATION.md`、
> `docs/CUMULTI_PGO_MAP_TAIL_DIAGNOSTIC.md`。

## 2026-11-17 协方差软重力发布验证

| 检查 | 结果 |
| --- | --- |
| 姿态协方差审计测试 | 4/4 通过 |
| PGO 几何与按节点因子测试 | 8/8 通过 |
| 软重力输入与尺度冻结测试 | 3/3 通过 |
| 离线 ATE/RPE/地图评价测试 | 7/7 通过 |
| Stage 11 SC/GICP 回归测试 | 7/7 通过 |
| 相关测试合计 | **29/29 通过** |
| 5 个相关 CLI `--help` | 通过 |
| 所有公开 JSON 解析 | 通过 |
| soft 指标由公开 CSV 独立复算 | 通过 |
| 最终图表视觉检查 | 通过，无截断、重叠或缺失标记 |
| 凭据、服务器/个人绝对路径扫描 | 待提交内容未检出 |
| `git diff --check` | 通过 |

真实数据实验在独立服务器目录完成：Robot1/Robot3 协方差覆盖 5795/5796，
GT-free 阶段冻结 soft ×8，随后仅对该候选运行 GT/点云评价。自由基线的上游
GT 与地图汇总均在 `1e-6` 容差内复现，pre-GT 决定未改变，离线评价未重跑
求解器。公开仓库不包含原始 ROS 包、GT、点云、逐节点轨迹、服务器路径或凭据。

复查命令：

```bash
python -m unittest discover -s src/kitti -p test_pipeline.py -v
python -m unittest discover -s src/cumulti -p test_stage11_gicp_value.py -v
python -m unittest discover -s src/cumulti -p test_pgo_gravity_validation.py -v
python -m unittest discover -s src/cumulti -p test_pgo_gravity_offline.py -v
python -m unittest discover -s src/cumulti -p test_pgo_orientation_covariance.py -v
python -m unittest discover -s src/cumulti -p test_pgo_soft_gravity_validation.py -v
python src/cumulti/run_pgo_soft_gravity_validation.py --help
python src/cumulti/plot_pgo_soft_gravity_comparison.py --help
python src/cumulti/audit_pgo_orientation_covariance.py --help
python src/cumulti/gicp_value_backend.py --help
python src/cumulti/run_stage11_gicp_value_study.py --help
python src/cumulti/run_stage11_latency.py --help
bash -n src/kitti/run_cvtnet_autodl.sh
bash -n src/kitti/run_geometry_autodl.sh
git diff --check
```

## 2026-11-17 地图尾部诊断验证

| 检查 | 结果 |
| --- | --- |
| 尾部方向/标签/体素/直方图/闭环上下文测试 | 7/7 通过 |
| 上述 PGO/Stage 11 回归测试 | 29/29 通过 |
| 相关测试合计 | **36/36 通过** |
| 尾部诊断与绘图 CLI `--help` | 通过 |
| 原全地图 free/soft 指标复现 | 通过 |
| 双方向指标由公开 CSV 独立复算 | 通过 |
| 五组闭环局部 p95 差值检查 | 全部为正，9.3–21.7 mm |
| 所有公开 JSON 解析 | 通过 |
| 汇总图视觉检查 | 通过，无截断或缺失标记 |
| GT/路径/凭据隔离扫描 | 待提交内容未检出 |
| `git diff --check` | 通过 |

真实数据诊断在独立服务器目录只读运行，没有修改 Stage 12D 或 soft ×8 冻结
结果。公开摘要声明 `GT_used=false`；keyframe 加载只允许 ID 和 LiDAR 文件名。
完整逐帧/逐回环文件没有复制到 Git。

新增复查命令：

```bash
python -m unittest discover -s src/cumulti -p test_pgo_map_tail.py -v
python src/cumulti/diagnose_pgo_map_tail.py --help
MPLCONFIGDIR=/tmp/fyp-mpl MPLBACKEND=Agg \
  python src/cumulti/plot_pgo_map_tail_diagnostic.py --help
```

## 2026-11-17 Loop Admission 审计验证

| 检查 | 结果 |
| --- | --- |
| 合成漏斗/哈希/稳定起点/变换保护测试 | 4/4 通过 |
| 当前环境可运行的 Stage 11/PGO 聚合回归测试 | 29/29 通过 |
| 依赖 GTSAM 的两组历史测试 | 当前 Python 3.13 环境未安装 GTSAM，未重跑；本次未修改其代码 |
| 上游 Top-3 接受数与聚类输入数 | 1403 = 1403 |
| Top-3 聚类代表 / sanitized 行数 | 139 / 139 |
| Rank-1 summary / sanitized 行数 | 120 / 120 |
| Rank-1 来源 SHA-256 与 Stage 12D provenance | 完全一致 |
| clean loops 是 Rank-1 representatives 精确子集 | 75/75 通过 |
| ID、时间戳、SC、quality 与变换逐值保持 | 通过 |
| clean 输出 GT 字段扫描 | 未发现 |
| 实际审计 CLI | `PASS: Rank-1 2000 -> 1233 -> 120 -> 75` |

复查命令：

```bash
python -m unittest discover -s src/cumulti -p test_loop_admission.py -v
python src/cumulti/audit_loop_admission.py \
  --team-root /path/to/fyp-place-recognition \
  --source-commit c09f9feb002947174d032971d5028487b9a35c85 \
  --output-dir outputs/cumulti_loop_admission
```

审计来源是团队提交 `c09f9feb002947174d032971d5028487b9a35c85`。上游没有
提交完整 pre-sanitation Rank-1 原表，因此 1233 读取自冻结 summary；120 行
代表边与 75 行 clean 边已逐行独立复核。公开输出只有聚合统计和哈希。
