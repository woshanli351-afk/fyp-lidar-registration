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

复查命令：

```bash
python -m unittest discover -s src/kitti -p test_pipeline.py -v
python -m unittest discover -s src/cumulti -p test_stage11_gicp_value.py -v
python src/cumulti/gicp_value_backend.py --help
python src/cumulti/run_stage11_gicp_value_study.py --help
python src/cumulti/run_stage11_latency.py --help
bash -n src/kitti/run_cvtnet_autodl.sh
bash -n src/kitti/run_geometry_autodl.sh
git diff --check
```
