# 安装、数据与运行

所有命令从仓库根目录执行。推荐 Python 3.10/3.11；安装和纯逻辑测试见根 README。
本仓库的 `results/` 是历史摘要，运行输出写到 `outputs/`。

## KITTI / CVTNet

将 KITTI Odometry 00 放到 `data/kitti/sequences/00/velodyne`，poses 放到 `data/kitti/poses/00.txt`。
如运行历史 PGO，还需要相应 `calib.txt`。

```bash
python src/kitti/scan_context_baseline.py \
  --kitti-root data/kitti --pose-file data/kitti/poses/00.txt \
  --output-dir outputs/kitti/baseline
git clone https://github.com/BIT-MJY/CVTNet.git extern/CVTNet
python -m pip install -r requirements-cvtnet.txt
# 按 CVTNet 官方 README 取得公开 NCLT checkpoint，放在其默认权重位置。
bash src/kitti/run_cvtnet_autodl.sh
python -m pip install -r requirements-gicp.txt
bash src/kitti/run_geometry_autodl.sh
```

两个 shell 入口名保留历史命名，但已支持本地/其他服务器。
可用 `PYTHON_BIN`、`CANDIDATES`、`VELODYNE`、`RUN_ROOT` 覆盖对应路径。
CVTNet 默认从 `extern/CVTNet/pretrained_model.pth.tar` 读取权重；可通过 shell 第二个参数指定。
此 checkpoint 是 NCLT→KITTI 迁移，不能称为 KITTI 训练模型。

## CU-Multi 数据布局

```text
data/cumulti/processed_v1/full_2hz/robot{1,2,3}/
  keyframes.csv
  scan_context_descriptors.npy
  lidar/<lidar_file>
data/cumulti/raw/                   # frame audit 所需原始包/GT/标定
external/fyp-place-recognition/     # 团队冻结 Stage 1/4/9 输出
```

描述子形状为 N×20×60。在线读取 ID、LiDAR 路径/时间戳；离线评估另需 keyframes 中的
世界坐标 `x,y,z`、四元数 `qx,qy,qz,qw`。不得把 GT 用作候选过滤。
Stage 11 M=1000 的检索假设数据库至少有 1000 个描述子；小型合成输入只用于逻辑测试。

如实际数据不在默认目录，运行前设置（示例值需要替换）：

```bash
export CUMULTI_PROCESSED_ROOT=/path/to/processed_v1
export CUMULTI_RAW_ROOT=/path/to/raw
export FYP_UPSTREAM_REPO=/path/to/team-repository
export OMP_NUM_THREADS=8
```

设置 `FYP_UPSTREAM_REPO` 仅指定上游冻结输入位置；Stage 11 输出仍位于本仓库的 `outputs/`。
Stage 5/8/9 文件作为依赖保留，其独立历史 main 入口不是本次支持的复现入口。

## 使用已有候选做配准

```bash
python -m pip install -r requirements-gicp.txt
python src/cumulti/gicp_value_backend.py \
  --manifest /path/to/candidates_top3_gtfree.csv \
  --direction robot2_to_robot3 --top-k 3 --limit 30 \
  --output outputs/smoke/gicp.csv
```

全量运行去掉 `--limit 30`，需要断点续跑则加 `--resume`。保留所有冻结配准默认参数。

## Stage 11 顺序

原始运行输入与哈希列在 `results/cumulti_stage11/run_manifest.json`；该公开副本的路径已脱敏。
上游 Git 仓库本身不保证附带冻结输出，需要从团队研究归档取得这些文件。
开发门槛复现需要 `outputs/cumulti_v1/09_sc_gicp_integration/rank1_gicp_results.csv`；
frame audit 需要冻结 protocol_validation、原始 LiDAR/TF、URDF 和 GT。

已有冻结策略可直接复制作为新输出的初始配置：

```bash
mkdir -p outputs/cumulti_v1/11_gicp_value_study
cp results/cumulti_stage11/frozen_policy.json outputs/cumulti_v1/11_gicp_value_study/
python src/cumulti/run_stage11_gicp_value_study.py prepare --direction robot2_to_robot3
python src/cumulti/run_stage11_gicp_value_study.py register --direction robot2_to_robot3 --limit 30
python src/cumulti/run_stage11_gicp_value_study.py register --direction robot2_to_robot3 --resume
python src/cumulti/run_stage11_gicp_value_study.py evaluate --direction robot2_to_robot3
python src/cumulti/run_stage11_latency.py --direction robot2_to_robot3 --repeats 3 --warmup 20 --measured 200
```

反向使用 `robot3_to_robot2` 重复。`freeze` 只在复现开发集门槛时运行，不能以 held-out 数据重新调参。
两方向完整运行后可执行 `combine`；`finalize` 还要求 Robot2 frame audit 和开发集 30 对历史复现文件，
缺少这些输入时会失败，不应绕过验收条件。

原始 30 对复现文件名为 `development_reproduction/rank1_30_new_backend.csv`。
原始逐对注册结果和该复现文件未公开；重新验证需按原归档样本 ID 重建。
本次仓库发布只做逻辑/导入/CLI 验证，不声称在缺少上述数据时完成端到端重现。
