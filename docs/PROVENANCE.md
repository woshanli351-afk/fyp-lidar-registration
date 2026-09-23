# 来源与署名

## 项目来源

- `src/kitti/`：用户提供的 2026-08-24 FYP core experiment package 中的项目流水线。
- `src/cumulti/gicp_value_backend.py`、两个 Stage 11 runner 和相应测试：Stage 11 工作提交
  `2ff22b8e5aa661d36f25414cdcac5de8e4ed87b1`，此前保存在独立研究分支。
- Stage 1/5/8/9 辅助模块：来自团队 [James-5757/fyp-place-recognition](https://github.com/James-5757/fyp-place-recognition)
  的共享项目代码，随 Stage 11 归档取得；保留原有文件头与算法，非本仓库新增原创算法。
- `docs/source_files.json` 记录上游 base commit 及发布前每个源文件的 SHA-256。路径参数化后的文件哈希会变化。

## 外部依赖

- [CVTNet 官方实现](https://github.com/BIT-MJY/CVTNet)：另行获取代码与 NCLT checkpoint，遵循其 LICENSE。
  本仓库只包含项目适配脚本，不附带第三方网络实现或 checkpoint。
- Open3D、NumPy、SciPy、pandas、Matplotlib、OpenCV、rosbags、PyTorch：通过包管理器安装，保留各自许可。
- KITTI 与 CU-Multi：使用各自官方数据访问和使用条款，原始数据不在本仓库分发。

本次整理不对团队代码或第三方材料重新授权，也不擅自附加覆盖全部文件的开源许可证。
公开仓库用于项目协作与研究复核；如需进一步分发或商用，请按对应来源确认许可。

## 结果归档

`results/` 保留历史数值；服务器和个人路径改为符号占位符。
manifest 中保存的 SHA-256 指向原研究运行输入，不代表公开脱敏副本的哈希。
原始 source commit 尚未保证可从团队远端直接下载，不能将其写成一个已发布的团队 commit 链接。
