# 更新记录

## 2026-10-12 — 准确率/效率面板与渐进式 Demo V2

- 新增五种 SC/GICP 策略的双向准确率与效率交互面板。
- 新增 Robot1/Robot3 六阶段渐进式地图重放和 75 条冻结 GICP 事件流。
- 使用 stable-segment clean backend 的 K1/K10/K75 batch 地图点集，不使用
  未通过数值一致性门槛的增量 iSAM2 结果作为最终系统结论。
- 增加可复现数据构建脚本；原始 PLY 不入库，浏览器事件字段不含 GT。

## 2026-09-23 — 首次公开仓库整理

- 导入 KITTI SC/CVTNet/GICP 流水线、原有测试与历史轻量结果。
- 导入 CU-Multi Stage 11 通用 backend、策略/延迟脚本及完整本地模块依赖。
- 用环境变量替换机器路径，backend 子进程使用当前 Python；更新 KITTI shell 入口目录。
- 补充项目上下文、运行指南、输入输出约定、代码来源与工作约定。
- 保留原始实验参数和结果；公开文档路径脱敏，原始文件 SHA-256 单独记录。
- 发布验证见 docs/VALIDATION.md；本次没有重跑全量点云实验。
