# Changelog

本文件记录项目的主要变更，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中

- 图像 / 多模态任务的界面输入通道（缩略图预览、检测框标注）
- 标注结果持久化（SQLite），支持任务断点恢复
- 引入官方 `krippendorff` 库替换简化实现
- 导入现成标注数据集（CoNLL / JSONL / Label Studio）
- 标注指南的 diff 视图与版本回滚

## [0.1.0] - 2026-11-01

首个公开版本。

### 新增

- **注册表驱动的领域框架**：8 个预置行业、19 个研究对象，覆盖文本 / 图像 / 多模态与分类、NER、关系抽取、情感、意图、检测、分割等任务
- **金标三集**：参考集（few-shot 引导）、优化集（调优校准）、测试集（独立评估）
- **五个协作 Agent**：标注、检验、裁决、复核、指南
- **标注能力**：单 AI 零样本 / 单 AI + 参考集 / 多 AI 独立互检；每条结果带置信度、判断理由、难度分级
- **一致性检验**：Cohen's κ、Fleiss' κ、Krippendorff's α（简化实现）、Landis-Koch 强度解读、分歧样本定位
- **裁决策略**：多数投票、置信度加权聚合、人工裁决
- **复核策略**：随机抽检 5% / 仅疑难 / 全量，支持应用人工修正
- **过程监控**：保守 / 平衡 / 宽松 / 手动四档停止条件，κ 稳定性窗口判定
- **成本预估**：按样本数 × 模型数 × 平均 token 估算费用与耗时
- **反馈闭环**：分歧样本驱动标注指南优化，版本号以 `Decimal` 递增
- **分批标注**：断点续标、整批重标、批间修改指南 / 预设 / 金标
- **LLM 接入**：OpenAI 兼容后端（DeepSeek / 通义 / OpenAI / 自定义）+ 免 Key 的 mock 后端；每个模型独立连接；正式标注前连接自检
- **导出**：CSV（UTF-8 BOM）/ JSONL / JSON / Excel / Markdown
- **Web 界面**：Gradio 七步式界面，步骤条 + 卡片化布局

### 工程

- 采用 `src/` 布局，模块收拢为单一可安装包 `data_annotation_agent`
- 新增 `pyproject.toml`，支持 `pip install -e .`，提供 `data-annotation-agent` 控制台入口与 `python -m data_annotation_agent` 模块入口
- 测试改造为 pytest 可收集的真实断言，覆盖 12 个文件、130 项用例；原脚本式 `check()` 只计数不断言，pytest 下会假通过
- 新增 GitHub Actions CI（Python 3.10 / 3.11 / 3.12），含静态检查与示例冒烟
- 代码通过 `ruff check`（E / F / W / I 规则）：清理 25 处未用导入、15 处导入顺序、3 处未用变量、2 处歧义变量名
- 新增 `docs/`（架构、扩展、一致性指标）与 `examples/`（四个可运行示例）
- 界面默认监听地址由 `0.0.0.0` 改为 `127.0.0.1`：界面无鉴权，不应默认暴露到局域网

### 修复

- 移除仓库根目录的空 `__init__.py`，它会把整个仓库变成一个无意义的 Python 包
- `guideline_agent.optimize` 由已弃用的 `BaseModel.copy()` 改为 `model_copy()`
