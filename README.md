# 数据标注智能体 · Data Annotation Agent

[![CI](https://github.com/lzitong01-wq/data-annotation-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/lzitong01-wq/data-annotation-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 通用数据标注框架：选行业与研究对象 → 配置 LLM → 标注 → 一致性检验 / 裁决 / 复核 → 反馈闭环 → 导出。
> 多 Agent 分工协作，标注过程可解释、可量化、可中途干预。

**没有 API Key 也能完整跑通**：内置 `mock` 后端，基于规则生成伪标注，用于教学演示、流程验证与二次开发。

---

## 为什么做这个

人工标注贵、慢、质量不稳定；直接让大模型标一遍，又回答不了"标得准不准、分歧在哪、该信谁"。

这个项目把标注当成一条**可度量的工业流水线**，而不是一次性的 LLM 调用：

- **可解释** — 每条结果都带 `confidence`（置信度）与 `rationale`（判断理由），不是黑盒标签。
- **可量化** — 用 Cohen's κ / Fleiss' κ / Krippendorff's α 量化标注者间一致性，分歧样本被显式暴露。
- **可干预** — 标注不必一次跑完：分批追加、中途改指南 / 改预设 / 改金标，然后继续标（断点续标）。
- **可闭环** — 分歧样本自动回流去优化标注指南，版本递增，下一轮更准。
- **可替换** — 行业与研究对象由注册表驱动，新增领域不改核心代码；LLM 走 OpenAI 兼容协议，DeepSeek / 通义 / OpenAI / 自建服务都能接。

---

## 核心特性

| 模块 | 能力 |
| --- | --- |
| **领域注册表** | 8 个行业 × 19 个研究对象，覆盖文本 / 图像 / 多模态与分类、NER、关系抽取、情感、意图、检测、分割等任务；可运行时注册扩展 |
| **金标三集** | 参考集（few-shot 引导）、优化集（调优校准）、测试集（独立评估），三者分离避免评估自欺 |
| **多 AI 互检** | 严谨模式下每个模型走各自独立的后端连接（可跨厂商、跨 Key），独立标注后聚合 |
| **一致性指标** | Cohen's κ / Fleiss' κ / Krippendorff's α + Landis-Koch 强度解读 |
| **难度分级路由** | 按置信度自动分 simple / medium / difficult，简单样本直接采用，疑难样本进人工复核 |
| **裁决策略** | 多数投票、置信度加权聚合、人工裁决 |
| **复核策略** | 随机抽检 5% / 仅疑难 / 全量 |
| **过程监控** | 保守（κ≥0.8 且稳定）/ 平衡（κ≥0.6 且稳定）/ 宽松 / 手动 四档停止条件 |
| **成本预估** | 按样本数 × 模型数 × 平均 token 估算费用与耗时 |
| **反馈闭环** | 分歧样本驱动指南优化，规则与边界情况追加，版本号用 `Decimal` 递增 |
| **分批标注** | 断点续标、整批重标、批间修改指南 / 预设 / 金标 |
| **多格式导出** | CSV（UTF-8 BOM，Excel 直接打开不乱码）/ JSONL / JSON / Excel / Markdown |

---

## 架构

```
┌───────────────────────────────────────────────────────────────┐
│  ui/app.py   Gradio 七步式界面（唯一依赖 gradio 的层）         │
└──────────────────────────────┬────────────────────────────────┘
                               │
                   ┌───────────▼───────────┐
                   │  core/pipeline.py     │  流程编排 / 分批 / 中途修改
                   └───────────┬───────────┘
        ┌──────────────┬───────┴────────┬──────────────┐
        ▼              ▼                ▼              ▼
  agents/         agents/          agents/        agents/
  annotator       inspector        adjudicator    reviewer
  标注(单/多AI)    一致性+分歧       多数/加权/人工   抽检/疑难/全量
        │              │                │              │
        └──────────────┴────────┬───────┴──────────────┘
                                ▼
                    agents/guideline_agent   指南起草 / 分歧驱动优化
                                │
                   ┌────────────▼────────────┐
                   │  core/models.py         │  注册表 / 金标三集 / 预设 / 结果
                   └────────────┬────────────┘
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
  llm/backend.py        quality/agreement.py     quality/monitor.py
  OpenAI 兼容 + mock      κ / α 一致性指标          停止条件 / 成本预估
```

一条样本的完整生命周期：

```
样本 → [标注] 单AI或多AI各出一版标签(带置信度/理由)
     → [检验] 算 κ、找出标签不一致的样本
     → [裁决] 多AI分歧时按多数投票/加权聚合/人工裁决收敛为一个标签
     → [复核] 按预设挑出疑难样本交人工
     → [闭环] 分歧样本回流优化标注指南，版本 +0.1
     → [导出] CSV / JSONL / JSON / Excel / Markdown
```

设计取舍见 [docs/architecture.md](docs/architecture.md)。

---

## 快速开始

```bash
git clone https://github.com/lzitong01-wq/data-annotation-agent.git
cd data-annotation-agent

# 推荐：可编辑安装（会装齐 pydantic / gradio / openai / openpyxl）
pip install -e .

# 或者只装运行依赖
pip install -r requirements.txt
```

### 启动 Web 界面

```bash
python -m data_annotation_agent
# 或安装后使用控制台命令
data-annotation-agent --port 7860

# 浏览器打开 http://127.0.0.1:7860
```

默认只监听 `127.0.0.1`。界面**没有鉴权**，需要局域网访问时显式指定 `--host 0.0.0.0`，并确认网络环境可信。

界面默认就是 `mock` 模式，**不填任何 Key 就能把七步全流程走完**。

### 七步流程

| 步骤 | 做什么 |
| --- | --- |
| ① 选择行业与对象 | 8 个预置行业、19 个研究对象，选完自动带出任务描述与标签体系 |
| ② 配置 LLM 与方案 | 选后端、填 Key、测连接；再选 5 个 Agent 的预设 |
| ③ 输入待标注数据 | 每行一条，提交后开始标注 |
| ④ 标注结果 | 结果明细表 + 难度路由 + 一致性检验/AI 互检 + 人工复核 + 反馈闭环 |
| ⑤ 版本管理 | 查看标注指南的版本演进 |
| ⑥ 中途修改 | 追加标注（断点续标）/ 重标 / 清空；批间改指南、预设、金标 |
| ⑦ 导出结果 | CSV / JSONL / JSON / Excel / Markdown |

### 编程方式调用

```python
from data_annotation_agent import AnnotationPipeline, get_backend
from data_annotation_agent.core.models import AgentPresets, AnnotatorPreset, InspectorPreset

pipeline = AnnotationPipeline(backend=get_backend("mock"))
pipeline.create_config(
    industry_id="medical",
    object_id="medical_ner",
    presets=AgentPresets(annotator=AnnotatorPreset.RIGOROUS, inspector=InspectorPreset.DEEP),
    llm_backend="mock",
    llm_models=["mock-1", "mock-2", "mock-3"],
)

out = pipeline.run_full_pipeline(["患者持续高烧三天", "确诊为病毒性肺炎"])

print(out["results"][0].label)              # 裁决后的最终标签
print(out["inspection"]["agreement"])       # 一致性指标
print(out["inspection"]["n_disagreements"]) # 分歧样本数
print(out["updated_guideline"])             # 指南优化后的版本（无分歧时为 None）
```

---

## 接入真实 LLM

在界面第 ② 步选厂商，`base_url` 与模型名自动填充，只需粘贴 API Key（仅本次会话内存使用，不落盘）：

| 后端 | Base URL | 常用模型 |
| --- | --- | --- |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat`、`deepseek-reasoner` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`、`qwen-max` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 自定义 | 任意兼容 `/chat/completions` 的地址 | 自行填写 |

多模型互检就在"参与标注的模型"里填多个（逗号分隔），例如 `deepseek-chat, qwen-max`。

也可以用环境变量兜底：`LLM_BACKEND` / `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。

---

## 项目结构

```
data-annotation-agent/
├── src/data_annotation_agent/       # 单一可安装包
│   ├── __init__.py                  # 公开 API 与版本号
│   ├── __main__.py                  # python -m data_annotation_agent
│   ├── core/
│   │   ├── models.py                # 注册表 / 金标三集 / 指南 / 预设 / 结果模型
│   │   ├── pipeline.py              # 流程编排、分批标注、中途修改
│   │   └── exporter.py              # 五种导出格式
│   ├── agents/
│   │   ├── annotator.py             # 标注：单 AI / 多 AI、难度分级、可解释输出
│   │   ├── inspector.py             # 检验：一致性计算 + 分歧定位
│   │   ├── adjudicator.py           # 裁决：多数投票 / 加权 / 人工
│   │   ├── reviewer.py              # 复核：抽检 / 疑难 / 全量
│   │   └── guideline_agent.py       # 指南：LLM 起草 + 版本化优化
│   ├── llm/backend.py               # OpenAI 兼容后端 + mock 后端 + 厂商预设
│   ├── quality/
│   │   ├── agreement.py             # Cohen's κ / Fleiss' κ / Krippendorff's α
│   │   └── monitor.py               # 停止条件 + 成本预估
│   └── ui/app.py                    # Gradio 七步式界面
├── tests/                           # pytest 用例（130 项）
├── examples/                        # 可直接运行的示例
├── docs/                            # 架构 / 扩展 / 指标说明
├── data/                            # 本地测试数据（不提交实际内容）
├── pyproject.toml
├── requirements.txt
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Agent 预设

| Agent | 预设 | 含义 |
| --- | --- | --- |
| 标注 | `fast` | 单 AI 零样本，最快，不看金标 |
| | `standard` | 单 AI + 参考集引导（默认） |
| | `rigorous` | 多 AI 独立标注 + 聚合互检 |
| 检验 | `light` | 只算 Cohen's κ（两标注者） |
| | `standard` | Fleiss' κ + 分歧定位（默认） |
| | `deep` | 多指标 + AI 互检 |
| 裁决 | `majority` | 多数投票（默认） |
| | `weighted` | 按置信度加权 |
| | `human` | 人工标签优先 |
| 复核 | `sample` | 随机抽检 5% |
| | `difficult` | 只复核分歧 / 低置信度样本（默认） |
| | `full` | 全量复核 |
| 监控 | `conservative` | κ > 0.8 且连续 3 次稳定才停 |
| | `balanced` | κ > 0.6 且稳定（默认） |
| | `lenient` | 标完预设数量即停 |
| | `manual` | 由使用者决定何时停 |

κ 指标的公式、分档标准与选用建议见 [docs/quality-metrics.md](docs/quality-metrics.md)。

---

## 示例与文档

| 文件 | 内容 |
| --- | --- |
| [examples/quickstart.py](examples/quickstart.py) | 最小可运行示例：mock 模式跑完整链路 |
| [examples/custom_domain.py](examples/custom_domain.py) | 注册新行业与研究对象，全链路复用 |
| [examples/batch_and_export.py](examples/batch_and_export.py) | 分批标注 + 批间改指南 + 多格式导出 |
| [examples/real_llm.py](examples/real_llm.py) | 接入 DeepSeek / 通义 / OpenAI 真实模型 |
| [docs/architecture.md](docs/architecture.md) | 分层、数据流、关键设计决策、并发限制 |
| [docs/extending.md](docs/extending.md) | 扩展行业 / LLM 后端 / 导出格式 / 裁决策略 |
| [docs/quality-metrics.md](docs/quality-metrics.md) | κ 与 α 的公式、解读、常见坑 |

```bash
python examples/quickstart.py
```

---

## 测试

```bash
pip install -e ".[dev]"
pytest -q                       # 130 项
pytest --cov                    # 带覆盖率
```

测试覆盖 12 个文件：一致性指标、监控与成本、注册表、标注、指南、检验、裁决、复核、导出、流程编排、端到端集成。全部在 `mock` 后端下运行，无需网络与 API Key。

CI 在 Python 3.10 / 3.11 / 3.12 上重复验证，并额外跑两个示例脚本做冒烟测试。

---

## 已知限制

- **图像 / 多模态任务只做了数据模型层支持**：界面第 ③ 步的输入是纯文本，图像类任务需要扩展输入与渲染逻辑（见 [docs/extending.md](docs/extending.md) 第 5 节）。
- **Krippendorff's α 为简化实现**：对名义数据复用 Fleiss' κ，且丢弃含缺失值的样本；要严格 α 值请引入 `krippendorff` 库替换。
- **状态存在内存里**：单进程单会话，重启后标注结果与指南版本会丢失，导出请及时。
- **非线程安全**：多用户同时操作同一实例会互相覆盖，多用户需按会话隔离 pipeline。
- **成本单价为硬编码估算值**，未随厂商调价更新，仅作量级参考。

---

## Roadmap

- [ ] 图像 / 多模态输入的界面支持（缩略图预览 + 检测框标注）
- [ ] 标注结果持久化（SQLite），支持任务断点恢复
- [ ] 引入官方 `krippendorff` 库替换简化实现
- [ ] 导入现成标注数据集（CoNLL / JSONL / Label Studio 格式）
- [ ] 标注指南的 diff 视图与版本回滚

---

## License

[MIT](LICENSE) © 2026 李子桐
