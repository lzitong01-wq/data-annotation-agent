# 数据标注智能体 · Data Annotation Agent

> 一个通用的数据标注框架：选行业与研究对象 → 配置 LLM → 标注 → 一致性检验 / 裁决 / 复核 → 反馈闭环 → 导出。
> 多 Agent 分工协作，标注过程可解释、可量化、可中途干预。

**没有 API Key 也能完整跑通**：内置 `mock` 模拟后端，基于规则生成伪标注，用于教学演示、流程验证和二次开发。

---

## 为什么做这个

人工标注贵、慢、质量不稳定；直接让大模型标一遍，又无法回答"标得准不准、分歧在哪、该信谁"。

这个项目把标注当成一条**可度量的工业流水线**而不是一次性的 LLM 调用：

- **可解释**：每条标注都带 `confidence`（置信度）和 `rationale`（判断理由），不是黑盒标签。
- **可量化**：用 Cohen's κ / Fleiss' κ / Krippendorff's α 量化标注者间一致性，分歧样本被显式暴露出来。
- **可干预**：标注不必一次跑完。可以分批追加、中途改指南 / 改预设 / 改金标，然后继续标（断点续标）。
- **可闭环**：分歧样本自动回流去优化标注指南，指南版本递增，下一轮标注更准。
- **可替换**：行业和研发对象是注册表驱动的，新增领域不改核心代码；LLM 后端走 OpenAI 兼容协议，DeepSeek / 通义 / OpenAI / 自建服务都能接。

---

## 界面流程

Gradio 七步式界面，把一条完整的标注工程链路拆成可回退的步骤：

| 步骤 | 做什么 |
| --- | --- |
| ① 选择行业与对象 | 8 个预置行业、19 个研究对象，选完自动带出任务描述与标签体系 |
| ② 配置 LLM 与方案 | 选后端（mock / DeepSeek / 通义 / OpenAI / 自定义）、填 Key、测连接；再选 5 个 Agent 的预设 |
| ③ 输入待标注数据 | 每行一条，提交后开始标注 |
| ④ 标注结果 | 结果明细表 + 难度分级路由 + 一致性检验/AI 互检 + 人工复核 + 反馈闭环 |
| ⑤ 版本管理 | 查看标注指南的版本演进历史 |
| ⑥ 中途修改 | 追加标注（断点续标）/ 重标 / 清空；批间修改指南、预设、金标参考集 |
| ⑦ 导出结果 | CSV / JSONL / JSON / Excel / Markdown 五种格式 |

---

## 架构

```
┌───────────────────────────────────────────────────────────────┐
│  ui/app.py   Gradio 七步式界面（步骤条 + 卡片化布局）           │
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

**一条样本的完整生命周期：**

```
样本 → [标注] 单AI或多AI各出一版标签(带置信度/理由)
     → [检验] 算 κ、找出标签不一致的样本
     → [裁决] 多AI分歧时按多数投票/加权聚合/人工裁决收敛为一个标签
     → [复核] 按预设挑出疑难样本交人工
     → [闭环] 分歧样本回流优化标注指南，版本 +0.1
     → [导出] CSV / JSONL / JSON / Excel / Markdown
```

---

## 核心特性

| 模块 | 能力 |
| --- | --- |
| **行业 / 对象注册表** | 8 个行业（医疗、金融、法律、自动驾驶、电商、教育、工业、通用）× 19 个研究对象，覆盖文本 / 图像 / 多模态与分类 / NER / 关系抽取 / 情感 / 意图 / 检测 / 分割等任务；可动态注册扩展 |
| **金标三集** | 参考集（示例引导）、优化集（调优校准）、测试集（独立评估），三者均可选 |
| **多 AI 互检** | 严谨模式下每个模型走各自独立的后端连接，独立标注后聚合；模拟模式下会注入受控分歧以演示互检价值 |
| **一致性指标** | Cohen's κ（两标注者）、Fleiss' κ + Krippendorff's α（三标注者及以上），按 Landis & Koch 标准给出文字解读 |
| **难度分级路由** | 按置信度自动分成 simple / medium / difficult，简单样本直接采用，疑难样本进人工复核 |
| **裁决策略** | 多数投票、置信度加权聚合、人工裁决 |
| **复核策略** | 随机抽检 5% / 只复核疑难 / 全量复核 |
| **过程监控** | 保守（κ>0.8 且稳定）、平衡（κ>0.6 且稳定）、宽松（标完预设量）、手动四档停止条件 |
| **成本预估** | 按样本数 × 模型数 × 平均 token 估算费用与耗时，覆盖主流模型单价 |
| **反馈闭环** | 分歧样本驱动指南优化，规则与边界情况追加，版本号用 `Decimal` 递增避免浮点误差 |
| **多格式导出** | CSV（UTF-8 BOM，Excel 直接打开不乱码）、JSONL、JSON、Excel（表头加粗）、Markdown 表格 |

---

## 快速开始

```bash
git clone https://github.com/lzitong01-wq/data-annotation-agent.git
cd data-annotation-agent

pip install -r requirements.txt

# 启动 Web 界面（必须在本目录下运行，因为用了包内相对导入）
python -m ui.app
# 浏览器打开 http://127.0.0.1:7860
```

界面默认就是 `mock` 模拟模式，**不填任何 Key 就能把七步全流程走完**，适合先看效果。

### 接入真实 LLM

在界面第 ② 步选择厂商，`base_url` 和模型名会自动填充，只需粘贴 API Key（仅本次会话内存使用，不落盘）：

| 后端 | Base URL | 常用模型 |
| --- | --- | --- |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat`、`deepseek-reasoner` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`、`qwen-max` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 自定义 | 任意兼容 `/chat/completions` 的地址 | 自行填写 |

多模型互检就在"参与标注的模型"里填多个（逗号分隔），例如 `deepseek-chat, qwen-max`。

也可以用环境变量兜底：`LLM_BACKEND` / `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。

### 编程方式调用

```python
from core.pipeline import AnnotationPipeline
from core.models import AgentPresets, AnnotatorPreset
from llm.backend import get_backend

pipeline = AnnotationPipeline(backend=get_backend("mock"))
pipeline.create_config(
    industry_id="medical",
    object_id="medical_ner",
    presets=AgentPresets(annotator=AnnotatorPreset.RIGOROUS),
)

out = pipeline.run_full_pipeline(["患者持续高烧三天", "确诊为病毒性肺炎"])
print(out["inspection"])        # 一致性指标 + 分歧样本
print(out["results"][0].label)  # 最终标签
```

---

## 目录结构

```
core/
  models.py      注册表(行业/对象)、金标三集、标注指南、Agent 预设、结果与配置模型
  pipeline.py    流程编排：配置 → 标注 → 检验 → 裁决 → 复核 → 监控 → 成本 → 闭环
  exporter.py    CSV / JSONL / JSON / Markdown / Excel 导出
agents/
  annotator.py      标注 Agent：单 AI / 多 AI，置信度与难度分级，可解释输出
  inspector.py      检验 Agent：一致性计算 + 分歧样本定位
  adjudicator.py    裁决 Agent：多数投票 / 加权聚合 / 人工裁决
  reviewer.py       复核 Agent：抽检 / 疑难 / 全量，应用人工修正
  guideline_agent.py 指南 Agent：LLM 起草 + 分歧驱动的版本化优化
llm/
  backend.py     LLM 抽象层：OpenAI 兼容后端 + mock 模拟后端 + 厂商预设 + 连接测试
quality/
  agreement.py   Cohen's κ / Fleiss' κ / Krippendorff's α / Landis-Koch 解读
  monitor.py     过程监控(四档停止条件) + 成本与耗时预估
ui/
  app.py         Gradio 七步式界面
tests/
  test_pipeline.py  端到端流程测试
  test_units.py     单元测试(10 组，覆盖各核心模块)
测试数据/
  reference_results/  放置本地参考结果（不随仓库提交内容）
```

---

## Agent 预设说明

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
| | `manual` | 由使用者自己决定何时停 |

---

## 测试

```bash
python tests/test_pipeline.py   # 端到端：模拟模式跑通全流程
python tests/test_units.py      # 单元测试：10 组，覆盖各核心模块
```

单元测试覆盖：`agreement` 一致性指标、`monitor` 监控与成本、`models` 注册表、`annotator` 标注、`guideline_agent` 指南、`inspector` 检验、`adjudicator` 裁决、`reviewer` 复核、`exporter` 导出、`pipeline` 增量标注。

---

## 扩展新领域

新增一个行业或研究对象不需要改核心逻辑，在 `core/models.py` 的 `_register_defaults()` 里注册即可：

```python
# 新行业
Industry(id="energy", name="能源", description="电力、油气、新能源", compliance=[])

# 新研究对象
ResearchObject(
    id="energy_defect_class",
    name="光伏板缺陷分类",
    data_type=DataType.IMAGE,
    task_type=TaskType.IMAGE_CLASSIFICATION,
    description="对光伏组件 EL 图像进行缺陷分类",
    label_schema={"labels": ["正常", "隐裂", "断栅", "黑斑"]},
    industry_id="energy",   # 显式关联所属行业
)
```

注册后界面下拉框会自动出现，标注 / 检验 / 裁决 / 复核 / 导出全链路直接复用。

---

## 技术栈

Python 3.10+ · Gradio（界面）· Pydantic v2（数据模型）· OpenAI SDK（兼容任意 OpenAI 协议服务）· openpyxl（Excel 导出）。核心依赖仅 4 个，无数据库、无外部服务。

---

## 已知限制

- **图像 / 多模态任务目前只做了数据模型层面的支持**，界面第 ③ 步的输入是纯文本，图像类任务需要自行扩展输入与渲染逻辑。
- **Krippendorff's α 为简化实现**：对名义数据复用 Fleiss' κ，且会丢弃含缺失值的样本；需要严格 α 值请引入 `krippendorff` 库替换 `quality/agreement.py`。
- **状态存在内存里**：单进程单会话，重启后标注结果与指南版本会丢失，导出请及时。
- **成本单价为硬编码估算值**，未随厂商调价更新，仅作量级参考。
- 界面以 `0.0.0.0` 启动且无鉴权，**仅建议本地或内网使用**，不要直接暴露到公网。

---

## Roadmap

- [ ] 图像 / 多模态输入的界面支持（缩略图预览 + 检测框标注）
- [ ] 标注结果持久化（SQLite），支持任务断点恢复
- [ ] 引入标准 `krippendorff` 库替换简化实现
- [ ] 支持导入现成标注数据集（CoNLL / JSONL / Label Studio 格式）
- [ ] 标注指南的 diff 视图与版本回滚

---

## License

[MIT](LICENSE) © 2026 李子桐
