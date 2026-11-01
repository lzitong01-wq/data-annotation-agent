# 架构说明

## 分层

```
┌───────────────────────────────────────────────────────────────┐
│  ui/          Gradio 七步式界面（唯一依赖 gradio 的层）        │
└──────────────────────────────┬────────────────────────────────┘
                               │ 只调用 pipeline 的公开方法
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

依赖方向自上而下，没有反向依赖：`core/models.py` 不导入任何其他内部模块，`llm/` 与 `quality/` 互不依赖。

## 一次完整调用的数据流

```
样本列表
  │
  ├─ [annotate]    每个样本 × 每个模型 → AnnotationResult(label, confidence, rationale, difficulty)
  │                多 AI 模式下同一样本会产生 N 条结果，sample_id 相同
  │
  ├─ [inspect]     按 sample_id 分组 → 标注矩阵 → Cohen's κ / Fleiss' κ / α
  │                标签不一致的样本被收集为 disagreements
  │
  ├─ [adjudicate]  同一 sample_id 的多条结果聚合为一条（多数投票 / 加权 / 人工）
  │
  ├─ [review]      按预设挑出待复核样本（抽检 5% / 疑难 / 全量）
  │
  ├─ [feedback]    disagreements 回流给 GuidelineAgent，追加规则与边界情况，版本 +0.1
  │
  └─ [export]      CSV / JSONL / JSON / Excel / Markdown
```

## 关键设计决策

### 注册表驱动，而非硬编码领域

行业（`Industry`）与研究对象（`ResearchObject`）都放在 `Registry` 里。新增领域只需注册条目，标注 / 检验 / 裁决 / 复核 / 导出全链路自动复用，不需要写新代码。

研究对象用 `industry_id` **显式**关联行业；早期版本靠 id 前缀（`medical_*`）推断，通用对象 `ner`、`text_classification` 因此挂不到任何行业下，这个坑已经修掉，前缀推断仅作为兼容旧对象的兜底。

### 金标三集分离

| 集合 | 作用 | 是否必需 |
| --- | --- | --- |
| 参考集 `reference_set` | 塞进 prompt 当 few-shot 示例，引导模型对齐标签口径 | 可选 |
| 优化集 `optimization_set` | 调优与校准（当前版本仅建模，未接入自动调优） | 可选 |
| 测试集 `test_set` | 独立评估，不参与 prompt | 可选 |

三者分离是为了避免「用测试集示例去引导模型，再拿测试集算准确率」这种自欺欺人的评估。

### 多 AI 互检用独立后端实例

严谨模式下每个模型一个 `LLMBackend` 实例，而不是共用一个客户端改 `model` 参数。这样不同模型可以指向不同厂商、不同 base_url、不同 Key，互检才有意义。

### 分歧回流到指南，而不是回流到模型

`feedback_loop` 不改模型参数，只把分歧样本交给 `GuidelineAgent.optimize()`，让它追加规则与边界情况。理由是：标注质量问题的根因通常是**指南歧义**，而不是模型能力。版本号用 `Decimal` 递增，避免 `1.1 + 0.1 = 1.2000000000000002` 这种浮点误差污染版本号。

### 状态全在内存

`AnnotationPipeline` 把配置、结果、反馈日志都放在实例属性里，没有数据库。好处是零依赖、易测试；代价是单进程单会话、重启即丢。需要持久化请自行包一层（见 Roadmap）。

## 模块职责

| 模块 | 职责 | 对外关键方法 |
| --- | --- | --- |
| `core/models.py` | 数据模型与注册表 | `Registry`、`AnnotationConfig`、`AnnotationGuideline`、`AgentPresets` |
| `core/pipeline.py` | 流程编排 | `create_config`、`annotate_batch`、`run_full_pipeline`、`feedback_loop` |
| `core/exporter.py` | 结果导出 | `export_results`、`EXPORTERS` |
| `agents/annotator.py` | 标注 | `annotate`、`_classify_difficulty` |
| `agents/inspector.py` | 一致性检验 | `inspect`、`find_disagreements`、`cross_check` |
| `agents/adjudicator.py` | 裁决 | `majority_vote`、`weighted_aggregate`、`human_adjudicate` |
| `agents/reviewer.py` | 复核 | `select_for_review`、`apply_review` |
| `agents/guideline_agent.py` | 指南起草与优化 | `draft`、`optimize` |
| `llm/backend.py` | LLM 接入 | `LLMFactory.create`、`get_backend`、`test_connection` |
| `quality/agreement.py` | 一致性指标 | `cohens_kappa`、`fleiss_kappa`、`krippendorff_alpha` |
| `quality/monitor.py` | 监控与成本 | `MonitorAgent.should_stop`、`CostEstimator.estimate` |
| `ui/app.py` | Web 界面 | `AnnotationUI.build`、`launch`、`main` |

## 并发与线程安全

当前实现**不是**线程安全的：`AnnotationPipeline` 的 `results`、`config` 都是可变实例状态，Gradio 多用户同时操作同一个 `AnnotationUI` 实例会互相覆盖。单机单用户演示没问题，多用户需要按会话隔离 pipeline 实例。
