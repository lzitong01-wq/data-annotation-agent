"""
Web 界面(Gradio)
美观、结构井然、通用
流程:① 选行业对象 → ② 配 LLM 与方案 → ③ 输入数据 → ④ 标注 → ⑤ 检验/复核/反馈
支持运行时配置真实 LLM(DeepSeek/通义/OpenAI/自定义),无需改代码
启动:python -m data_annotation_agent  或  data-annotation-agent
"""
import os
from typing import Any, Dict, List, Optional

import gradio as gr

from data_annotation_agent.core.exporter import EXPORTERS, export_results
from data_annotation_agent.core.models import (
    AdjudicatorPreset,
    AgentPresets,
    AnnotationGuideline,
    AnnotatorPreset,
    GoldenData,
    GoldSample,
    GoldSet,
    InspectorPreset,
    MonitorPreset,
    Registry,
    ReviewerPreset,
)
from data_annotation_agent.core.pipeline import AnnotationPipeline
from data_annotation_agent.llm.backend import PROVIDER_PRESETS
from data_annotation_agent.quality.monitor import CostEstimator

# 界面配色(统一、清晰)
PRIMARY = "#2563eb"
BG = "#f8fafc"
CARD = "#ffffff"
TEXT = "#1e293b"
MUTED = "#64748b"

# 金标模式常量(避免硬编码中文比较)
GOLDEN_MODE_NONE = "无金标"
GOLDEN_MODE_REF = "提供金标(参考集)"

# 自定义 CSS:应用配色、卡片化、步骤进度条
CUSTOM_CSS = f"""
:root {{
  --primary: {PRIMARY};
  --bg: {BG};
  --card: {CARD};
  --text: {TEXT};
  --muted: {MUTED};
}}
body {{ background: var(--bg); }}
.gradio-container {{ max-width: 1200px !important; margin: 0 auto !important; }}
.step-badge {{
  display: inline-block; background: var(--primary); color: #fff;
  border-radius: 999px; padding: 2px 12px; font-size: 13px; font-weight: 600;
  margin-right: 6px;
}}
.step-nav {{
  display: flex; justify-content: space-between; gap: 8px;
  background: var(--card); border: 1px solid #e2e8f0; border-radius: 12px;
  padding: 12px 16px; margin-bottom: 16px;
}}
.step-nav .step {{
  flex: 1; text-align: center; font-size: 13px; color: var(--muted);
  padding: 6px 4px; border-radius: 8px;
}}
.step-nav .step.active {{ background: #eff6ff; color: var(--primary); font-weight: 600; }}
.card-panel {{
  background: var(--card); border: 1px solid #e2e8f0; border-radius: 12px;
  padding: 16px; margin-bottom: 12px;
}}
"""


class AnnotationUI:
    """数据标注智能体 Web 界面"""
    def __init__(self):
        self.registry = Registry()
        self.pipeline: Optional[AnnotationPipeline] = None
        self.samples: List[str] = []
        self.results_display = []
        self._inspection_cache: Optional[Dict[str, Any]] = None
        self._llm_state = {"backend": "mock", "base_url": "", "api_key": "", "models": []}

    # ============ 界面构建 ============

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="数据标注智能体") as demo:
            # ---------- 顶部横幅 ----------
            gr.Markdown(
                """
                <div style="text-align:center; padding:18px 0 6px 0;">
                  <div style="font-size:30px; font-weight:700; color:#1e293b;">🧠 数据标注智能体</div>
                  <div style="font-size:15px; color:#64748b; margin-top:6px;">
                    通用数据标注框架 · 先选行业与研究对象,系统推荐方案,你自由选择 · 支持真实 LLM 与模拟模式
                  </div>
                </div>
                """
            )

            # ---------- 步骤导航(视觉进度) ----------
            gr.Markdown(
                """
                <div class="step-nav">
                  <div class="step active">① 选行业对象</div>
                  <div class="step">② 配 LLM 与方案</div>
                  <div class="step">③ 输入数据</div>
                  <div class="step">④ 标注</div>
                  <div class="step">⑤ 检验/复核/反馈</div>
                  <div class="step">⑥ 中途修改</div>
                  <div class="step">⑦ 导出结果</div>
                </div>
                """
            )

            # ============ ① 选择行业与对象 ============
            with gr.Tab("① 选择行业与对象"):
                with gr.Row():
                    with gr.Column(scale=1):
                        industry_dd = gr.Dropdown(
                            choices=[i.name for i in self.registry.list_industries()],
                            label="🏭 选择行业", value="医疗健康"
                        )
                    with gr.Column(scale=1):
                        object_dd = gr.Dropdown(
                            choices=self._object_names_for_industry("医疗健康"),
                            label="🎯 选择研究对象", value="医疗实体识别",
                            info="选项随所选行业联动刷新"
                        )
                object_info = gr.Markdown(self._show_object_info("医疗健康", "医疗实体识别"))
                industry_dd.change(
                    self._on_industry_change,
                    inputs=[industry_dd], outputs=[object_dd, object_info]
                )
                object_dd.change(
                    lambda obj: self._show_object_info(industry_dd.value, obj),
                    inputs=[object_dd], outputs=[object_info]
                )

            # ============ ② 配置 LLM 与标注方案 ============
            with gr.Tab("② 配置 LLM 与方案"):
                gr.Markdown("### 🤖 第一步:选择标注引擎(LLM 后端)")
                with gr.Row():
                    llm_backend_dd = gr.Dropdown(
                        choices=list(PROVIDER_PRESETS.keys()),
                        label="标注引擎",
                        value="mock",
                        info="mock=模拟模式(无需 Key);其他需填写下方 API Key"
                    )
                    llm_docs = gr.Markdown(PROVIDER_PRESETS["mock"]["docs"])
                with gr.Row():
                    base_url_tb = gr.Textbox(
                        label="API 地址(Base URL)", value="",
                        placeholder="选择厂商后自动填充,自定义时手动填写"
                    )
                    api_key_tb = gr.Textbox(
                        label="API Key", value="", type="password",
                        placeholder="粘贴你的 API Key(仅本次会话使用,不落盘)"
                    )
                with gr.Row():
                    models_tb = gr.Textbox(
                        label="参与标注的模型(逗号分隔;多模型互检需选 rigorous 预设)",
                        value="mock-1, mock-2, mock-3",
                        placeholder="例如: deepseek-chat, deepseek-reasoner"
                    )
                with gr.Row():
                    test_btn = gr.Button("🔌 测试连接", variant="secondary")
                    test_output = gr.Markdown("**连接状态:** 未测试")

                gr.Markdown("### 🧩 第二步:选择各 Agent 预设")
                with gr.Row():
                    annotator_preset_dd = gr.Dropdown(
                        choices=[p.value for p in AnnotatorPreset],
                        label="标注 Agent 预设", value="standard",
                        info="fast=单AI零样本 / standard=单AI+参考集 / rigorous=多AI互检"
                    )
                    inspector_preset_dd = gr.Dropdown(
                        choices=[p.value for p in InspectorPreset],
                        label="检验 Agent 预设", value="standard",
                        info="light=Cohen's κ / standard=Fleiss' κ / deep=多指标+互检"
                    )
                with gr.Row():
                    adjudicator_preset_dd = gr.Dropdown(
                        choices=[p.value for p in AdjudicatorPreset],
                        label="裁决 Agent 预设", value="majority",
                        info="majority=多数投票 / weighted=加权 / human=人工"
                    )
                    reviewer_preset_dd = gr.Dropdown(
                        choices=[p.value for p in ReviewerPreset],
                        label="复核 Agent 预设", value="difficult",
                        info="sample=抽检 / difficult=疑难 / full=全量"
                    )
                with gr.Row():
                    monitor_preset_dd = gr.Dropdown(
                        choices=[p.value for p in MonitorPreset],
                        label="监控预设", value="balanced",
                        info="conservative=κ>0.8 / balanced=κ>0.6 / lenient=标完 / manual=手动"
                    )

                gr.Markdown("### 📋 第三步:标注指南与金标(可选)")
                with gr.Row():
                    with gr.Column(scale=1):
                        guideline_mode = gr.Radio(
                            choices=["无指南", "系统起草指南", "用户提供指南"],
                            label="标注指南", value="系统起草指南"
                        )
                        guideline_text = gr.Textbox(
                            label="用户指南(选'用户提供指南'时填写)",
                            lines=4, placeholder="粘贴你的标注指南..."
                        )
                    with gr.Column(scale=1):
                        golden_mode = gr.Radio(
                            choices=[GOLDEN_MODE_NONE, GOLDEN_MODE_REF],
                            label="金标数据", value=GOLDEN_MODE_NONE
                        )
                        golden_text = gr.Textbox(
                            label="金标参考集(格式: 数据|标签,每行一条)",
                            lines=4, placeholder="例如:\n这个药很有效|正面\n发烧咳嗽|症状"
                        )
                with gr.Row():
                    explainability_cb = gr.Checkbox(label="启用可解释性(输出标注理由)", value=True)
                    cost_btn = gr.Button("💰 预估成本与时间", variant="secondary")
                    cost_output = gr.Markdown("**成本预估:** 待计算")

            # ============ ③ 输入待标注数据 ============
            with gr.Tab("③ 输入待标注数据"):
                samples_text = gr.Textbox(
                    label="待标注数据(每行一条)",
                    lines=8, placeholder="输入要标注的文本,每行一条..."
                )
                with gr.Row():
                    start_btn = gr.Button("🚀 开始标注", variant="primary", size="lg")
                    clear_btn = gr.Button("清空", variant="secondary")

            # ============ ④ 标注结果 ============
            with gr.Tab("④ 标注结果"):
                results_md = gr.Markdown("**标注结果:** 待标注")
                results_df = gr.Dataframe(
                    headers=["样本", "标签", "置信度", "标注者", "难度", "理由"],
                    datatype=["str", "str", "number", "str", "str", "str"],
                    label="标注结果明细", interactive=False
                )
                with gr.Row():
                    route_btn = gr.Button("🧭 难度分级路由", variant="secondary")
                    route_output = gr.Markdown("**难度路由:** 待标注后查看")
                with gr.Row():
                    inspect_btn = gr.Button("🔍 一致性检验 / AI互检", variant="secondary")
                    inspect_output = gr.Markdown("**检验结果:** 待检验")
                with gr.Row():
                    review_btn = gr.Button("👁 人工复核", variant="secondary")
                    review_output = gr.Markdown("**复核列表:** 待复核")
                with gr.Row():
                    feedback_btn = gr.Button("🔄 反馈闭环(优化指南)", variant="secondary")
                    feedback_output = gr.Markdown("**反馈闭环:** 待执行")

            # ============ ⑤ 版本管理 ============
            with gr.Tab("⑤ 版本管理"):
                gr.Markdown("### 📚 标注指南版本历史")
                gr.Markdown("**版本历史:** 待标注后查看")
                with gr.Row():
                    version_btn = gr.Button("🕘 查看版本历史", variant="secondary")
                    version_output = gr.Markdown("**版本历史:** 待查看")

            # ============ ⑥ 中途修改机制 ============
            with gr.Tab("⑥ 中途修改"):
                gr.Markdown("### 🔧 分批标注与中途修改")
                gr.Markdown(
                    "在标注过程中可随时暂停、追加新批次(断点续标)、修改指南/预设/金标后继续,"
                    "或重标已完成批次。适合长任务与迭代式标注。"
                )
                with gr.Row():
                    batch_samples = gr.Textbox(
                        label="本批次待标注数据(每行一条)",
                        lines=5, placeholder="输入本批次要标注的文本,每行一条..."
                    )
                with gr.Row():
                    append_btn = gr.Button("➕ 追加标注本批次(断点续标)", variant="primary")
                    relabel_btn = gr.Button("♻️ 重标本批次(覆盖已有结果)", variant="secondary")
                    reset_btn = gr.Button("🗑 清空全部结果", variant="secondary")
                batch_output = gr.Markdown("**中途修改:** 待操作")
                gr.Markdown("### ✏️ 批间修改(修改后继续标注)")
                with gr.Row():
                    with gr.Column(scale=1):
                        mid_guideline_mode = gr.Radio(
                            choices=["无指南", "系统起草指南", "用户提供指南"],
                            label="批间修改标注指南", value="系统起草指南"
                        )
                        mid_guideline_text = gr.Textbox(
                            label="批间用户指南(选'用户提供指南'时填写)",
                            lines=3, placeholder="粘贴新的标注指南..."
                        )
                    with gr.Column(scale=1):
                        mid_annotator_preset = gr.Dropdown(
                            choices=[p.value for p in AnnotatorPreset],
                            label="批间修改标注预设", value="standard"
                        )
                        mid_golden_text = gr.Textbox(
                            label="批间修改金标参考集(格式: 数据|标签,每行一条)",
                            lines=3, placeholder="例如:\
这个药很有效|正面"
                        )
                with gr.Row():
                    apply_mid_btn = gr.Button("💾 应用批间修改", variant="secondary")
                    mid_apply_output = gr.Markdown("**批间修改:** 待应用")

            # ============ ⑦ 导出结果 ============
            with gr.Tab("⑦ 导出结果"):
                gr.Markdown("### 📤 导出标注结果")
                gr.Markdown("支持 CSV / JSONL / JSON / Excel(xlsx)/ Markdown 多种格式。")
                with gr.Row():
                    export_fmt_dd = gr.Dropdown(
                        choices=list(EXPORTERS.keys()),
                        label="导出格式", value="csv"
                    )
                    export_path_tb = gr.Textbox(
                        label="导出文件路径(相对项目根目录)",
                        value="exports/annotation_results.csv",
                        placeholder="例如: exports/annotation_results.csv"
                    )
                with gr.Row():
                    export_btn = gr.Button("📥 导出标注结果", variant="primary")
                    export_output = gr.Markdown("**导出结果:** 待导出")

            # ============ 事件绑定 ============

            # 厂商切换:自动填充 base_url 与默认模型
            llm_backend_dd.change(
                lambda b: (PROVIDER_PRESETS[b]["base_url"], PROVIDER_PRESETS[b]["models_default"],
                           PROVIDER_PRESETS[b]["docs"]),
                inputs=[llm_backend_dd],
                outputs=[base_url_tb, models_tb, llm_docs]
            )

            # 测试连接
            test_btn.click(
                lambda b, u, k, m: self._test_connection(b, u, k, m),
                inputs=[llm_backend_dd, base_url_tb, api_key_tb, models_tb],
                outputs=[test_output]
            )

            # 成本预估(独立于 pipeline,不依赖已标注)
            cost_btn.click(
                lambda backend, models, n: self._estimate_cost(backend, models, n),
                inputs=[llm_backend_dd, models_tb, samples_text],
                outputs=[cost_output]
            )

            # 开始标注
            start_btn.click(
                lambda ind, obj, backend, base_url, api_key, models, ap, ip, adjp, rp, mp,
                       gmode, gtext, gmode2, gtext2, exp, samples:
                self._run_annotation(ind, obj, backend, base_url, api_key, models,
                                     ap, ip, adjp, rp, mp, gmode, gtext, gmode2, gtext2, exp, samples),
                inputs=[industry_dd, object_dd, llm_backend_dd, base_url_tb, api_key_tb, models_tb,
                        annotator_preset_dd, inspector_preset_dd, adjudicator_preset_dd,
                        reviewer_preset_dd, monitor_preset_dd, guideline_mode, guideline_text,
                        golden_mode, golden_text, explainability_cb, samples_text],
                outputs=[results_md, results_df]
            )

            # 清空
            clear_btn.click(
                lambda: ("", "**标注结果:** 待标注", []),
                outputs=[samples_text, results_md, results_df]
            )

            # 难度分级路由
            route_btn.click(
                lambda: self._run_route(),
                outputs=[route_output]
            )

            # 版本历史
            version_btn.click(
                lambda: self._run_version(),
                outputs=[version_output]
            )

            # 一致性检验
            inspect_btn.click(
                lambda: self._run_inspection(),
                outputs=[inspect_output]
            )

            # 人工复核
            review_btn.click(
                lambda: self._run_review(),
                outputs=[review_output]
            )

            # 反馈闭环
            feedback_btn.click(
                lambda: self._run_feedback(),
                outputs=[feedback_output]
            )

            # ---- 中途修改机制(⑥) ----
            append_btn.click(
                lambda batch: self._run_batch_append(batch),
                inputs=[batch_samples], outputs=[batch_output]
            )
            relabel_btn.click(
                lambda batch: self._run_batch_relabel(batch),
                inputs=[batch_samples], outputs=[batch_output]
            )
            reset_btn.click(
                lambda: self._run_batch_reset(),
                outputs=[batch_output]
            )
            apply_mid_btn.click(
                lambda gmode, gtext, ap, gtext2: self._apply_mid_changes(gmode, gtext, ap, gtext2),
                inputs=[mid_guideline_mode, mid_guideline_text, mid_annotator_preset, mid_golden_text],
                outputs=[mid_apply_output]
            )

            # ---- 导出结果(⑦) ----
            export_btn.click(
                lambda fmt, path: self._run_export(fmt, path),
                inputs=[export_fmt_dd, export_path_tb], outputs=[export_output]
            )

        return demo

    # ============ 辅助方法 ============

    def _object_names_for_industry(self, industry_name: str) -> List[str]:
        """某个行业下可选的研究对象名称（行业→对象联动用）"""
        industry = next((i for i in self.registry.list_industries() if i.name == industry_name), None)
        if industry is None:
            # 行业名对不上时退回全量，避免下拉框变空
            return [o.name for o in self.registry.list_objects()]
        return [o.name for o in self.registry.list_objects_by_industry(industry.id)]

    def _on_industry_change(self, industry_name: str):
        """切换行业：联动刷新研究对象选项，并同步下方说明文字"""
        names = self._object_names_for_industry(industry_name)
        first = names[0] if names else None
        return gr.update(choices=names, value=first), self._show_object_info(industry_name, first or "")

    def _show_object_info(self, industry_name: str, object_name: str) -> str:
        industry = next((i for i in self.registry.list_industries() if i.name == industry_name), None)
        obj = next((o for o in self.registry.list_objects() if o.name == object_name), None)
        if not obj:
            return "**对象信息:** 未找到"
        labels = obj.label_schema.get("labels", []) if obj.label_schema else []
        compliance = industry.compliance if industry else []
        return (
            f"**对象:** {obj.name}\n"
            f"**类型:** {obj.data_type.value} / {obj.task_type.value}\n"
            f"**描述:** {obj.description}\n"
            f"**标签:** {labels}\n"
            f"**合规要求:** {compliance or '无'}"
        )

    def _test_connection(self, backend: str, base_url: str, api_key: str, models: str) -> str:
        """测试 LLM 连接"""
        if backend == "mock":
            return "**连接状态:** ✅ 模拟模式无需连接,可直接开始标注"
        if not api_key.strip():
            return "**连接状态:** ⚠️ 请先填写 API Key"
        model = models.strip().split(",")[0].strip() if models.strip() else ""
        if not self.pipeline:
            self.pipeline = AnnotationPipeline()
        result = self.pipeline.test_llm_connection(backend, base_url, api_key, model)
        icon = "✅" if result.get("ok") else "❌"
        return f"**连接状态:** {icon} {result.get('message', '')}"

    def _estimate_cost(self, backend: str, models: str, samples_text: str) -> str:
        """成本预估:独立于 pipeline,不依赖已标注"""
        n = len([s for s in samples_text.split("\n") if s.strip()])
        if n == 0:
            return "**成本预估:** 请输入待标注数据"
        model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else ["mock-model"]
        n_models = len(model_list)
        model = model_list[0] if backend != "mock" else "mock-model"
        est = CostEstimator().estimate(n, n_models, model=model)
        return (
            f"**成本预估:**\n"
            f"- 样本数: {est['n_samples']}\n"
            f"- 模型数: {est['n_models']}\n"
            f"- 预估 token: {est['total_tokens_est']}\n"
            f"- 预估成本: ${est['cost_est_usd']}\n"
            f"- 预估时间: {est['time_est_minutes']} 分钟"
        )

    def _run_annotation(self, industry_name, object_name, backend, base_url, api_key, models,
                        ap, ip, adjp, rp, mp, gmode, gtext, gmode2, gtext2, exp, samples_text) -> tuple:
        """执行标注,返回 (结果Markdown, 结果Dataframe)"""
        # 解析模型列表
        model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else ["mock-model"]

        # 真实后端标注前校验连接
        if backend != "mock" and not api_key.strip():
            return "**错误:** 真实 LLM 后端需要填写 API Key(第②步)", []

        # 创建 pipeline(运行时 LLM 配置)
        self.pipeline = AnnotationPipeline()
        industry = next((i for i in self.registry.list_industries() if i.name == industry_name), None)
        obj = next((o for o in self.registry.list_objects() if o.name == object_name), None)
        if not industry or not obj:
            return "**错误:** 无效的行业/对象", []

        # 构建金标
        golden = GoldenData()
        if gmode2 == GOLDEN_MODE_REF and gtext2.strip():
            samples = []
            for line in gtext2.strip().split("\n"):
                if "|" in line:
                    data, label = line.split("|", 1)
                    samples.append(GoldSample(data=data.strip(), label=label.strip()))
            golden.reference_set = GoldSet(name="参考集", samples=samples)

        # 构建指南
        guideline = None
        if gmode == "系统起草指南":
            guideline = self.pipeline.guideline_agent.draft(industry, obj)
        elif gmode == "用户提供指南" and gtext.strip():
            guideline = AnnotationGuideline(
                title=f"{obj.name}标注指南",
                description=gtext.strip(),
                rules=[gtext.strip()],
                source="user",
            )

        # 创建配置(真实后端需 base_url/api_key)
        presets = AgentPresets(
            annotator=AnnotatorPreset(ap),
            inspector=InspectorPreset(ip),
            adjudicator=AdjudicatorPreset(adjp),
            reviewer=ReviewerPreset(rp),
            monitor=MonitorPreset(mp),
        )
        try:
            self.pipeline.create_config(
                industry.id, obj.id, golden_data=golden, guideline=guideline,
                presets=presets, llm_backend=backend, llm_models=model_list,
                base_url=base_url, api_key=api_key,
            )
        except Exception as e:
            return f"**错误:** {type(e).__name__}: {str(e)[:200]}", []

        # 标注
        self.samples = [s for s in samples_text.split("\n") if s.strip()]
        if not self.samples:
            return "**错误:** 请输入待标注数据", []
        try:
            results = self.pipeline.annotate_samples(self.samples)
        except Exception as e:
            return f"**标注失败:** {type(e).__name__}: {str(e)[:200]}", []
        self.results_display = results
        self._inspection_cache = None  # 新标注后失效缓存

        # 展示结果(Markdown 摘要)
        lines = ["**标注结果:**", ""]
        for r in results:
            lines.append(f"- **{r.sample_id}** → `{r.label}` (置信度 {r.confidence:.2f}, 标注者 {r.annotator_id})")
            if exp and r.rationale:
                lines.append(f"  - 理由: {r.rationale}")
        md = "\n".join(lines)

        # 结构化明细(Dataframe,含难度列)
        df_rows = []
        for r in results:
            df_rows.append([
                r.sample_id, r.label, round(r.confidence, 3),
                r.annotator_id, r.difficulty, (r.rationale if exp else "")
            ])
        return md, df_rows

    def _get_inspection(self) -> Dict[str, Any]:
        """获取(并缓存)检验结果,避免重复计算"""
        if self._inspection_cache is None:
            self._inspection_cache = self.pipeline.inspect_results()
        return self._inspection_cache

    def _run_inspection(self) -> str:
        if not self.pipeline or not self.results_display:
            return "**检验结果:** 请先标注"
        inspection = self._get_inspection()
        agreement = inspection.get("agreement", {})
        lines = ["**检验结果:**", ""]
        lines.append(f"- 方法: {agreement.get('method', '无')}")
        if "value" in agreement:
            lines.append(f"- 一致性: {agreement.get('value')} ({agreement.get('interpretation', '')})")
        if "fleiss_kappa" in agreement:
            lines.append(f"- Fleiss' κ: {agreement.get('fleiss_kappa')}")
            lines.append(f"- Krippendorff's α: {agreement.get('krippendorff_alpha')}")
        lines.append(f"- 分歧样本数: {inspection.get('n_disagreements', 0)}")
        return "\n".join(lines)

    def _run_review(self) -> str:
        if not self.pipeline or not self.results_display:
            return "**复核列表:** 请先标注"
        inspection = self._get_inspection()
        review_list = self.pipeline.review_results(inspection.get("disagreements", []))
        lines = ["**复核列表:**", ""]
        if not review_list:
            lines.append("无需要复核的样本")
        for item in review_list:
            lines.append(f"- {item.get('sample_id')} (原因: {item.get('reason', '')})")
        return "\n".join(lines)

    def _run_feedback(self) -> str:
        if not self.pipeline or not self.results_display:
            return "**反馈闭环:** 请先标注"
        inspection = self._get_inspection()
        updated = self.pipeline.feedback_loop(inspection.get("disagreements", []))
        if updated:
            return f"**反馈闭环:** 指南已优化到版本 {updated.version},新增规则 {len(updated.rules)} 条"
        return "**反馈闭环:** 无分歧或无需优化指南"

    def _run_route(self) -> str:
        """难度分级路由:按难度分组,简单直接采用,疑难进入复核"""
        if not self.pipeline or not self.results_display:
            return "**难度路由:** 请先标注"
        routing = self.pipeline.route_by_difficulty()
        stats = routing["stats"]
        lines = ["**难度分级路由:**", ""]
        lines.append(f"- 总样本: {stats['total']}")
        lines.append(f"- 🟢 简单(simple): {stats['simple']} 条 → 直接采用")
        lines.append(f"- 🟡 中等(medium): {stats['medium']} 条 → 常规处理")
        lines.append(f"- 🔴 疑难(difficult): {stats['difficult']} 条 → 进入人工复核")
        lines.append("")
        lines.append("**疑难样本明细:**")
        difficult = routing["buckets"]["difficult"]
        if not difficult:
            lines.append("无疑难样本")
        for r in difficult:
            lines.append(f"- {r.sample_id} → `{r.label}` (置信度 {r.confidence:.2f})")
        return "\n".join(lines)

    def _run_version(self) -> str:
        """版本管理:展示指南版本历史与反馈日志"""
        if not self.pipeline or not self.pipeline.config:
            return "**版本历史:** 请先标注"
        guideline = self.pipeline.config.guideline
        lines = ["**标注指南版本历史:**", ""]
        if guideline:
            lines.append(f"- 当前版本: v{guideline.version}")
            lines.append(f"- 标题: {guideline.title}")
            lines.append(f"- 来源: {guideline.source}(user=用户/drafted=系统起草/optimized=系统优化)")
            lines.append(f"- 规则数: {len(guideline.rules)}")
            lines.append(f"- 边界情况: {len(guideline.boundary_cases)}")
            lines.append(f"- 创建时间: {guideline.created_at.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"- 更新时间: {guideline.updated_at.strftime('%Y-%m-%d %H:%M')}")
        else:
            lines.append("未配置标注指南")
        lines.append("")
        lines.append("**反馈闭环日志(可追溯):**")
        log = getattr(self.pipeline, "feedback_log", [])
        if not log:
            lines.append("暂无反馈闭环记录")
        for entry in log:
            lines.append(f"- [{entry.get('type')}] 版本 v{entry.get('version')}, 分歧样本 {entry.get('n_disagreements')} 条")
        return "\n".join(lines)



    # ============ 中途修改机制(⑥) ============

    def _require_pipeline(self) -> Optional[str]:
        """校验 pipeline 与配置是否就绪,返回错误信息或 None"""
        if not self.pipeline or not self.pipeline.config:
            return "**中途修改:** 请先在②③④步完成首次标注"
        return None

    def _run_batch_append(self, batch: str) -> str:
        """断点续标:追加标注本批次"""
        err = self._require_pipeline()
        if err:
            return err
        samples = [s for s in batch.split("\n") if s.strip()]
        if not samples:
            return "**中途修改:** 请输入本批次待标注数据"
        try:
            new_results = self.pipeline.annotate_batch(samples, append=True)
        except Exception as e:
            return f"**追加失败:** {type(e).__name__}: {str(e)[:200]}"
        self.results_display = self.pipeline.results
        self._inspection_cache = None
        return (f"**追加标注成功:** 本批次 {len(new_results)} 条,累计 {len(self.pipeline.results)} 条。"
                f"可继续追加或进入⑤检验/复核/反馈。")

    def _run_batch_relabel(self, batch: str) -> str:
        """重标本批次:覆盖已有结果"""
        err = self._require_pipeline()
        if err:
            return err
        samples = [s for s in batch.split("\n") if s.strip()]
        if not samples:
            return "**中途修改:** 请输入本批次待标注数据"
        try:
            new_results = self.pipeline.annotate_batch(samples, append=False)
        except Exception as e:
            return f"**重标失败:** {type(e).__name__}: {str(e)[:200]}"
        self.results_display = self.pipeline.results
        self._inspection_cache = None
        return f"**重标成功:** 已覆盖为 {len(new_results)} 条结果。"

    def _run_batch_reset(self) -> str:
        """清空全部结果"""
        err = self._require_pipeline()
        if err:
            return err
        self.pipeline.reset_results()
        self.results_display = []
        self._inspection_cache = None
        return "**已清空全部标注结果与反馈日志。** 可重新标注。"

    def _apply_mid_changes(self, gmode: str, gtext: str, ap: str, gtext2: str) -> str:
        """批间修改:更新指南/预设/金标"""
        err = self._require_pipeline()
        if err:
            return err
        applied = []
        # 修改指南
        if gmode == "系统起草指南":
            industry = next((i for i in self.registry.list_industries()
                             if i.id == self.pipeline.config.industry_id), None)
            obj = next((o for o in self.registry.list_objects()
                        if o.id == self.pipeline.config.object_id), None)
            if industry and obj:
                new_guideline = self.pipeline.guideline_agent.draft(industry, obj)
                self.pipeline.update_guideline(new_guideline)
                applied.append(f"指南(系统起草 v{new_guideline.version})")
        elif gmode == "用户提供指南" and gtext.strip():
            new_guideline = AnnotationGuideline(
                title="批间修改指南", description=gtext.strip(),
                rules=[gtext.strip()], source="user",
            )
            self.pipeline.update_guideline(new_guideline)
            applied.append("指南(用户提供)")
        # 修改预设
        if ap:
            presets = self.pipeline.config.presets
            presets.annotator = AnnotatorPreset(ap)
            self.pipeline.update_presets(presets)
            applied.append(f"标注预设({ap})")
        # 修改金标
        if gtext2.strip():
            samples = []
            for line in gtext2.strip().split("\n"):
                if "|" in line:
                    data, label = line.split("|", 1)
                    samples.append(GoldSample(data=data.strip(), label=label.strip()))
            if samples:
                golden = self.pipeline.config.golden_data or GoldenData()
                golden.reference_set = GoldSet(name="参考集(批间修改)", samples=samples)
                self.pipeline.update_golden_data(golden)
                applied.append(f"金标参考集({len(samples)}条)")
        if not applied:
            return "**批间修改:** 未检测到有效修改内容"
        return "**批间修改已应用:** " + ", ".join(applied) + "。可继续追加标注。"

    # ============ 导出结果(⑦) ============

    def _run_export(self, fmt: str, path: str) -> str:
        """导出标注结果为指定格式"""
        if not self.pipeline or not self.pipeline.results:
            return "**导出结果:** 请先标注"
        if not path.strip():
            return "**导出结果:** 请填写导出文件路径"
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            export_results(self.pipeline.results, fmt, path)
        except Exception as e:
            return f"**导出失败:** {type(e).__name__}: {str(e)[:200]}"
        return f"**导出成功:** 已导出 {len(self.pipeline.results)} 条结果为 `{fmt}` 格式 → `{path}`"


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """启动 Web 界面

    默认只监听回环地址：界面没有鉴权，不应直接暴露到公网。
    需要局域网访问时显式传 ``host="0.0.0.0"``。
    """
    ui = AnnotationUI()
    demo = ui.build()
    demo.launch(server_name=host, server_port=port, share=share, css=CUSTOM_CSS)


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口：``data-annotation-agent [--host HOST] [--port PORT] [--share]``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="data-annotation-agent",
        description="数据标注智能体 —— Gradio 七步式标注界面",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址（默认 127.0.0.1；局域网访问用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=7860, help="监听端口（默认 7860）")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网临时链接")
    args = parser.parse_args(argv)

    launch(host=args.host, port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

