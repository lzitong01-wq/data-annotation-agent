"""
主流程编排器
把整个流程串起来:选行业→选对象→选AI→选指南→选预设→标注→检验→复核→反馈闭环
"""
import os
from typing import List, Dict, Any, Optional
from .models import (
    Registry, GoldenData, AnnotationGuideline, AgentPresets,
    AnnotationConfig, AnnotationResult, AnnotatorPreset,
    InspectorPreset, AdjudicatorPreset, ReviewerPreset, MonitorPreset,
    GoldSet, GoldSample, Industry, ResearchObject
)
from llm.backend import LLMBackend, get_backend, LLMFactory, MockBackend
from agents.annotator import AnnotatorAgent
from agents.inspector import InspectorAgent
from agents.adjudicator import AdjudicatorAgent
from agents.reviewer import ReviewerAgent
from agents.guideline_agent import GuidelineAgent
from quality.monitor import MonitorAgent, CostEstimator


class AnnotationPipeline:
    """
    数据标注主流程
    支持通用框架,行业/对象可扩展
    """
    def __init__(self, backend: LLMBackend = None, registry: Registry = None):
        self.registry = registry or Registry()
        self.backend = backend or get_backend()
        self.annotator = AnnotatorAgent(self.backend, models=["mock-model"])
        self.inspector = InspectorAgent()
        self.adjudicator = AdjudicatorAgent()
        self.reviewer = ReviewerAgent()
        self.guideline_agent = GuidelineAgent(self.backend)
        self.monitor = MonitorAgent()
        self.cost_estimator = CostEstimator()
        self.config: Optional[AnnotationConfig] = None
        self.results: List[AnnotationResult] = []
        self.feedback_log: List[Dict[str, Any]] = []

    # ============ 配置阶段 ============

    def create_config(self, industry_id: str, object_id: str,
                      golden_data: Optional[GoldenData] = None,
                      guideline: Optional[AnnotationGuideline] = None,
                      presets: Optional[AgentPresets] = None,
                      llm_backend: str = "mock",
                      llm_models: List[str] = None,
                      base_url: str = None, api_key: str = None) -> AnnotationConfig:
        """
        创建标注配置
        llm_backend: mock / deepseek / qwen / openai / custom
        llm_models: 参与标注的模型列表(多 AI 互检时多个模型)
        base_url / api_key: 运行时配置(真实后端必填;mock 可空)
        """
        industry = self.registry.get_industry(industry_id)
        obj = self.registry.get_object(object_id)
        if not industry or not obj:
            raise ValueError(f"无效的行业/对象: {industry_id}/{object_id}")

        # 构建主后端(用于指南起草/优化等)
        self.backend = get_backend(llm_backend, base_url=base_url, api_key=api_key,
                                   model=llm_models[0] if llm_models else None)
        self.guideline_agent = GuidelineAgent(self.backend)

        # 为每个模型构建独立后端(多 AI 互检时各真实模型各用各的连接)
        model_backends: Dict[str, LLMBackend] = {}
        for m in (llm_models or ["mock-model"]):
            if llm_backend == "mock":
                model_backends[m] = MockBackend(model=m)
            else:
                model_backends[m] = get_backend(llm_backend, base_url=base_url,
                                                api_key=api_key, model=m)

        self.config = AnnotationConfig(
            industry_id=industry_id,
            object_id=object_id,
            golden_data=golden_data or GoldenData(),
            guideline=guideline,
            presets=presets or AgentPresets(),
            llm_backend=llm_backend,
            llm_models=llm_models or ["mock-model"],
        )
        # 更新 annotator 的模型列表与独立后端(用于多AI互检)
        self.annotator.models = self.config.llm_models
        self.annotator.backends = model_backends
        return self.config

    def test_llm_connection(self, llm_backend: str, base_url: str = None,
                            api_key: str = None, model: str = None) -> Dict[str, Any]:
        """测试 LLM 连接(正式标注前验证 Key/地址/模型是否可用)"""
        try:
            backend = get_backend(llm_backend, base_url=base_url, api_key=api_key, model=model)
            return backend.test_connection()
        except Exception as e:
            return {"ok": False, "message": f"配置错误: {type(e).__name__}: {str(e)[:200]}"}

    def get_object_info(self) -> Dict[str, Any]:
        """获取研究对象信息(用于 prompt)"""
        obj = self.registry.get_object(self.config.object_id)
        return {
            "id": obj.id,
            "name": obj.name,
            "description": obj.description,
            "label_schema": obj.label_schema,
            "data_type": obj.data_type,
            "task_type": obj.task_type,
        }

    # ============ 标注阶段 ============

    def annotate_samples(self, samples: List[Any]) -> List[AnnotationResult]:
        """标注一批样本(覆盖式:重置已有结果)"""
        return self.annotate_batch(samples, append=False)

    def annotate_batch(self, samples: List[Any], append: bool = True) -> List[AnnotationResult]:
        """
        增量标注一批样本(中途修改机制核心)
        append=True: 追加到已有结果(断点续标/分批标注)
        append=False: 覆盖已有结果(重标)
        批间可先修改指南/预设/金标,再继续标注
        """
        if not self.config:
            raise ValueError("请先创建配置")
        object_info = self.get_object_info()
        guideline = self.config.guideline
        reference_set = self.config.golden_data.reference_set if self.config.golden_data else None

        all_results = []
        for sample in samples:
            results = self.annotator.annotate(
                data=sample,
                preset=self.config.presets.annotator,
                guideline=guideline,
                reference_set=reference_set,
                object_info=object_info,
            )
            all_results.extend(results)
        if append:
            self.results = self.results + all_results
        else:
            self.results = all_results
        return all_results

    def reset_results(self) -> None:
        """清空已标注结果(用于重新开始)"""
        self.results = []
        self.feedback_log = []

    def update_guideline(self, guideline: AnnotationGuideline) -> None:
        """批间修改标注指南(中途修改机制)"""
        if not self.config:
            raise ValueError("请先创建配置")
        self.config.guideline = guideline

    def update_presets(self, presets: AgentPresets) -> None:
        """批间修改 Agent 预设(中途修改机制)"""
        if not self.config:
            raise ValueError("请先创建配置")
        self.config.presets = presets

    def update_golden_data(self, golden_data: GoldenData) -> None:
        """批间修改金标数据(中途修改机制)"""
        if not self.config:
            raise ValueError("请先创建配置")
        self.config.golden_data = golden_data

    # ============ 难度分级路由 ============

    def route_by_difficulty(self, results: List[AnnotationResult] = None) -> Dict[str, Any]:
        """
        难度分级路由:按样本难度分组
        简单样本直接采用,疑难样本进入人工复核
        返回 {simple: [...], medium: [...], difficult: [...], stats: {...}}
        """
        results = results or self.results
        buckets = {"simple": [], "medium": [], "difficult": []}
        for r in results:
            d = r.difficulty if r.difficulty in buckets else "medium"
            buckets[d].append(r)
        stats = {k: len(v) for k, v in buckets.items()}
        stats["total"] = len(results)
        return {"buckets": buckets, "stats": stats}

    # ============ 检验阶段 ============

    def inspect_results(self) -> Dict[str, Any]:
        """检验标注结果(一致性 + AI 互检)"""
        if not self.results:
            return {"agreement": {"method": "无", "value": 0.0}, "disagreements": []}
        return self.inspector.inspect(self.results, self.config.presets.inspector)

    # ============ 裁决阶段 ============

    def adjudicate_results(self, human_labels: Dict[str, Any] = None) -> List[AnnotationResult]:
        """
        裁决标注结果
        对多 AI 结果进行聚合,解决分歧
        """
        if not self.results:
            return []

        # 按 sample_id 分组
        groups: Dict[str, List[AnnotationResult]] = {}
        for r in self.results:
            groups.setdefault(r.sample_id, []).append(r)

        final_results = []
        for sample_id, group in groups.items():
            if len(group) == 1:
                # 单 AI:直接采用
                final_results.append(group[0])
            else:
                # 多 AI:裁决
                human_label = human_labels.get(sample_id) if human_labels else None
                final = self.adjudicator.adjudicate(
                    group, self.config.presets.adjudicator, human_label=human_label
                )
                final_results.append(final)

        self.results = final_results
        return final_results

    # ============ 复核阶段 ============

    def review_results(self, disagreements: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """选择需要复核的样本"""
        return self.reviewer.select_for_review(
            self.results, self.config.presets.reviewer, disagreements
        )

    def apply_review(self, review_decisions: Dict[str, Any]) -> List[AnnotationResult]:
        """应用人工复核结果"""
        self.results = self.reviewer.apply_review(self.results, review_decisions)
        return self.results

    # ============ 监控阶段 ============

    def monitor_progress(self, kappa: float, accuracy: Optional[float] = None,
                         n_labeled: int = 0, total: int = 0) -> Dict[str, Any]:
        """监控标注进度,判断是否停止"""
        self.monitor.record(kappa, accuracy, n_labeled, total)
        return self.monitor.should_stop(
            self.config.presets.monitor, kappa, accuracy, n_labeled, total
        )

    # ============ 成本预估 ============

    def estimate_cost(self, n_samples: int, n_models: int,
                      enable_explainability: bool = False) -> Dict[str, Any]:
        """预估成本和时间"""
        model = self.config.llm_models[0] if self.config.llm_models else "mock-model"
        return self.cost_estimator.estimate(
            n_samples, n_models, model=model,
            has_golden=self.config.golden_data.has_reference() if self.config.golden_data else False,
            enable_explainability=enable_explainability,
        )

    # ============ 反馈闭环 ============

    def feedback_loop(self, disagreements: List[Dict[str, Any]]) -> Optional[AnnotationGuideline]:
        """
        反馈闭环:分歧样本回流到指南优化
        返回更新后的指南(如有)
        """
        if not disagreements or not self.config.guideline:
            return None
        updated = self.guideline_agent.optimize(self.config.guideline, disagreements)
        if updated != self.config.guideline:
            self.config.guideline = updated
            self.feedback_log.append({
                "type": "guideline_optimized",
                "version": updated.version,
                "n_disagreements": len(disagreements),
            })
            return updated
        return None

    # ============ 完整流程 ============

    def run_full_pipeline(self, samples: List[Any],
                          human_labels: Dict[str, Any] = None,
                          review_decisions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        运行完整流程
        标注 → 检验 → 裁决 → 复核 → 反馈闭环
        """
        # 1. 标注
        results = self.annotate_samples(samples)

        # 2. 检验(一致性 + AI 互检)
        inspection = self.inspect_results()

        # 3. 裁决(聚合多 AI 结果)
        final_results = self.adjudicate_results(human_labels)

        # 4. 复核(选择疑难样本)
        review_list = self.review_results(inspection.get("disagreements", []))

        # 5. 应用复核
        if review_decisions:
            final_results = self.apply_review(review_decisions)

        # 6. 反馈闭环(分歧回流到指南)
        updated_guideline = self.feedback_loop(inspection.get("disagreements", []))

        return {
            "results": final_results,
            "inspection": inspection,
            "review_list": review_list,
            "updated_guideline": updated_guideline,
            "feedback_log": self.feedback_log,
        }
