"""
标注 Agent:单 AI 标注器 + 多 AI 标注器
支持可解释性输出(标注理由)、置信度、难度分级
"""
from typing import List, Dict, Any, Optional
from core.models import (
    AnnotationResult, AnnotationGuideline, GoldSet, AnnotatorPreset,
    DataType, TaskType
)
from llm.backend import LLMBackend


class AnnotatorAgent:
    """
    标注 Agent
    根据预设选择单 AI 或多 AI 标注方式
    backends: 可选,model -> 独立后端实例(多 AI 互检时各真实模型各用各的连接)
    """
    def __init__(self, backend: LLMBackend, models: List[str] = None,
                 backends: Optional[Dict[str, LLMBackend]] = None):
        self.backend = backend
        self.models = models or [backend.model if hasattr(backend, 'model') else "default"]
        self.backends = backends or {}

    def _get_backend_for(self, model: str) -> LLMBackend:
        """获取指定模型的后端(没有独立后端时退回默认)"""
        return self.backends.get(model, self.backend)

    def _build_system_prompt(self, guideline: Optional[AnnotationGuideline],
                             reference_set: Optional[GoldSet],
                             object_info: Dict[str, Any]) -> str:
        """构建系统提示词"""
        parts = ["你是一个专业的数据标注员,请严格按照标注指南进行标注。"]

        # 任务描述
        if object_info:
            parts.append(f"\n任务: {object_info.get('description', '')}")
            labels = object_info.get('label_schema', {}).get('labels', [])
            if labels:
                parts.append(f"\n标签类别: {labels}")

        # 标注指南
        if guideline:
            parts.append(f"\n{guideline.to_prompt_text()}")

        # 参考集示例
        if reference_set and reference_set.samples:
            parts.append("\n## 参考示例(金标)")
            for i, sample in enumerate(reference_set.samples[:5], 1):
                parts.append(f"示例{i}: 输入={sample.data}, 输出={sample.label}")

        parts.append("\n请输出 JSON 格式: {\"label\": \"标签\", \"confidence\": 0.0-1.0, \"rationale\": \"标注理由\"}")
        return "\n".join(parts)

    def annotate_single(self, data: Any, guideline: Optional[AnnotationGuideline],
                        reference_set: Optional[GoldSet],
                        object_info: Dict[str, Any]) -> AnnotationResult:
        """单 AI 标注"""
        system_prompt = self._build_system_prompt(guideline, reference_set, object_info)
        user_prompt = f"请标注以下数据:\n{data}"

        # 单 AI 也用与标注者 ID 对应的后端(与 annotate_multi 保持一致)
        model = self.models[0] if self.models else "ai"
        backend = self._get_backend_for(model)
        result = backend.complete_json(system_prompt, user_prompt)
        confidence = float(result.get("confidence", 0.5))
        return AnnotationResult(
            sample_id=str(data) if not hasattr(data, 'id') else data.id,
            label=result.get("label", ""),
            confidence=confidence,
            rationale=result.get("rationale", ""),
            annotator_id=model,
            difficulty=self._classify_difficulty(confidence),
        )

    @staticmethod
    def _classify_difficulty(confidence: float) -> str:
        """根据置信度分级:高置信度=simple,中=medium,低=difficult"""
        if confidence >= 0.8:
            return "simple"
        elif confidence >= 0.5:
            return "medium"
        return "difficult"

    def annotate_multi(self, data: Any, guideline: Optional[AnnotationGuideline],
                       reference_set: Optional[GoldSet],
                       object_info: Dict[str, Any],
                       models: List[str] = None) -> List[AnnotationResult]:
        """
        多 AI 标注:多个模型独立标注
        返回多个标注结果,供后续聚合/互检
        模拟模式下,不同模型引入轻微差异以展示互检价值
        """
        models = models or self.models
        results = []
        for i, model in enumerate(models):
            # 每个模型用各自的后端独立标注(多 AI 互检时各真实模型各用各的连接)
            backend_i = self._get_backend_for(model)
            system_prompt = self._build_system_prompt(guideline, reference_set, object_info)
            user_prompt = f"请标注以下数据:\n{data}"
            result = backend_i.complete_json(system_prompt, user_prompt)

            # 模拟模式:为不同模型引入轻微差异(模拟不同模型的判断差异)
            label = result.get("label", "")
            if backend_i.name == "mock" and len(models) > 1:
                # 第2个及以后的模型,有一定概率给出不同标签(模拟分歧)
                import random
                labels = self._extract_labels(system_prompt)
                if labels and len(labels) > 1 and i > 0 and random.random() < 0.3:
                    # 30% 概率选一个不同的标签
                    others = [l for l in labels if l != label]
                    if others:
                        label = random.choice(others)

            confidence = float(result.get("confidence", 0.5))
            results.append(AnnotationResult(
                sample_id=str(data) if not hasattr(data, 'id') else data.id,
                label=label,
                confidence=confidence,
                rationale=result.get("rationale", ""),
                annotator_id=model,
                difficulty=self._classify_difficulty(confidence),
            ))
        return results

    def _extract_labels(self, system_prompt: str) -> List[str]:
        """从系统提示词提取标签列表"""
        import re
        labels = re.findall(r"['\"]([^'\"]+)['\"]", system_prompt)
        # 过滤掉非标签的文本(如"请标注以下数据"、"标签"、"标注理由"等)
        stopwords = {
            "请", "标注", "以下", "数据", "输出", "JSON", "格式",
            "label", "confidence", "rationale",
            "标签", "标注理由", "理由", "类别", "实体", "文本", "内容",
            "系统", "提示", "词", "参考", "示例", "样本", "结果", "列表",
        }
        return [l for l in labels if l not in stopwords and len(l) <= 10]

    def annotate(self, data: Any, preset: AnnotatorPreset,
                 guideline: Optional[AnnotationGuideline],
                 reference_set: Optional[GoldSet],
                 object_info: Dict[str, Any]) -> List[AnnotationResult]:
        """
        根据预设执行标注
        返回标注结果列表(单 AI 返回 1 个,多 AI 返回多个)
        """
        if preset == AnnotatorPreset.FAST:
            # 快速模式:单 AI,零样本(不用参考集)
            return [self.annotate_single(data, guideline, None, object_info)]
        elif preset == AnnotatorPreset.STANDARD:
            # 标准模式:单 AI + 参考集引导
            return [self.annotate_single(data, guideline, reference_set, object_info)]
        elif preset == AnnotatorPreset.RIGOROUS:
            # 严谨模式:多 AI 互检 + 聚合
            return self.annotate_multi(data, guideline, reference_set, object_info)
        else:
            # 自定义:默认单 AI
            return [self.annotate_single(data, guideline, reference_set, object_info)]
