"""
标注指南 Agent:起草 / 优化
用 LLM 帮用户把"想法"变成"清晰规则"
用户主导,系统辅助
"""
from datetime import datetime
from typing import Any, Dict, List

from data_annotation_agent.core.models import AnnotationGuideline, Industry, ResearchObject
from data_annotation_agent.llm.backend import LLMBackend


class GuidelineAgent:
    """标注指南 Agent"""
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def draft(self, industry: Industry, obj: ResearchObject,
              user_description: str = "") -> AnnotationGuideline:
        """
        起草标注指南
        根据行业+对象+用户描述,用 LLM 生成一版指南
        """
        labels = obj.label_schema.get("labels", []) if obj.label_schema else []
        system_prompt = (
            "你是一个专业的标注指南编写专家。请根据给定的行业、任务和标签类别,"
            "起草一份清晰、无歧义、可执行的标注指南。"
            "指南必须包含:标注规则、边界情况处理、示例。"
            "请输出 JSON 格式: {\"title\": \"...\", \"description\": \"...\", "
            "\"rules\": [\"...\"], \"boundary_cases\": [\"...\"], \"examples\": [{\"input\": \"...\", \"output\": \"...\"}]}"
        )
        user_prompt = (
            f"行业: {industry.name}\n"
            f"任务: {obj.name}\n"
            f"任务描述: {obj.description}\n"
            f"标签类别: {labels}\n"
            f"用户补充说明: {user_description or '无'}\n"
            f"请起草标注指南。"
        )

        try:
            result = self.backend.complete_json(system_prompt, user_prompt)
            return AnnotationGuideline(
                title=result.get("title", f"{obj.name}标注指南"),
                description=result.get("description", ""),
                rules=result.get("rules", []),
                boundary_cases=result.get("boundary_cases", []),
                examples=result.get("examples", []),
                label_schema=obj.label_schema or {},
                source="drafted",
            )
        except Exception:
            # 模拟模式或失败时,生成一个基础指南
            return AnnotationGuideline(
                title=f"{obj.name}标注指南",
                description=f"针对{industry.name}行业的{obj.name}任务",
                rules=[f"根据标签类别 {labels} 对数据进行标注"],
                boundary_cases=["遇到边界情况时,优先参考示例"],
                examples=[],
                label_schema=obj.label_schema or {},
                source="drafted",
            )

    def optimize(self, guideline: AnnotationGuideline,
                 disagreements: List[Dict[str, Any]]) -> AnnotationGuideline:
        """
        优化标注指南
        分析分歧样本,建议修改指南
        """
        if not disagreements:
            return guideline

        # 分析分歧原因(简化:基于分歧样本的标签)
        system_prompt = (
            "你是一个标注指南优化专家。根据当前指南和标注分歧,"
            "分析分歧原因,并建议修改指南以减少歧义。"
            "请输出 JSON 格式: {\"new_rules\": [\"...\"], \"new_boundary_cases\": [\"...\"]}"
        )
        user_prompt = (
            f"当前指南:\n{guideline.to_prompt_text()}\n\n"
            f"标注分歧样本:\n{disagreements[:5]}\n\n"
            f"请分析分歧原因并建议修改指南。"
        )

        try:
            result = self.backend.complete_json(system_prompt, user_prompt)
            # 更新指南
            updated = guideline.model_copy()
            updated.rules = guideline.rules + result.get("new_rules", [])
            updated.boundary_cases = guideline.boundary_cases + result.get("new_boundary_cases", [])
            updated.source = "optimized"
            # 版本号递增:用 Decimal 避免浮点误差(1.1 -> 1.2,而非 1.2000000000000002)
            from decimal import Decimal
            updated.version = str(Decimal(guideline.version) + Decimal("0.1"))
            updated.updated_at = datetime.now()
            return updated
        except Exception:
            # 失败时返回原指南
            return guideline
