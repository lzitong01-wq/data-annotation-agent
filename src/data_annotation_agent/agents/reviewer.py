"""
复核 Agent:人工复核接口
自动标记疑难样本,提供人工复核
"""
from typing import List, Dict, Any, Optional
from data_annotation_agent.core.models import AnnotationResult, ReviewerPreset


class ReviewerAgent:
    """复核 Agent"""
    def __init__(self):
        pass

    def mark_difficult(self, results: List[AnnotationResult],
                       disagreements: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        标记疑难样本
        基于置信度低、分歧大等特征
        """
        difficult = []
        # 分歧样本
        if disagreements:
            for d in disagreements:
                difficult.append({
                    "sample_id": d["sample_id"],
                    "reason": "多标注者分歧",
                    "labels": d.get("labels", []),
                })

        # 低置信度样本
        for r in results:
            if r.confidence < 0.5:
                difficult.append({
                    "sample_id": r.sample_id,
                    "reason": "低置信度",
                    "confidence": r.confidence,
                })

        # 去重
        seen = set()
        unique = []
        for d in difficult:
            if d["sample_id"] not in seen:
                seen.add(d["sample_id"])
                unique.append(d)
        return unique

    def select_for_review(self, results: List[AnnotationResult],
                          preset: ReviewerPreset,
                          disagreements: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """根据预设选择需要复核的样本"""
        if preset == ReviewerPreset.SAMPLE:
            # 抽检模式:随机抽 5%
            import random
            n = max(1, int(len(results) * 0.05))
            sampled = random.sample(results, min(n, len(results)))
            return [{"sample_id": r.sample_id, "reason": "随机抽检", "label": r.label} for r in sampled]
        elif preset == ReviewerPreset.DIFFICULT:
            # 疑难模式:只复核疑难样本
            return self.mark_difficult(results, disagreements)
        elif preset == ReviewerPreset.FULL:
            # 全量模式:全部复核
            return [{"sample_id": r.sample_id, "reason": "全量复核", "label": r.label} for r in results]
        else:
            # 自定义:默认疑难模式
            return self.mark_difficult(results, disagreements)

    def apply_review(self, results: List[AnnotationResult],
                     review_decisions: Dict[str, Any]) -> List[AnnotationResult]:
        """
        应用人工复核结果
        review_decisions: {sample_id: 人工标签}
        返回更新后的标注结果
        """
        updated = []
        for r in results:
            if r.sample_id in review_decisions:
                # 用人工标签覆盖
                updated.append(AnnotationResult(
                    sample_id=r.sample_id,
                    label=review_decisions[r.sample_id],
                    confidence=1.0,
                    rationale="人工复核修正",
                    annotator_id="human",
                ))
            else:
                updated.append(r)
        return updated
