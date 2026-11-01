"""
裁决 Agent:多数投票 / 加权聚合 / 人工裁决
用于聚合多个标注者的结果,解决分歧
"""
from collections import Counter
from typing import Any, Dict, List, Optional

from data_annotation_agent.core.models import AdjudicatorPreset, AnnotationResult


class AdjudicatorAgent:
    """裁决 Agent"""
    def __init__(self):
        pass

    def majority_vote(self, results: List[AnnotationResult]) -> AnnotationResult:
        """多数投票:取出现最多的标签"""
        if not results:
            raise ValueError("无标注结果")
        labels = [r.label for r in results]
        counter = Counter(labels)
        # 取出现最多的标签(平局时取第一个)
        most_common = counter.most_common(1)[0][0]
        # 找到对应结果
        for r in results:
            if r.label == most_common:
                return r
        return results[0]

    def weighted_aggregate(self, results: List[AnnotationResult],
                           weights: Optional[Dict[str, float]] = None) -> AnnotationResult:
        """
        加权聚合:按可靠度加权
        weights: 标注者ID -> 权重
        """
        if not results:
            raise ValueError("无标注结果")
        if weights is None:
            # 默认按置信度加权
            weights = {r.annotator_id: r.confidence for r in results}

        # 按标签聚合加权分数
        label_scores: Dict[Any, float] = {}
        for r in results:
            w = weights.get(r.annotator_id, 1.0)
            label_scores[r.label] = label_scores.get(r.label, 0.0) + w

        # 取分数最高的标签
        best_label = max(label_scores, key=label_scores.get)
        for r in results:
            if r.label == best_label:
                return r
        return results[0]

    def human_adjudicate(self, results: List[AnnotationResult],
                         human_label: Any) -> AnnotationResult:
        """人工裁决:以人工标签为准"""
        if not results:
            raise ValueError("无标注结果")
        return AnnotationResult(
            sample_id=results[0].sample_id,
            label=human_label,
            confidence=1.0,
            rationale="人工裁决",
            annotator_id="human",
        )

    def adjudicate(self, results: List[AnnotationResult],
                   preset: AdjudicatorPreset,
                   human_label: Optional[Any] = None,
                   weights: Optional[Dict[str, float]] = None) -> AnnotationResult:
        """根据预设执行裁决"""
        if preset == AdjudicatorPreset.MAJORITY:
            return self.majority_vote(results)
        elif preset == AdjudicatorPreset.WEIGHTED:
            return self.weighted_aggregate(results, weights)
        elif preset == AdjudicatorPreset.HUMAN:
            if human_label is None:
                raise ValueError("人工裁决需要提供 human_label")
            return self.human_adjudicate(results, human_label)
        else:
            # 自定义:默认多数投票
            return self.majority_vote(results)
