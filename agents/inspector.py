"""
检验 Agent:一致性检验 + AI 互检
量化标注者间一致性,暴露分歧样本
"""
from typing import List, Dict, Any, Optional
from core.models import AnnotationResult, InspectorPreset
from quality.agreement import compute_agreement, cohens_kappa, fleiss_kappa


class InspectorAgent:
    """检验 Agent"""
    def __init__(self):
        pass

    def check_agreement(self, annotations: List[List[Any]]) -> Dict[str, Any]:
        """计算一致性指标"""
        return compute_agreement(annotations)

    def find_disagreements(self, results: List[AnnotationResult]) -> List[Dict[str, Any]]:
        """
        找出分歧样本
        返回分歧样本列表(标注结果不一致的)
        """
        if not results:
            return []
        # 按 sample_id 分组
        groups: Dict[str, List[AnnotationResult]] = {}
        for r in results:
            groups.setdefault(r.sample_id, []).append(r)

        disagreements = []
        for sample_id, group in groups.items():
            labels = set(r.label for r in group)
            if len(labels) > 1:
                disagreements.append({
                    "sample_id": sample_id,
                    "labels": list(labels),
                    "results": group,
                })
        return disagreements

    def cross_check(self, results: List[AnnotationResult]) -> Dict[str, Any]:
        """
        AI 互检:多 agent 交叉验证
        返回一致性指标 + 分歧样本
        """
        # 按 sample_id 分组,构建标注矩阵
        groups: Dict[str, List[AnnotationResult]] = {}
        for r in results:
            groups.setdefault(r.sample_id, []).append(r)

        # 构建标注矩阵(每个样本的标注者标签)
        annotations = []
        for sample_id, group in groups.items():
            annotations.append([r.label for r in group])

        # 计算一致性
        agreement = compute_agreement(annotations)

        # 找分歧样本
        disagreements = self.find_disagreements(results)

        return {
            "agreement": agreement,
            "disagreements": disagreements,
            "n_disagreements": len(disagreements),
            "n_samples": len(groups),
        }

    def inspect(self, results: List[AnnotationResult], preset: InspectorPreset) -> Dict[str, Any]:
        """根据预设执行检验"""
        if preset == InspectorPreset.LIGHT:
            # 轻量检验:Cohen's κ(两标注者)
            # 按 sample_id 分组,取前两个标注者的标签序列
            groups: Dict[str, List[AnnotationResult]] = {}
            for r in results:
                groups.setdefault(r.sample_id, []).append(r)
            # 取每个样本的前两个标注者标签
            annotations = []
            for sample_id, group in groups.items():
                if len(group) >= 2:
                    annotations.append([group[0].label, group[1].label])
            if len(annotations) >= 1:
                return {"agreement": compute_agreement(annotations), "disagreements": self.find_disagreements(results)}
            return {"agreement": {"method": "无", "value": 0.0, "interpretation": "标注者不足"}, "disagreements": []}
        elif preset == InspectorPreset.STANDARD:
            # 标准检验:Fleiss' κ(多标注者)
            return self.cross_check(results)
        elif preset == InspectorPreset.DEEP:
            # 深度检验:多指标 + AI 互检
            return self.cross_check(results)
        else:
            # 自定义:默认标准检验
            return self.cross_check(results)
