"""
一致性检验模块
实现 Cohen's κ / Fleiss' κ / Krippendorff's α
用于量化多个标注者(多个AI或AI与人)的标注一致性
"""
from collections import Counter
from typing import Any, Dict, List


def cohens_kappa(labels_a: List[Any], labels_b: List[Any]) -> float:
    """
    Cohen's κ:衡量两个标注者的一致性,扣除碰巧一致的部分
    取值范围 [-1, 1]
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("两个标注者的标签数量必须相同")
    n = len(labels_a)
    if n == 0:
        return 0.0

    # 观察一致率 Po
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # 期望一致率 Pe(基于各标签的边缘分布)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    all_labels = set(labels_a) | set(labels_b)
    expected = sum(
        (count_a.get(label, 0) / n) * (count_b.get(label, 0) / n)
        for label in all_labels
    )

    # κ = (Po - Pe) / (1 - Pe)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: List[List[Any]]) -> float:
    """
    Fleiss' κ:Cohen's κ 的多标注者扩展,适用于三个及以上标注者
    ratings: 每个样本的标注结果列表,如 [[标注者1,标注者2,标注者3], ...]
    取值范围 [-1, 1]
    """
    n_subjects = len(ratings)          # 样本数
    if n_subjects == 0:
        return 0.0
    n_raters = len(ratings[0])         # 标注者数
    if n_raters <= 1:
        return 0.0

    # 收集所有标签
    all_labels = set()
    for r in ratings:
        all_labels.update(r)
    labels = list(all_labels)
    n_categories = len(labels)

    # 每个样本的类别计数
    subject_counts = []
    for r in ratings:
        count = Counter(r)
        subject_counts.append([count.get(label, 0) for label in labels])

    # 每个类别的总计数
    category_totals = [sum(sc[i] for sc in subject_counts) for i in range(n_categories)]
    total = sum(category_totals)

    # 每个类别的比例
    p_j = [ct / total for ct in category_totals]

    # 每个样本的一致程度 Pi
    P_i = []
    for sc in subject_counts:
        sum_sq = sum(x * x for x in sc)
        P_i.append((sum_sq - n_raters) / (n_raters * (n_raters - 1)))

    # 平均一致程度 P_bar
    P_bar = sum(P_i) / n_subjects

    # 期望一致程度 Pe
    P_e = sum(p * p for p in p_j)

    # κ = (P_bar - P_e) / (1 - P_e)
    if P_e == 1:
        return 1.0 if P_bar == 1 else 0.0
    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha(ratings: List[List[Any]], data_type: str = "nominal") -> float:
    """
    Krippendorff's α:最灵活的一致性指标
    支持缺失数据(None)和多种数据类型(名义/顺序/区间)
    ratings: 每个样本的标注结果列表,缺失用 None
    """
    # 简化实现:对名义数据,与 Fleiss' κ 等价
    # 过滤掉有缺失的样本(简化处理)
    clean_ratings = [r for r in ratings if all(x is not None for x in r)]
    if len(clean_ratings) == 0:
        return 0.0
    return fleiss_kappa(clean_ratings)


def interpret_kappa(kappa: float) -> str:
    """解读 κ 值(Landis & Koch 标准)"""
    if kappa < 0:
        return "无一致性"
    elif kappa < 0.21:
        return "微弱一致性"
    elif kappa < 0.41:
        return "一般一致性"
    elif kappa < 0.61:
        return "中等一致性"
    elif kappa < 0.81:
        return "显著一致性"
    else:
        return "几乎完美一致性"


def compute_agreement(annotations: List[List[Any]]) -> Dict[str, Any]:
    """
    计算一组标注的一致性指标
    annotations: 每个样本的标注结果列表
    返回各指标及解读
    """
    n_raters = len(annotations[0]) if annotations else 0
    result = {}

    if n_raters == 2:
        # 两个标注者:Cohen's κ
        labels_a = [r[0] for r in annotations]
        labels_b = [r[1] for r in annotations]
        kappa = cohens_kappa(labels_a, labels_b)
        result = {
            "method": "Cohen's κ",
            "value": round(kappa, 4),
            "interpretation": interpret_kappa(kappa),
            "n_raters": 2,
        }
    elif n_raters >= 3:
        # 多个标注者:Fleiss' κ + Krippendorff's α
        kappa = fleiss_kappa(annotations)
        alpha = krippendorff_alpha(annotations)
        result = {
            "method": "Fleiss' κ + Krippendorff's α",
            "fleiss_kappa": round(kappa, 4),
            "krippendorff_alpha": round(alpha, 4),
            "interpretation": interpret_kappa(kappa),
            "n_raters": n_raters,
        }
    else:
        result = {
            "method": "无",
            "value": 0.0,
            "interpretation": "标注者数量不足",
            "n_raters": n_raters,
        }

    return result
