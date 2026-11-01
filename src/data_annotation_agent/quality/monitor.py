"""
过程监控与停止机制
实时监控标注质量,达到阈值时提示停止
提供预设(保守/平衡/宽松/手动)
"""
from typing import List, Dict, Any, Optional
from data_annotation_agent.core.models import MonitorPreset


class MonitorAgent:
    """过程监控 Agent"""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []  # 监控历史

    def _get_stop_condition(self, preset: MonitorPreset) -> Dict[str, Any]:
        """获取预设的停止条件"""
        conditions = {
            MonitorPreset.CONSERVATIVE: {"kappa_threshold": 0.8, "require_stable": True},
            MonitorPreset.BALANCED: {"kappa_threshold": 0.6, "require_stable": True},
            MonitorPreset.LENIENT: {"kappa_threshold": 0.0, "require_stable": False, "max_samples": None},
            MonitorPreset.MANUAL: {"kappa_threshold": 0.0, "require_stable": False, "manual": True},
        }
        return conditions.get(preset, conditions[MonitorPreset.BALANCED])

    def record(self, kappa: float, accuracy: Optional[float] = None,
               n_labeled: int = 0, total: int = 0) -> None:
        """记录一次监控快照"""
        self.history.append({
            "kappa": kappa,
            "accuracy": accuracy,
            "n_labeled": n_labeled,
            "total": total,
        })

    def should_stop(self, preset: MonitorPreset, kappa: float,
                    accuracy: Optional[float] = None,
                    n_labeled: int = 0, total: int = 0) -> Dict[str, Any]:
        """
        判断是否应该停止
        返回 {should_stop, reason}
        """
        cond = self._get_stop_condition(preset)

        # 手动模式:由用户决定
        if cond.get("manual"):
            return {"should_stop": False, "reason": "手动模式,由用户决定"}

        # 宽松模式:标完预设数量
        if preset == MonitorPreset.LENIENT:
            if total > 0 and n_labeled >= total:
                return {"should_stop": True, "reason": f"已标注 {n_labeled}/{total},达到预设数量"}
            return {"should_stop": False, "reason": "继续标注"}

        # 保守/平衡模式:κ 达到阈值
        kappa_threshold = cond.get("kappa_threshold", 0.6)
        if kappa >= kappa_threshold:
            # 检查是否稳定(连续多次达到阈值)
            stable = self._is_stable(kappa_threshold)
            if cond.get("require_stable") and not stable:
                return {"should_stop": False, "reason": f"κ={kappa:.2f} 已达阈值但尚未稳定"}
            return {"should_stop": True, "reason": f"κ={kappa:.2f} 达到阈值 {kappa_threshold}"}

        return {"should_stop": False, "reason": f"κ={kappa:.2f} 未达阈值 {kappa_threshold}"}

    def _is_stable(self, threshold: float, window: int = 3) -> bool:
        """检查最近 window 次是否都达到阈值"""
        recent = [h for h in self.history[-window:] if h["kappa"] >= threshold]
        return len(recent) >= window

    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.history:
            return {"n_records": 0}
        latest = self.history[-1]
        return {
            "n_records": len(self.history),
            "latest_kappa": latest.get("kappa"),
            "latest_accuracy": latest.get("accuracy"),
            "n_labeled": latest.get("n_labeled", 0),
            "total": latest.get("total", 0),
        }


class CostEstimator:
    """成本与时间预估"""
    # 每 token 成本(美元),按模型
    COST_PER_1K_TOKENS = {
        "gpt-4o-mini": 0.00015,
        "gpt-4o": 0.0025,
        "deepseek-chat": 0.00014,
        "qwen-max": 0.002,
        "mock-model": 0.0,
    }
    # 每样本平均 token 数
    AVG_TOKENS_PER_SAMPLE = 500

    def estimate(self, n_samples: int, n_models: int,
                 model: str = "gpt-4o-mini",
                 has_golden: bool = False,
                 enable_explainability: bool = False) -> Dict[str, Any]:
        """
        预估成本和时间
        """
        # 每样本 token 数(考虑可解释性增加输出)
        tokens_per_sample = self.AVG_TOKENS_PER_SAMPLE
        if enable_explainability:
            tokens_per_sample *= 1.5

        # 总 token 数
        total_tokens = n_samples * n_models * tokens_per_sample

        # 成本
        cost_per_1k = self.COST_PER_1K_TOKENS.get(model, 0.0001)
        cost = total_tokens / 1000 * cost_per_1k

        # 时间(假设每样本 2 秒)
        time_seconds = n_samples * n_models * 2
        time_minutes = time_seconds / 60

        return {
            "n_samples": n_samples,
            "n_models": n_models,
            "model": model,
            "total_tokens_est": int(total_tokens),
            "cost_est_usd": round(cost, 4),
            "time_est_minutes": round(time_minutes, 1),
            "has_golden": has_golden,
            "enable_explainability": enable_explainability,
        }
