"""
LLM 后端抽象层
支持 OpenAI 兼容 API(DeepSeek/通义/OpenAI 等)和模拟模式
- 运行时配置:界面直接填 base_url/api_key/model(也可用环境变量兜底)
- 多后端实例:多 AI 互检时每个模型一个独立后端
- 连接测试:正式标注前先验证 Key 是否可用
"""
import os
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


# ============ 厂商预设(市面常见 OpenAI 兼容服务) ============

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "mock": {
        "label": "模拟模式(无需 API Key)",
        "base_url": "",
        "model": "mock-model",
        "models_default": "mock-1, mock-2, mock-3",
        "docs": "基于规则生成伪标注,用于跑通流程、教学演示与无 Key 体验",
    },
    "deepseek": {
        "label": "DeepSeek(深度求索)",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "models_default": "deepseek-chat, deepseek-reasoner",
        "docs": "性价比高;deepseek-chat 通用对话、deepseek-reasoner 推理更强",
    },
    "qwen": {
        "label": "通义千问(阿里云百炼)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "models_default": "qwen-plus, qwen-max",
        "docs": "qwen-turbo 快、qwen-plus 均衡、qwen-max 最强",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "models_default": "gpt-4o-mini",
        "docs": "生态成熟;gpt-4o-mini 便宜够用",
    },
    "custom": {
        "label": "自定义(任意 OpenAI 兼容 API)",
        "base_url": "",
        "model": "",
        "models_default": "",
        "docs": "填入任意兼容 /chat/completions 的服务地址与模型名",
    },
}


class LLMBackend(ABC):
    """LLM 后端抽象基类"""
    name: str = "base"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """生成文本"""
        pass

    @abstractmethod
    def complete_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """生成 JSON"""
        pass

    def test_connection(self) -> Dict[str, Any]:
        """测试连接(子类可覆盖)"""
        return {"ok": True, "message": f"后端 {self.name} 无需连接测试"}


class OpenAICompatBackend(LLMBackend):
    """
    OpenAI 兼容后端
    支持 DeepSeek、通义千问、OpenAI 等所有兼容 OpenAI API 格式的服务
    配置优先级:显式参数 > 环境变量(LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)
    """
    name = "openai_compat"

    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")).strip()
        self.api_key = (api_key or os.getenv("LLM_API_KEY", "")).strip()
        self.model = (model or os.getenv("LLM_MODEL", "deepseek-chat")).strip()
        if not self.base_url:
            raise ValueError("缺少 API 地址(base_url),请填写或设置环境变量 LLM_BASE_URL")
        if not self.api_key:
            raise ValueError("缺少 API Key,请在界面填写或设置环境变量 LLM_API_KEY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        text = self.complete(system_prompt, user_prompt, **kwargs)
        # 尝试解析 JSON(处理可能的 markdown 包裹)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 部分
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"无法解析 JSON: {text[:200]}")

    def test_connection(self) -> Dict[str, Any]:
        """发起一次极小请求,验证地址/Key/模型是否可用"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "连接测试,请只回复:OK"}],
                max_tokens=8,
                timeout=20,
            )
            text = (resp.choices[0].message.content or "").strip()
            return {"ok": True, "message": f"连接成功({self.model} @ {self.base_url}),模型返回: {text[:30]}"}
        except Exception as e:
            msg = str(e)
            if len(msg) > 220:
                msg = msg[:220] + "..."
            return {"ok": False, "message": f"连接失败: {type(e).__name__}: {msg}"}


class MockBackend(LLMBackend):
    """
    模拟后端(无需 API key)
    用于跑通流程、测试、演示
    基于规则生成"伪标注",便于验证系统逻辑
    """
    name = "mock"

    def __init__(self, model: str = "mock-model"):
        self.model = model

    def test_connection(self) -> Dict[str, Any]:
        return {"ok": True, "message": f"模拟模式({self.model})无需连接,可直接开始标注"}

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # 从 prompt 中提取标签模式,生成一个合理的"伪标签"
        # 实际项目中,这里会调用真实 LLM
        return self._mock_label(system_prompt, user_prompt)

    def complete_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        label = self.complete(system_prompt, user_prompt, **kwargs)
        return {
            "label": label,
            "confidence": 0.8,
            "rationale": f"[模拟] 基于规则判断,该样本被标注为: {label}",
        }

    def _mock_label(self, system_prompt: str, user_prompt: str) -> str:
        """基于简单规则生成伪标签(用于演示)"""
        # 提取标签列表(支持单引号和双引号)
        import re
        labels = re.findall(r'["\']([^"\']+)["\']', system_prompt)
        if not labels:
            # 尝试从"标签类别: [...]"提取
            m = re.search(r"标签类别[::]\s*\[(.*?)\]", system_prompt)
            if m:
                labels = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        if not labels:
            labels = ["类别1", "类别2", "类别3"]

        # 简单规则:根据文本长度和关键词
        text = user_prompt.lower()
        if "疾病" in labels and any(k in text for k in ["发烧", "咳嗽", "头痛", "疼痛", "感染", "肺炎", "高烧"]):
            return "疾病"
        if "症状" in labels and any(k in text for k in ["发热", "乏力", "恶心", "头晕", "高烧", "头痛"]):
            return "症状"
        if "药物" in labels and any(k in text for k in ["药", "片", "胶囊", "注射", "阿司匹林", "抗生素"]):
            return "药物"
        if "正面" in labels and any(k in text for k in ["好", "赞", "满意", "推荐", "喜欢", "有效"]):
            return "正面"
        if "负面" in labels and any(k in text for k in ["差", "烂", "失望", "投诉", "讨厌"]):
            return "负面"
        if "中性" in labels:
            return "中性"
        # 默认返回第一个标签
        return labels[0]


class LLMFactory:
    """LLM 后端工厂"""
    _backends = {
        "openai": OpenAICompatBackend,
        "deepseek": OpenAICompatBackend,
        "qwen": OpenAICompatBackend,
        "custom": OpenAICompatBackend,
        "mock": MockBackend,
    }

    @classmethod
    def create(cls, backend: str, base_url: str = None, api_key: str = None,
               model: str = None, **kwargs) -> LLMBackend:
        """
        创建后端实例
        backend: mock / deepseek / qwen / openai / custom
        base_url / api_key / model: 运行时配置(优先于环境变量)
        """
        backend = (backend or "mock").lower()
        if backend == "mock":
            return MockBackend(model=model or "mock-model")
        if backend in ("openai", "deepseek", "qwen", "custom"):
            preset = PROVIDER_PRESETS.get(backend, {})
            return OpenAICompatBackend(
                base_url=base_url or preset.get("base_url") or None,
                api_key=api_key,
                model=model or preset.get("model") or None,
            )
        raise ValueError(f"未知后端: {backend},可选: {list(cls._backends.keys())}")

    @classmethod
    def list_backends(cls) -> List[str]:
        return list(cls._backends.keys())


# 便捷函数:创建后端
def get_backend(backend: str = None, base_url: str = None, api_key: str = None,
                model: str = None, **kwargs) -> LLMBackend:
    """获取后端,默认 mock(环境变量 LLM_BACKEND 可覆盖)"""
    if backend is None:
        backend = os.getenv("LLM_BACKEND", "mock")
    return LLMFactory.create(backend, base_url=base_url, api_key=api_key, model=model, **kwargs)
