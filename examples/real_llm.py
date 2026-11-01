"""接入真实 LLM：DeepSeek / 通义千问 / OpenAI / 任意 OpenAI 兼容服务。

先设置环境变量（也可在 Web 界面第 ② 步直接填写）：

    export LLM_BACKEND=deepseek
    export LLM_BASE_URL=https://api.deepseek.com
    export LLM_API_KEY=sk-xxxxxxxx
    export LLM_MODEL=deepseek-chat

运行：
    pip install -e .
    python examples/real_llm.py
"""

import os
import sys

from data_annotation_agent import AnnotationPipeline, get_backend
from data_annotation_agent.core.models import AgentPresets, AnnotatorPreset

SAMPLES = [
    "患者出现持续高烧和剧烈头痛",
    "医生开具阿司匹林进行治疗",
    "确诊为病毒性肺炎",
]


def main() -> int:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        print("未检测到 LLM_API_KEY。")
        print("请先设置：export LLM_API_KEY=sk-xxxxxxxx")
        print("可选：LLM_BACKEND(deepseek/qwen/openai/custom) LLM_BASE_URL LLM_MODEL")
        print("不想配 Key 也可以跑模拟示例：python examples/quickstart.py")
        return 1

    backend_name = os.getenv("LLM_BACKEND", "deepseek").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    model = os.getenv("LLM_MODEL", "").strip() or None

    # 正式标注前先验证 Key / 地址 / 模型是否可用
    backend = get_backend(backend_name, base_url=base_url, api_key=api_key, model=model)
    check = backend.test_connection()
    print(f"连接测试：{check['message']}")
    if not check["ok"]:
        return 1

    pipeline = AnnotationPipeline(backend=backend)
    pipeline.create_config(
        "medical",
        "medical_ner",
        presets=AgentPresets(annotator=AnnotatorPreset.STANDARD),
        llm_backend=backend_name,
        llm_models=[backend.model],
        base_url=base_url,
        api_key=api_key,
    )

    results = pipeline.annotate_samples(SAMPLES)
    print(f"\n标注完成：{len(results)} 条\n")
    for result in results:
        print(f"  {result.sample_id}")
        print(f"    标签={result.label}  置信度={result.confidence:.2f}  难度={result.difficulty}")
        print(f"    理由={result.rationale}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
