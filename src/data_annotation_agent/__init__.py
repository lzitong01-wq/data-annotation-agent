"""数据标注智能体 —— 通用数据标注框架。

多 Agent 协作完成一条完整的标注工程链路：

    选行业与对象 → 配 LLM → 标注 → 一致性检验 → 裁决 → 复核 → 反馈闭环 → 导出

典型用法::

    from data_annotation_agent import AnnotationPipeline, get_backend

    pipeline = AnnotationPipeline(backend=get_backend("mock"))
    pipeline.create_config("medical", "medical_ner")
    out = pipeline.run_full_pipeline(["患者持续高烧三天"])
    print(out["inspection"]["agreement"])

Web 界面::

    python -m data_annotation_agent
"""

from data_annotation_agent.core.models import (
    AgentPresets,
    AnnotationConfig,
    AnnotationGuideline,
    AnnotationResult,
    DataType,
    GoldSample,
    GoldSet,
    GoldenData,
    Industry,
    Registry,
    ResearchObject,
    TaskType,
)
from data_annotation_agent.core.exporter import EXPORTERS, export_results
from data_annotation_agent.core.pipeline import AnnotationPipeline
from data_annotation_agent.llm.backend import LLMBackend, get_backend

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 流程
    "AnnotationPipeline",
    # 注册表与任务定义
    "Registry",
    "Industry",
    "ResearchObject",
    "DataType",
    "TaskType",
    # 标注数据与配置
    "AnnotationGuideline",
    "AnnotationResult",
    "AnnotationConfig",
    "AgentPresets",
    "GoldenData",
    "GoldSet",
    "GoldSample",
    # LLM 后端
    "LLMBackend",
    "get_backend",
    # 导出
    "export_results",
    "EXPORTERS",
]
