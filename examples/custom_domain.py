"""扩展示例：注册一个新行业与研究对象，全链路直接复用。

不修改核心代码，只在运行时往注册表里加条目。

运行：
    pip install -e .
    python examples/custom_domain.py
"""

from data_annotation_agent import AnnotationPipeline, get_backend
from data_annotation_agent.core.models import (
    DataType,
    Industry,
    ResearchObject,
    TaskType,
)


def main() -> None:
    pipeline = AnnotationPipeline(backend=get_backend("mock"))
    registry = pipeline.registry

    # 1. 注册新行业
    registry.register_industry(
        Industry(id="energy", name="能源", description="电力、油气与新能源")
    )

    # 2. 注册该行业下的研究对象
    registry.register_object(
        ResearchObject(
            id="energy_defect_class",
            name="光伏板缺陷分类",
            data_type=DataType.IMAGE,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            description="对光伏组件 EL 图像进行缺陷分类",
            label_schema={"labels": ["正常", "隐裂", "断栅", "黑斑"]},
            industry_id="energy",
        )
    )

    # 3. 用新对象走标准流程
    pipeline.create_config(
        "energy",
        "energy_defect_class",
        llm_backend="mock",
        llm_models=["mock-1"],
    )
    results = pipeline.annotate_samples(["组件A 出现细长裂纹", "组件B 表面有黑色斑点"])

    print(f"行业: {registry.get_industry('energy').name}")
    print(f"对象: {registry.get_object('energy_defect_class').name}")
    print(f"标签体系: {registry.get_object('energy_defect_class').label_schema['labels']}\n")

    for result in results:
        print(f"  {result.sample_id} → {result.label}")


if __name__ == "__main__":
    main()
