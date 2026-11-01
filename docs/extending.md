# 扩展指南

## 1. 新增行业与研究对象

不需要改任何核心代码，往注册表里加条目即可。

```python
from data_annotation_agent.core.models import DataType, Industry, ResearchObject, TaskType

registry.register_industry(
    Industry(id="energy", name="能源", description="电力、油气与新能源", compliance=[])
)

registry.register_object(
    ResearchObject(
        id="energy_defect_class",
        name="光伏板缺陷分类",
        data_type=DataType.IMAGE,
        task_type=TaskType.IMAGE_CLASSIFICATION,
        description="对光伏组件 EL 图像进行缺陷分类",
        label_schema={"labels": ["正常", "隐裂", "断栅", "黑斑"]},
        industry_id="energy",   # 必须显式关联，否则挂不到行业下
    )
)
```

Web 界面的下拉框会自动出现新条目（`Registry._register_defaults()` 只负责预置项，运行时注册的条目同样生效）。

> 若要**永久**内置，把条目加进 `core/models.py` 的 `_register_defaults()`。

## 2. 接入新的 LLM 服务

只要服务兼容 OpenAI 的 `/chat/completions` 协议，直接选 `custom` 后端填 base_url 即可，**不用写代码**。

若协议不兼容，实现一个 `LLMBackend` 子类：

```python
from data_annotation_agent.llm.backend import LLMBackend, LLMFactory, PROVIDER_PRESETS


class MyVendorBackend(LLMBackend):
    name = "my_vendor"

    def __init__(self, model: str = "my-model", api_key: str = ""):
        self.model = model
        self.api_key = api_key

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        ...  # 返回纯文本

    def complete_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        ...  # 返回已解析的 dict，至少含 label / confidence / rationale

    def test_connection(self) -> dict:
        return {"ok": True, "message": "连接正常"}


LLMFactory._backends["my_vendor"] = MyVendorBackend
PROVIDER_PRESETS["my_vendor"] = {
    "label": "自建服务",
    "base_url": "",
    "model": "my-model",
    "models_default": "my-model",
    "docs": "内部部署模型",
}
```

`complete_json` 必须容错：真实模型经常在 JSON 外面包 ```json 代码块，`OpenAICompatBackend` 已经处理了这种情况，自建后端请照抄该逻辑。

## 3. 新增导出格式

```python
from data_annotation_agent.core.exporter import EXPORTERS, _rows


def export_tsv(results, path: str) -> str:
    rows = _rows(results)
    with open(path, "w", encoding="utf-8") as f:
        if rows:
            f.write("\t".join(rows[0].keys()) + "\n")
            for row in rows:
                f.write("\t".join(str(v) for v in row.values()) + "\n")
    return path


EXPORTERS["tsv"] = export_tsv
```

注册后 `export_results(results, "tsv", path)` 与界面第 ⑦ 步都会自动支持。

## 4. 自定义裁决策略

```python
from data_annotation_agent.agents.adjudicator import AdjudicatorAgent
from data_annotation_agent.core.models import AnnotationResult


class ConfidenceFirstAdjudicator(AdjudicatorAgent):
    """永远采信置信度最高的那条，忽略投票。"""

    def majority_vote(self, results: list[AnnotationResult]) -> AnnotationResult:
        return max(results, key=lambda r: r.confidence)


pipeline.adjudicator = ConfidenceFirstAdjudicator()
```

`AnnotationPipeline` 把各 Agent 存成实例属性，运行时可整体替换，不需要继承 pipeline。

## 5. 支持图像 / 多模态任务

数据模型层（`DataType.IMAGE`、`TaskType.OBJECT_DETECTION` 等）已经就绪，缺的是**输入通道**：

1. **界面层**：`ui/app.py` 第 ③ 步的 `gr.Textbox` 换成 `gr.Gallery` / `gr.File`，把图片路径或 base64 传下去。
2. **标注层**：`AnnotatorAgent._build_system_prompt` 与 `annotate_*` 目前把 `data` 直接 `str()` 拼进 prompt。多模态需要改成 OpenAI 的 content 数组格式：

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "请标注这张图的缺陷类别"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ],
}]
```

3. **导出层**：检测框类结果不适合扁平 CSV，建议自定义字段或直接导出 COCO / Label Studio JSON。

## 6. 持久化标注状态

`AnnotationPipeline` 的状态都在内存里。最小改造：

```python
import json

def snapshot(pipeline) -> dict:
    return {
        "config": pipeline.config.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in pipeline.results],
        "feedback_log": pipeline.feedback_log,
    }
```

反向恢复时用 `AnnotationConfig.model_validate(...)` 与 `AnnotationResult.model_validate(...)` 重建，再赋回 `pipeline.config` / `pipeline.results`。存 SQLite 或 JSON 文件都行。
