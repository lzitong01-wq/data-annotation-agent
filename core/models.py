"""
核心数据模型:行业/研究对象注册、金标三集、标注指南、Agent 预设
通用框架设计:行业/对象可扩展注册,不限于特定领域
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ============ 基础枚举 ============

class DataType(str, Enum):
    """数据类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class TaskType(str, Enum):
    """标注任务类型"""
    # 文本
    TEXT_CLASSIFICATION = "text_classification"      # 文本分类
    NER = "ner"                                      # 实体识别
    RELATION_EXTRACTION = "relation_extraction"      # 关系抽取
    SENTIMENT = "sentiment"                          # 情感分析
    INTENT = "intent"                                # 意图识别
    # 图像
    IMAGE_CLASSIFICATION = "image_classification"    # 图像分类
    OBJECT_DETECTION = "object_detection"            # 目标检测
    IMAGE_SEGMENTATION = "image_segmentation"        # 图像分割
    # 多模态
    IMAGE_TEXT_MATCH = "image_text_match"            # 图文匹配
    CUSTOM = "custom"                                # 自定义


class AgentRole(str, Enum):
    """Agent 角色"""
    ANNOTATOR = "annotator"      # 标注者
    INSPECTOR = "inspector"      # 检验者
    ADJUDICATOR = "adjudicator"  # 裁决者
    REVIEWER = "reviewer"        # 复核者


# ============ 行业/研究对象注册机制(通用框架) ============

class Industry(BaseModel):
    """行业"""
    id: str
    name: str
    description: str = ""
    compliance: List[str] = Field(default_factory=list)  # 合规要求,如 HIPAA, GDPR


class ResearchObject(BaseModel):
    """研究对象"""
    id: str
    name: str
    data_type: DataType
    task_type: TaskType
    description: str = ""
    label_schema: Optional[Dict[str, Any]] = None  # 标签模式
    industry_id: str = ""  # 所属行业(显式关联;空则按 id 前缀推断)


class Registry:
    """
    行业/研究对象注册中心(通用框架)
    支持动态注册,不限于特定领域
    """
    def __init__(self):
        self._industries: Dict[str, Industry] = {}
        self._objects: Dict[str, ResearchObject] = {}
        self._register_defaults()

    def register_industry(self, industry: Industry):
        self._industries[industry.id] = industry

    def register_object(self, obj: ResearchObject):
        self._objects[obj.id] = obj

    def get_industry(self, industry_id: str) -> Optional[Industry]:
        return self._industries.get(industry_id)

    def get_object(self, object_id: str) -> Optional[ResearchObject]:
        return self._objects.get(object_id)

    def list_industries(self) -> List[Industry]:
        return list(self._industries.values())

    def list_objects(self) -> List[ResearchObject]:
        return list(self._objects.values())

    def list_objects_by_industry(self, industry_id: str) -> List[ResearchObject]:
        # 按显式 industry_id 关联过滤(优先);兼容旧对象(无 industry_id 时按 id 前缀推断)
        matched = [o for o in self._objects.values() if o.industry_id == industry_id]
        if not matched:
            prefix = industry_id + "_"
            matched = [o for o in self._objects.values() if o.id.startswith(prefix)]
        return matched

    def _register_defaults(self):
        """注册默认行业和研究对象(通用框架,可扩展)"""
        # 默认行业
        default_industries = [
            Industry(id="medical", name="医疗健康", description="医疗影像、电子病历、医学文献", compliance=["HIPAA"]),
            Industry(id="finance", name="金融", description="金融文本、风控、合规", compliance=["GDPR", "PCI-DSS"]),
            Industry(id="legal", name="法律", description="法律文件、合同、判例", compliance=["GDPR"]),
            Industry(id="autonomous", name="自动驾驶", description="多传感器融合、3D点云", compliance=["ISO 26262"]),
            Industry(id="ecommerce", name="电商零售", description="商品评论、用户行为", compliance=[]),
            Industry(id="education", name="教育", description="教育内容、学习行为", compliance=["FERPA"]),
            Industry(id="industrial", name="工业制造", description="工业质检、设备监控", compliance=[]),
            Industry(id="general", name="通用/其他", description="通用领域", compliance=[]),
        ]
        for ind in default_industries:
            self.register_industry(ind)

        # 默认研究对象(覆盖多种数据类型和任务;industry_id 显式关联所属行业)
        default_objects = [
            # ---- 医疗健康 ----
            ResearchObject(id="medical_ner", name="医疗实体识别", data_type=DataType.TEXT, task_type=TaskType.NER,
                          description="从医疗文本中识别疾病、症状、药物、检查等实体",
                          label_schema={"labels": ["疾病", "症状", "药物", "检查", "治疗"]},
                          industry_id="medical"),
            ResearchObject(id="medical_symptom_class", name="症状严重度分级", data_type=DataType.TEXT, task_type=TaskType.TEXT_CLASSIFICATION,
                          description="对患者主诉进行症状严重度分级",
                          label_schema={"labels": ["轻度", "中度", "重度"]},
                          industry_id="medical"),
            # ---- 金融 ----
            ResearchObject(id="finance_intent", name="金融意图识别", data_type=DataType.TEXT, task_type=TaskType.INTENT,
                          description="识别金融文本的用户意图",
                          label_schema={"labels": ["查询", "投诉", "咨询", "办理"]},
                          industry_id="finance"),
            ResearchObject(id="finance_risk_class", name="金融风险文本分类", data_type=DataType.TEXT, task_type=TaskType.TEXT_CLASSIFICATION,
                          description="对金融文本进行风险等级分类",
                          label_schema={"labels": ["低风险", "中风险", "高风险"]},
                          industry_id="finance"),
            # ---- 法律 ----
            ResearchObject(id="legal_contract_clause", name="合同条款分类", data_type=DataType.TEXT, task_type=TaskType.TEXT_CLASSIFICATION,
                          description="对合同条款进行类型分类",
                          label_schema={"labels": ["付款", "违约", "保密", "期限", "其他"]},
                          industry_id="legal"),
            ResearchObject(id="legal_ner", name="法律实体识别", data_type=DataType.TEXT, task_type=TaskType.NER,
                          description="从法律文本中识别当事人、金额、日期等实体",
                          label_schema={"labels": ["当事人", "金额", "日期", "机构", "其他"]},
                          industry_id="legal"),
            # ---- 电商零售 ----
            ResearchObject(id="ecommerce_sentiment", name="电商情感分类", data_type=DataType.TEXT, task_type=TaskType.SENTIMENT,
                          description="对电商评论进行情感极性分类",
                          label_schema={"labels": ["正面", "负面", "中性"]},
                          industry_id="ecommerce"),
            ResearchObject(id="ecommerce_aspect", name="商品属性抽取", data_type=DataType.TEXT, task_type=TaskType.NER,
                          description="从商品描述中抽取属性实体",
                          label_schema={"labels": ["品牌", "型号", "价格", "材质", "其他"]},
                          industry_id="ecommerce"),
            # ---- 教育 ----
            ResearchObject(id="education_question_class", name="试题知识点分类", data_type=DataType.TEXT, task_type=TaskType.TEXT_CLASSIFICATION,
                          description="对试题按知识点进行分类",
                          label_schema={"labels": ["数学", "语文", "英语", "物理", "化学"]},
                          industry_id="education"),
            ResearchObject(id="education_feedback_sentiment", name="教学反馈情感分析", data_type=DataType.TEXT, task_type=TaskType.SENTIMENT,
                          description="对学生教学反馈进行情感分析",
                          label_schema={"labels": ["满意", "一般", "不满意"]},
                          industry_id="education"),
            # ---- 工业制造 ----
            ResearchObject(id="industrial_defect_class", name="工业缺陷分类", data_type=DataType.IMAGE, task_type=TaskType.IMAGE_CLASSIFICATION,
                          description="对工业质检图像进行缺陷分类",
                          label_schema={"labels": ["正常", "划痕", "裂纹", "污渍", "变形"]},
                          industry_id="industrial"),
            ResearchObject(id="industrial_defect_detect", name="工业缺陷检测", data_type=DataType.IMAGE, task_type=TaskType.OBJECT_DETECTION,
                          description="在工业质检图像中定位缺陷目标",
                          label_schema={"labels": ["缺陷"]},
                          industry_id="industrial"),
            # ---- 自动驾驶 ----
            ResearchObject(id="autonomous_scene_class", name="驾驶场景分类", data_type=DataType.IMAGE, task_type=TaskType.IMAGE_CLASSIFICATION,
                          description="对驾驶场景图像进行分类",
                          label_schema={"labels": ["城市道路", "高速", "乡村", "夜间", "恶劣天气"]},
                          industry_id="autonomous"),
            ResearchObject(id="autonomous_object_detect", name="交通目标检测", data_type=DataType.IMAGE, task_type=TaskType.OBJECT_DETECTION,
                          description="检测车辆、行人、交通标志等目标",
                          label_schema={"labels": ["车辆", "行人", "自行车", "交通标志", "其他"]},
                          industry_id="autonomous"),
            # ---- 通用/其他 ----
            ResearchObject(id="text_classification", name="通用文本分类", data_type=DataType.TEXT, task_type=TaskType.TEXT_CLASSIFICATION,
                          description="通用文本分类任务",
                          label_schema={"labels": ["类别1", "类别2", "类别3"]},
                          industry_id="general"),
            ResearchObject(id="ner", name="通用实体识别", data_type=DataType.TEXT, task_type=TaskType.NER,
                          description="通用命名实体识别",
                          label_schema={"labels": ["人名", "地名", "机构名", "其他"]},
                          industry_id="general"),
            ResearchObject(id="image_classification", name="通用图像分类", data_type=DataType.IMAGE, task_type=TaskType.IMAGE_CLASSIFICATION,
                          description="通用图像分类",
                          label_schema={"labels": ["类别1", "类别2"]},
                          industry_id="general"),
            ResearchObject(id="object_detection", name="通用目标检测", data_type=DataType.IMAGE, task_type=TaskType.OBJECT_DETECTION,
                          description="通用目标检测",
                          label_schema={"labels": ["目标1", "目标2"]},
                          industry_id="general"),
            ResearchObject(id="image_text_match", name="图文匹配", data_type=DataType.MULTIMODAL, task_type=TaskType.IMAGE_TEXT_MATCH,
                          description="图文匹配标注",
                          label_schema={"labels": ["匹配", "不匹配"]},
                          industry_id="general"),
        ]
        for obj in default_objects:
            self.register_object(obj)


# ============ 金标三集 ============

class GoldSample(BaseModel):
    """金标样本"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: Any                                # 原始数据(文本/图像路径等)
    label: Any                               # 金标标签
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class GoldSet(BaseModel):
    """金标数据集"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    samples: List[GoldSample] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class GoldenData(BaseModel):
    """
    金标三集(可选)
    参考集:示例引导;优化集:调优校准;测试集:独立评估
    """
    reference_set: Optional[GoldSet] = None      # 参考集(可选)
    optimization_set: Optional[GoldSet] = None   # 优化集(可选)
    test_set: Optional[GoldSet] = None           # 测试集(可选)

    def has_reference(self) -> bool:
        return self.reference_set is not None and len(self.reference_set.samples) > 0

    def has_optimization(self) -> bool:
        return self.optimization_set is not None and len(self.optimization_set.samples) > 0

    def has_test(self) -> bool:
        return self.test_set is not None and len(self.test_set.samples) > 0

    def summary(self) -> str:
        parts = []
        if self.has_reference():
            parts.append(f"参考集{len(self.reference_set.samples)}条")
        if self.has_optimization():
            parts.append(f"优化集{len(self.optimization_set.samples)}条")
        if self.has_test():
            parts.append(f"测试集{len(self.test_set.samples)}条")
        return ", ".join(parts) if parts else "无金标"


# ============ 标注指南 ============

class AnnotationGuideline(BaseModel):
    """标注指南"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0"
    title: str
    description: str = ""
    rules: List[str] = Field(default_factory=list)        # 标注规则
    boundary_cases: List[str] = Field(default_factory=list)  # 边界情况
    examples: List[Dict[str, Any]] = Field(default_factory=list)  # 示例
    label_schema: Dict[str, Any] = Field(default_factory=dict)    # 标签模式
    source: str = "user"  # user(用户上传)/drafted(系统起草)/optimized(系统优化)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_prompt_text(self) -> str:
        """转成 prompt 文本"""
        lines = [f"# {self.title}", ""]
        if self.description:
            lines.append(f"## 描述\n{self.description}\n")
        if self.rules:
            lines.append("## 标注规则")
            for i, rule in enumerate(self.rules, 1):
                lines.append(f"{i}. {rule}")
            lines.append("")
        if self.boundary_cases:
            lines.append("## 边界情况处理")
            for i, case in enumerate(self.boundary_cases, 1):
                lines.append(f"{i}. {case}")
            lines.append("")
        if self.examples:
            lines.append("## 示例")
            for ex in self.examples:
                lines.append(f"- 输入: {ex.get('input', '')}")
                lines.append(f"  输出: {ex.get('output', '')}")
            lines.append("")
        return "\n".join(lines)


# ============ Agent 预设 ============

class AnnotatorPreset(str, Enum):
    """标注 Agent 预设"""
    FAST = "fast"          # 快速模式:单AI零样本
    STANDARD = "standard"  # 标准模式:单AI+参考集引导
    RIGOROUS = "rigorous"  # 严谨模式:多AI互检+聚合
    CUSTOM = "custom"      # 自定义


class InspectorPreset(str, Enum):
    """检验 Agent 预设"""
    LIGHT = "light"        # 轻量检验:Cohen's κ
    STANDARD = "standard"  # 标准检验:Fleiss' κ
    DEEP = "deep"          # 深度检验:多指标+AI互检
    CUSTOM = "custom"      # 自定义


class AdjudicatorPreset(str, Enum):
    """裁决 Agent 预设"""
    MAJORITY = "majority"  # 多数投票
    WEIGHTED = "weighted"  # 加权聚合
    HUMAN = "human"        # 人工裁决
    CUSTOM = "custom"      # 自定义


class ReviewerPreset(str, Enum):
    """复核 Agent 预设"""
    SAMPLE = "sample"      # 抽检模式:随机抽5%
    DIFFICULT = "difficult"  # 疑难模式:只复核疑难样本
    FULL = "full"          # 全量模式:全部复核
    CUSTOM = "custom"      # 自定义


class MonitorPreset(str, Enum):
    """监控预设"""
    CONSERVATIVE = "conservative"  # 保守模式:κ>0.8且准确率稳定
    BALANCED = "balanced"          # 平衡模式:κ>0.6且准确率稳定
    LENIENT = "lenient"            # 宽松模式:标完预设数量
    MANUAL = "manual"              # 手动模式:用户自己决定


class AgentPresets(BaseModel):
    """Agent 预设组合"""
    annotator: AnnotatorPreset = AnnotatorPreset.STANDARD
    inspector: InspectorPreset = InspectorPreset.STANDARD
    adjudicator: AdjudicatorPreset = AdjudicatorPreset.MAJORITY
    reviewer: ReviewerPreset = ReviewerPreset.DIFFICULT
    monitor: MonitorPreset = MonitorPreset.BALANCED


# ============ 标注结果 ============

class AnnotationResult(BaseModel):
    """标注结果"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_id: str
    label: Any                                # 标注标签
    confidence: float = 0.0                   # 置信度
    rationale: str = ""                       # 标注理由(可解释性)
    annotator_id: str = ""                    # 标注者 ID(AI 或人)
    difficulty: str = "medium"                # 难度:simple/medium/difficult
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class AnnotationConfig(BaseModel):
    """标注配置(版本管理)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    industry_id: str
    object_id: str
    golden_data: GoldenData = Field(default_factory=GoldenData)
    guideline: Optional[AnnotationGuideline] = None
    presets: AgentPresets = Field(default_factory=AgentPresets)
    llm_backend: str = "mock"                 # LLM 后端:openai/deepseek/qwen/mock
    llm_models: List[str] = Field(default_factory=lambda: ["gpt-4o-mini"])  # 候选模型
    cost_estimate: Optional[Dict[str, Any]] = None  # 成本预估
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"
