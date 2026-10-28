"""数据模型 + 检验项目参考区间库。

参考区间按「项目 → 人群分层（性别 / 年龄）」组织，这是通用大模型最容易出错、
而检验专业出身的人天然会做对的地方。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #
class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class Flag(str, Enum):
    """与报告单上的箭头对应。"""

    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"


class Severity(str, Enum):
    NORMAL = "normal"
    BORDERLINE = "borderline"  # 轻度偏离
    MODERATE = "moderate"  # 中度
    CRITICAL = "critical"  # 危急值，需立即处理


class PanelType(str, Enum):
    CBC = "血常规"
    BIOCHEM = "生化"
    URINE = "尿常规"
    IMMUNE = "免疫/传染病标志物"
    THYROID = "甲状腺功能"
    TUMOR = "肿瘤标志物"
    UNKNOWN = "未知"


# --------------------------------------------------------------------------- #
# 参考区间
# --------------------------------------------------------------------------- #
class RefRange(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None
    text: Optional[str] = None  # 定性项目（阴性/阳性）用

    def contains(self, value: float) -> bool:
        if self.low is None or self.high is None:
            return True
        return self.low <= value <= self.high


# key 约定："CODE" 或 "CODE|male" / "CODE|female"
REFERENCE_RANGES: dict[str, RefRange] = {
    # ---------------- 血常规 ----------------
    "WBC": RefRange(low=3.5, high=9.5),
    "RBC|male": RefRange(low=4.3, high=5.8),
    "RBC|female": RefRange(low=3.8, high=5.1),
    "HGB|male": RefRange(low=130, high=175),
    "HGB|female": RefRange(low=115, high=150),
    "HCT|male": RefRange(low=0.40, high=0.50),
    "HCT|female": RefRange(low=0.35, high=0.45),
    "PLT": RefRange(low=125, high=350),
    "MCV": RefRange(low=82, high=100),
    "MCH": RefRange(low=27, high=34),
    "MCHC": RefRange(low=316, high=354),
    "RDW": RefRange(low=11.5, high=14.5),
    "NEUT%": RefRange(low=40, high=75),
    "LYMPH%": RefRange(low=20, high=50),
    "MONO%": RefRange(low=3, high=10),
    "EOS%": RefRange(low=0.4, high=8),
    "BASO%": RefRange(low=0, high=1),
    # ---------------- 生化 ----------------
    "ALT|male": RefRange(low=9, high=50),
    "ALT|female": RefRange(low=7, high=40),
    "AST": RefRange(low=15, high=40),
    "GGT|male": RefRange(low=10, high=60),
    "GGT|female": RefRange(low=7, high=45),
    "ALP": RefRange(low=45, high=125),
    "TBIL": RefRange(low=3.4, high=20.5),
    "DBIL": RefRange(low=0, high=6.8),
    "TP": RefRange(low=65, high=85),
    "ALB": RefRange(low=40, high=55),
    "BUN": RefRange(low=2.9, high=8.2),
    "Cr|male": RefRange(low=57, high=97),
    "Cr|female": RefRange(low=41, high=73),
    "UA|male": RefRange(low=208, high=428),
    "UA|female": RefRange(low=155, high=357),
    "GLU": RefRange(low=3.9, high=6.1),
    "TC": RefRange(low=0, high=5.2),
    "TG": RefRange(low=0, high=1.7),
    "HDL-C|male": RefRange(low=1.0, high=None),
    "HDL-C|female": RefRange(low=1.3, high=None),
    "LDL-C": RefRange(low=0, high=3.4),
    "K": RefRange(low=3.5, high=5.3),
    "NA": RefRange(low=137, high=147),
    "CL": RefRange(low=99, high=110),
    "CA": RefRange(low=2.11, high=2.52),
    # ---------------- 甲状腺功能 ----------------
    "TSH": RefRange(low=0.55, high=4.78),
    "FT3": RefRange(low=3.5, high=6.5),
    "FT4": RefRange(low=11.5, high=22.7),
    # ---------------- 肿瘤标志物 ----------------
    "AFP": RefRange(low=0, high=7.0),
    "CEA": RefRange(low=0, high=5.0),
    "CA125": RefRange(low=0, high=35),
    "CA199": RefRange(low=0, high=37),
    "PSA": RefRange(low=0, high=4.0),
}

# 中文名 → 标准 code，用于解析时归一化
NAME_TO_CODE: dict[str, str] = {
    "白细胞": "WBC", "白细胞计数": "WBC",
    "红细胞": "RBC", "红细胞计数": "RBC",
    "血红蛋白": "HGB",
    "红细胞比容": "HCT", "红细胞压积": "HCT",
    "血小板": "PLT", "血小板计数": "PLT",
    "平均红细胞体积": "MCV",
    "平均红细胞血红蛋白量": "MCH",
    "平均红细胞血红蛋白浓度": "MCHC",
    "红细胞体积分布宽度": "RDW",
    "中性粒细胞百分比": "NEUT%",
    "淋巴细胞百分比": "LYMPH%",
    "单核细胞百分比": "MONO%",
    "嗜酸性粒细胞百分比": "EOS%",
    "嗜碱性粒细胞百分比": "BASO%",
    "丙氨酸氨基转移酶": "ALT", "谷丙转氨酶": "ALT",
    "天门冬氨酸氨基转移酶": "AST", "谷草转氨酶": "AST",
    "谷氨酰转肽酶": "GGT",
    "碱性磷酸酶": "ALP",
    "总胆红素": "TBIL",
    "直接胆红素": "DBIL",
    "总蛋白": "TP",
    "白蛋白": "ALB",
    "尿素": "BUN", "尿素氮": "BUN",
    "肌酐": "Cr",
    "尿酸": "UA",
    "葡萄糖": "GLU", "血糖": "GLU", "空腹血糖": "GLU",
    "总胆固醇": "TC",
    "甘油三酯": "TG",
    "高密度脂蛋白胆固醇": "HDL-C",
    "低密度脂蛋白胆固醇": "LDL-C",
    "钾": "K", "钠": "NA", "氯": "CL", "钙": "CA",
    "促甲状腺激素": "TSH",
    "游离三碘甲状腺原氨酸": "FT3",
    "游离甲状腺素": "FT4",
    "甲胎蛋白": "AFP",
    "癌胚抗原": "CEA",
    "糖类抗原125": "CA125",
    "糖类抗原199": "CA199", "糖类抗原19-9": "CA199",
    "前列腺特异性抗原": "PSA",
}

# 危急值：触及即告警，不允许模型"轻描淡写"
CRITICAL_THRESHOLDS: dict[str, tuple[float, float]] = {
    "K": (2.5, 6.5),
    "GLU": (2.8, 27.8),
    "PLT": (20, 999),
    "HGB": (50, 200),
    "CR": (0, 707),
}


def resolve_code(name: str) -> Optional[str]:
    """把中文项目名归一化成标准 code。"""
    key = name.strip().replace("（", "(").replace("）", ")")
    if key in NAME_TO_CODE:
        return NAME_TO_CODE[key]
    # 大小写不敏感兜底（报告单常写英文缩写）
    upper = key.upper()
    for code in REFERENCE_RANGES:
        if code.split("|")[0].upper() == upper:
            return code.split("|")[0]
    return None


def get_range(code: str, sex: Sex) -> Optional[RefRange]:
    """按性别取参考区间，无性别分层则回落到通用区间。"""
    if sex in (Sex.MALE, Sex.FEMALE):
        specific = REFERENCE_RANGES.get(f"{code}|{sex.value}")
        if specific:
            return specific
    return REFERENCE_RANGES.get(code)


# --------------------------------------------------------------------------- #
# 业务模型
# --------------------------------------------------------------------------- #
class LabItem(BaseModel):
    """单个检验项目。"""

    code: Optional[str] = None
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    qualitative: Optional[str] = None  # 阴性 / 阳性 / +
    flag: Flag = Flag.NORMAL
    severity: Severity = Severity.NORMAL
    ref: Optional[str] = None  # 形如 "130-175"


class Correlation(BaseModel):
    """指标关联推理结论。"""

    pattern: str = Field(description="如：小细胞低色素性贫血")
    evidence: list[str] = Field(default_factory=list, description="支撑该模式的指标组合")
    confidence: float = 0.0
    note: Optional[str] = None


class LabReport(BaseModel):
    """最终输出报告。"""

    panel_type: PanelType = PanelType.UNKNOWN
    abnormal_items: list[LabItem] = Field(default_factory=list)
    correlations: list[Correlation] = Field(default_factory=list)
    summary: str = ""
    advice: list[str] = Field(default_factory=list)
    disclaimer: str = "本报告为辅助解读，不构成诊断依据，请以临床医师意见为准。"
    review_passed: bool = True
    review_notes: list[str] = Field(default_factory=list)


class LabState(BaseModel):
    """LangGraph 节点间流转的状态。"""

    raw_text: str
    sex: Sex = Sex.UNKNOWN
    age: Optional[int] = None
    items: list[LabItem] = Field(default_factory=list)
    correlations: list[Correlation] = Field(default_factory=list)
    retrieved_docs: list[str] = Field(default_factory=list)
    draft_summary: str = ""
    draft_advice: list[str] = Field(default_factory=list)
    report: Optional[LabReport] = None
    review_round: int = 0
    review_notes: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
