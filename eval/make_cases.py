"""生成 24 例检验单回归集。

标签怎么来的（重要，面试会问）：
- 异常项标签：由 `src/models.py` 里的 REFERENCE_RANGES 直接判定（谁超出区间谁异常），
  因此这一项衡量的是「解析 + 区间判定链路是否正确」，不是临床诊断准确率；
- 关联模式标签：按检验科经典判读定义手工指定（如 HGB↓+MCV↓+MCH↓ → 小细胞低色素性贫血），
  引擎必须自己推出来才算命中。

用途：改动规则 / 解析 / 检索后跑一遍，防止退化。

    python eval/make_cases.py      # 生成 eval/labels.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import REFERENCE_RANGES, Sex, get_range  # noqa: E402

# (中文名, 缩写, 数值, 单位)
Row = tuple[str, str, float, str]

# 常用正常项，作为背景噪声，避免"只有异常项"的失真样本
NORMAL_BG: list[Row] = [
    ("白细胞", "WBC", 6.2, "10^9/L"),
    ("血小板", "PLT", 236, "10^9/L"),
    ("中性粒细胞百分比", "NEUT%", 61.0, "%"),
    ("淋巴细胞百分比", "LYMPH%", 30.0, "%"),
]

CASES: list[dict] = [
    # ---------- 正常 ----------
    {"id": "c01", "sex": "male", "age": 34, "rows": [
        ("红细胞", "RBC", 4.80, "10^12/L"),
        ("血红蛋白", "HGB", 152, "g/L"),
        ("红细胞比容", "HCT", 0.45, "L/L"),
        ("平均红细胞体积", "MCV", 90.0, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 30.5, "pg"),
        ("平均红细胞血红蛋白浓度", "MCHC", 335, "g/L"),
        ("红细胞体积分布宽度", "RDW", 12.8, "%"),
    ] + NORMAL_BG, "patterns": []},
    {"id": "c02", "sex": "female", "age": 28, "rows": [
        ("红细胞", "RBC", 4.35, "10^12/L"),
        ("血红蛋白", "HGB", 126, "g/L"),
        ("红细胞比容", "HCT", 0.39, "L/L"),
        ("平均红细胞体积", "MCV", 88.0, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 29.0, "pg"),
        ("平均红细胞血红蛋白浓度", "MCHC", 325, "g/L"),
    ] + NORMAL_BG, "patterns": []},
    # 边界值：正好等于下限，应判正常
    {"id": "c03", "sex": "male", "age": 40, "rows": [
        ("血红蛋白", "HGB", 130, "g/L"),
        ("红细胞", "RBC", 4.30, "10^12/L"),
        ("平均红细胞体积", "MCV", 82.0, "fL"),
    ] + NORMAL_BG, "patterns": []},

    # ---------- 小细胞低色素性贫血 ----------
    {"id": "c04", "sex": "male", "age": 34, "rows": [
        ("红细胞", "RBC", 4.10, "10^12/L"),
        ("血红蛋白", "HGB", 98, "g/L"),
        ("红细胞比容", "HCT", 0.36, "L/L"),
        ("平均红细胞体积", "MCV", 76.5, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 23.9, "pg"),
        ("平均红细胞血红蛋白浓度", "MCHC", 312, "g/L"),
        ("红细胞体积分布宽度", "RDW", 16.8, "%"),
    ] + NORMAL_BG, "patterns": ["小细胞低色素性贫血"]},
    {"id": "c05", "sex": "female", "age": 26, "rows": [
        ("血红蛋白", "HGB", 92, "g/L"),
        ("红细胞", "RBC", 3.60, "10^12/L"),
        ("平均红细胞体积", "MCV", 74.0, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 22.0, "pg"),
        ("平均红细胞血红蛋白浓度", "MCHC", 300, "g/L"),
        ("红细胞体积分布宽度", "RDW", 17.5, "%"),
    ] + NORMAL_BG, "patterns": ["小细胞低色素性贫血"]},
    {"id": "c06", "sex": "male", "age": 55, "rows": [
        ("血红蛋白", "HGB", 105, "g/L"),
        ("平均红细胞体积", "MCV", 78.0, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 25.0, "pg"),
        ("红细胞体积分布宽度", "RDW", 15.9, "%"),
        ("红细胞", "RBC", 4.20, "10^12/L"),
    ] + NORMAL_BG, "patterns": ["小细胞低色素性贫血"]},

    # ---------- 大细胞性贫血 ----------
    {"id": "c07", "sex": "male", "age": 62, "rows": [
        ("血红蛋白", "HGB", 108, "g/L"),
        ("红细胞", "RBC", 3.20, "10^12/L"),
        ("平均红细胞体积", "MCV", 108.0, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 35.5, "pg"),
    ] + NORMAL_BG, "patterns": ["大细胞性贫血"]},
    {"id": "c08", "sex": "female", "age": 45, "rows": [
        ("血红蛋白", "HGB", 104, "g/L"),
        ("红细胞", "RBC", 3.10, "10^12/L"),
        ("平均红细胞体积", "MCV", 112.0, "fL"),
    ] + NORMAL_BG, "patterns": ["大细胞性贫血"]},

    # ---------- 正细胞性贫血 ----------
    {"id": "c09", "sex": "male", "age": 48, "rows": [
        ("血红蛋白", "HGB", 112, "g/L"),
        ("红细胞", "RBC", 3.90, "10^12/L"),
        ("平均红细胞体积", "MCV", 89.0, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 28.5, "pg"),
    ] + NORMAL_BG, "patterns": ["正细胞性贫血"]},

    # ---------- 肝功能 ----------
    {"id": "c10", "sex": "male", "age": 38, "rows": [
        ("丙氨酸氨基转移酶", "ALT", 128, "U/L"),
        ("天门冬氨酸氨基转移酶", "AST", 96, "U/L"),
        ("总胆红素", "TBIL", 18.0, "umol/L"),
        ("白蛋白", "ALB", 44, "g/L"),
    ] + NORMAL_BG, "patterns": ["肝细胞损伤型肝功能异常"]},
    {"id": "c11", "sex": "female", "age": 41, "rows": [
        ("丙氨酸氨基转移酶", "ALT", 78, "U/L"),
        ("天门冬氨酸氨基转移酶", "AST", 52, "U/L"),
        ("白蛋白", "ALB", 42, "g/L"),
    ] + NORMAL_BG, "patterns": ["肝细胞损伤型肝功能异常"]},
    {"id": "c12", "sex": "male", "age": 57, "rows": [
        ("谷氨酰转肽酶", "GGT", 186, "U/L"),
        ("碱性磷酸酶", "ALP", 248, "U/L"),
        ("总胆红素", "TBIL", 36.5, "umol/L"),
        ("丙氨酸氨基转移酶", "ALT", 42, "U/L"),
    ] + NORMAL_BG, "patterns": ["胆汁淤积型肝功能异常"]},

    # ---------- 肾功能 ----------
    {"id": "c13", "sex": "male", "age": 66, "rows": [
        ("肌酐", "Cr", 168, "umol/L"),
        ("尿素", "BUN", 12.6, "mmol/L"),
        ("尿酸", "UA", 402, "umol/L"),
    ] + NORMAL_BG, "patterns": ["肾功能指标异常"]},
    {"id": "c14", "sex": "female", "age": 58, "rows": [
        ("肌酐", "Cr", 96, "umol/L"),
        ("尿素", "BUN", 9.4, "mmol/L"),
    ] + NORMAL_BG, "patterns": ["肾功能指标异常"]},
    {"id": "c15", "sex": "male", "age": 44, "rows": [
        ("尿酸", "UA", 528, "umol/L"),
    ] + NORMAL_BG, "patterns": ["高尿酸血症"]},

    # ---------- 白细胞分类 ----------
    {"id": "c16", "sex": "male", "age": 30, "rows": [
        ("白细胞", "WBC", 14.6, "10^9/L"),
        ("中性粒细胞百分比", "NEUT%", 84.0, "%"),
        ("淋巴细胞百分比", "LYMPH%", 12.0, "%"),
        ("血小板", "PLT", 240, "10^9/L"),
    ], "patterns": ["中性粒细胞增多"]},
    {"id": "c17", "sex": "female", "age": 24, "rows": [
        ("白细胞", "WBC", 11.2, "10^9/L"),
        ("中性粒细胞百分比", "NEUT%", 79.5, "%"),
        ("淋巴细胞百分比", "LYMPH%", 16.0, "%"),
    ] + [("血小板", "PLT", 255, "10^9/L")], "patterns": ["中性粒细胞增多"]},
    {"id": "c18", "sex": "male", "age": 22, "rows": [
        ("白细胞", "WBC", 9.0, "10^9/L"),
        ("淋巴细胞百分比", "LYMPH%", 62.0, "%"),
        ("中性粒细胞百分比", "NEUT%", 32.0, "%"),
    ] + [("血小板", "PLT", 210, "10^9/L")], "patterns": ["淋巴细胞增多"]},
    {"id": "c19", "sex": "female", "age": 35, "rows": [
        ("白细胞", "WBC", 7.8, "10^9/L"),
        ("嗜酸性粒细胞百分比", "EOS%", 12.5, "%"),
    ] + NORMAL_BG, "patterns": ["嗜酸性粒细胞增多"]},

    # ---------- 血小板 / 白细胞异常 ----------
    {"id": "c20", "sex": "male", "age": 29, "rows": [
        ("血小板", "PLT", 68, "10^9/L"),
    ] + NORMAL_BG[0:1] + NORMAL_BG[2:], "patterns": []},
    {"id": "c21", "sex": "female", "age": 33, "rows": [
        ("白细胞", "WBC", 2.6, "10^9/L"),
    ] + [("血小板", "PLT", 220, "10^9/L"), ("中性粒细胞百分比", "NEUT%", 58.0, "%")],
        "patterns": []},

    # ---------- 复合 ----------
    {"id": "c22", "sex": "male", "age": 51, "rows": [
        ("血红蛋白", "HGB", 96, "g/L"),
        ("平均红细胞体积", "MCV", 75.0, "fL"),
        ("平均红细胞血红蛋白量", "MCH", 22.8, "pg"),
        ("红细胞体积分布宽度", "RDW", 16.4, "%"),
        ("丙氨酸氨基转移酶", "ALT", 112, "U/L"),
        ("天门冬氨酸氨基转移酶", "AST", 88, "U/L"),
    ] + NORMAL_BG, "patterns": ["小细胞低色素性贫血", "肝细胞损伤型肝功能异常"]},

    # ---------- 甲功 / 血糖 ----------
    {"id": "c23", "sex": "female", "age": 39, "rows": [
        ("促甲状腺激素", "TSH", 9.60, "mIU/L"),
        ("游离甲状腺素", "FT4", 9.8, "pmol/L"),
    ] + NORMAL_BG, "patterns": []},
    {"id": "c24", "sex": "male", "age": 46, "rows": [
        ("空腹血糖", "GLU", 9.8, "mmol/L"),
        ("甘油三酯", "TG", 2.6, "mmol/L"),
        ("总胆固醇", "TC", 6.1, "mmol/L"),
    ] + NORMAL_BG, "patterns": []},
]


def expected_abnormal(rows: list[Row], sex: Sex) -> list[str]:
    """按区间表判定哪些项异常（用于生成标签，同时作为解析链路的对照）。"""
    from src.models import resolve_code

    out = []
    for name, abbr, value, _unit in rows:
        code = resolve_code(name) or resolve_code(abbr)
        if not code:
            continue
        rng = get_range(code, sex)
        if rng is None or rng.contains(value):
            continue
        out.append(code)
    return sorted(set(out))


def render(case: dict) -> str:
    sex_cn = "男" if case["sex"] == "male" else "女"
    lines = [f"性别：{sex_cn}  年龄：{case['age']}", "标本类型：静脉血", "检验项目：检验", ""]
    for name, abbr, value, unit in case["rows"]:
        v = f"{value:g}"
        lines.append(f"{name} {abbr:<6} {v:<8} {unit}")
    return "\n".join(lines) + "\n"


def main() -> None:
    out = Path(__file__).resolve().parent / "labels.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for c in CASES:
            sex = Sex.MALE if c["sex"] == "male" else Sex.FEMALE
            rec = {
                "id": c["id"],
                "sex": c["sex"],
                "age": c["age"],
                "expect_abnormal": expected_abnormal(c["rows"], sex),
                "expect_patterns": c["patterns"],
                "text": render(c),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"写入 {n} 例 -> {out}")


if __name__ == "__main__":
    main()
