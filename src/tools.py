"""Function Calling 工具集。

暴露给模型的工具刻意做得"窄而准"：每个工具只解决一类确定性问题，
把能从代码算出来的东西全部从模型手里拿走，减少幻觉。
"""

from __future__ import annotations

from typing import Any

from .models import NAME_TO_CODE, REFERENCE_RANGES, Sex, get_range, resolve_code

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_reference_range",
            "description": "查询某个检验项目的参考区间。需要传入项目中文名或英文缩写，以及性别。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "检验项目名称或缩写，如 血红蛋白 / HGB"},
                    "sex": {"type": "string", "enum": ["male", "female", "unknown"], "description": "性别"},
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_egfr",
            "description": "用 CKD-EPI 公式根据血肌酐估算肾小球滤过率 eGFR。",
            "parameters": {
                "type": "object",
                "properties": {
                    "creatinine": {"type": "number", "description": "血肌酐 μmol/L"},
                    "sex": {"type": "string", "enum": ["male", "female"]},
                    "age": {"type": "integer", "description": "年龄"},
                },
                "required": ["creatinine", "sex", "age"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_critical",
            "description": "判断某个检验数值是否达到危急值，需要立即临床干预。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "检验项目名称或缩写"},
                    "value": {"type": "number"},
                },
                "required": ["item", "value"],
            },
        },
    },
]


def query_reference_range(item: str, sex: str = "unknown") -> dict[str, Any]:
    code = resolve_code(item)
    if code is None:
        return {"found": False, "reason": f"未收录项目：{item}"}
    rng = get_range(code, Sex(sex))
    if rng is None:
        return {"found": False, "reason": f"未收录参考区间：{code}"}
    return {
        "found": True,
        "code": code,
        "low": rng.low,
        "high": rng.high,
        "text": rng.text,
    }


def calc_egfr(creatinine: float, sex: str, age: int = 40) -> dict[str, Any]:
    k = 0.9 if sex == "male" else 0.7
    alpha = -0.302 if sex == "male" else -0.241
    factor = 1.0 if sex == "male" else 1.012
    ratio = creatinine / k
    egfr = (
        142
        * min(ratio, 1.0) ** alpha
        * max(ratio, 1.0) ** -1.200
        * (0.9938**age)
        * factor
    )
    stage = "G1" if egfr >= 90 else "G2" if egfr >= 60 else "G3a" if egfr >= 45 else "G3b" if egfr >= 30 else "G4/5"
    return {"egfr": round(egfr, 1), "ckd_stage": stage}


def check_critical(item: str, value: float) -> dict[str, Any]:
    from .models import CRITICAL_THRESHOLDS

    code = (resolve_code(item) or item).upper()
    crit = CRITICAL_THRESHOLDS.get(code)
    if not crit:
        return {"is_critical": False, "reason": "该项目未配置危急值阈值"}
    low, high = crit
    return {
        "is_critical": value < low or (high and value > high),
        "threshold": {"low": low, "high": high},
    }


TOOL_REGISTRY = {
    "query_reference_range": query_reference_range,
    "calc_egfr": calc_egfr,
    "check_critical": check_critical,
}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """工具分发入口。"""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"未知工具：{name}"}
    try:
        return fn(**arguments)
    except Exception as exc:  # 工具报错要能反馈给模型重试，而不是整条链路崩掉
        return {"error": str(exc)}


def list_supported_items() -> list[str]:
    return sorted({c.split("|")[0] for c in REFERENCE_RANGES} | set(NAME_TO_CODE.values()))
