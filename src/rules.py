"""异常判定规则引擎 + 指标关联推理。

这两块刻意不用 LLM：
- 参考区间比对是确定性计算，交给模型既贵又会错；
- 关联推理用「模式库」表达检验科的经典判读思路，可解释、可审计、能写进病历质控。

LLM 只负责最后一步：把结构化结论翻译成患者看得懂的话。
"""

from __future__ import annotations

from loguru import logger

from .models import (
    CRITICAL_THRESHOLDS,
    Correlation,
    Flag,
    LabItem,
    Severity,
    Sex,
    get_range,
)


def judge_items(items: list[LabItem], sex: Sex) -> list[LabItem]:
    """逐项比对参考区间，打上 flag / severity / ref。"""
    for item in items:
        if item.value is None:
            # 定性项目：非"阴性"即异常
            if item.qualitative and item.qualitative not in ("阴性", "-", "正常"):
                item.flag = Flag.HIGH
                item.severity = Severity.MODERATE
            continue

        rng = get_range(item.code or "", sex) if item.code else None
        if rng is None:
            continue

        item.ref = (
            f"{_fmt(rng.low)}-{_fmt(rng.high)}" if rng.high is not None else f"≥{_fmt(rng.low)}"
        )

        if rng.contains(item.value):
            item.flag = Flag.NORMAL
            item.severity = Severity.NORMAL
            continue

        item.flag = Flag.HIGH if rng.low is not None and item.value > (rng.high or rng.low) else Flag.LOW
        item.severity = _grade(item, rng.low, rng.high)
    return items


def _fmt(value: float | None) -> str:
    """去掉浮点尾巴：130.0 → 130，0.4 → 0.4。"""
    if value is None:
        return "-"
    return f"{value:g}"


def _grade(item: LabItem, low: float | None, high: float | None) -> Severity:
    """偏离程度分级 + 危急值判定。"""
    value = item.value or 0

    # 危急值：绝对值阈值，与参考区间无关
    crit = CRITICAL_THRESHOLDS.get((item.code or "").upper())
    if crit:
        clow, chigh = crit
        if value < clow or (chigh and value > chigh):
            item.severity = Severity.CRITICAL
            return Severity.CRITICAL

    # 用「偏离倍数」而非「占参考区间跨度的比例」分级。
    # 参考区间跨度窄的项目（如 RDW 11.5-14.5），用跨度比例会把轻微升高也判成危急值，
    # 与临床直觉不符。偏离倍数 = 实测值 / 上限（偏高）或 下限 / 实测值（偏低）。
    if value > high if high is not None else False:
        fold = value / high
    elif low is not None and low > 0:
        fold = low / value
    else:
        fold = 1.5

    if fold < 1.15:
        return Severity.BORDERLINE
    if fold < 1.5:
        return Severity.MODERATE
    return Severity.CRITICAL


def _get(items: list[LabItem], code: str) -> LabItem | None:
    for item in items:
        if item.code == code:
            return item
    return None


def _low(item: LabItem | None) -> bool:
    return item is not None and item.flag == Flag.LOW


def _high(item: LabItem | None) -> bool:
    return item is not None and item.flag == Flag.HIGH


# 模式库：(名称, 判定函数, 置信度, 说明)
def build_correlations(items: list[LabItem], sex: Sex = Sex.UNKNOWN) -> list[Correlation]:
    """指标关联推理 —— 检验科经典判读思路的形式化。"""
    results: list[Correlation] = []

    hgb, mcv, mch, rdw = _get(items, "HGB"), _get(items, "MCV"), _get(items, "MCH"), _get(items, "RDW")

    if _low(hgb) and _low(mcv) and _low(mch):
        results.append(
            Correlation(
                pattern="小细胞低色素性贫血",
                evidence=[f"HGB {hgb.value}{hgb.unit or ''}↓", f"MCV {mcv.value}↓", f"MCH {mch.value}↓"],
                confidence=0.88 if _high(rdw) else 0.78,
                note="常见于缺铁性贫血，建议查血清铁、铁蛋白；RDW 升高支持缺铁。",
            )
        )
    elif _low(hgb) and _high(mcv):
        results.append(
            Correlation(
                pattern="大细胞性贫血",
                evidence=[f"HGB {hgb.value}↓", f"MCV {mcv.value}↑"],
                confidence=0.82,
                note="常见于叶酸 / 维生素 B12 缺乏，建议查血清叶酸、B12。",
            )
        )
    elif _low(hgb):
        results.append(
            Correlation(
                pattern="正细胞性贫血",
                evidence=[f"HGB {hgb.value}↓", f"MCV {mcv.value if mcv else '正常'}"],
                confidence=0.7,
                note="可见于慢性病贫血、急性失血、溶血等，需结合网织红细胞。",
            )
        )

    wbc, neut = _get(items, "WBC"), _get(items, "NEUT%")
    lymph = _get(items, "LYMPH%")
    eos = _get(items, "EOS%")
    # 百分比会骗人：白细胞总数正常、淋巴比例 60% 时，绝对值其实已经升高。
    # 所以判「增多」要先换算绝对值（WBC × 百分比 / 100），再用绝对值阈值判断。
    wbc_v = wbc.value if (wbc and wbc.value is not None) else None

    def _abs_count(pct_item) -> float | None:
        if wbc_v is None or pct_item is None or pct_item.value is None:
            return None
        return wbc_v * pct_item.value / 100.0

    neut_abs = _abs_count(neut)
    lymph_abs = _abs_count(lymph)

    if (neut_abs is not None and neut_abs > 7.5) or (_high(wbc) and _high(neut)):
        results.append(
            Correlation(
                pattern="中性粒细胞增多",
                evidence=(
                    [f"中性粒细胞绝对值 {neut_abs:.2f}↑"]
                    if neut_abs is not None and neut_abs > 7.5
                    else [f"WBC {getattr(wbc, 'value', '-')}↑", f"NEUT% {getattr(neut, 'value', '-')}↑"]
                ),
                confidence=0.85,
                note="典型细菌感染或应激反应表现，建议结合 CRP、PCT 与临床症状。",
            )
        )
    if (lymph_abs is not None and lymph_abs > 3.2) or (_high(wbc) and _high(lymph)):
        results.append(
            Correlation(
                pattern="淋巴细胞增多",
                evidence=(
                    [f"淋巴细胞绝对值 {lymph_abs:.2f}↑"]
                    if lymph_abs is not None and lymph_abs > 3.2
                    else [f"WBC {getattr(wbc, 'value', '-')}↑", f"LYMPH% {getattr(lymph, 'value', '-')}↑"]
                ),
                confidence=0.78,
                note="常见于病毒感染，必要时查外周血涂片与病毒血清学。",
            )
        )
    if _high(eos):
        results.append(
            Correlation(
                pattern="嗜酸性粒细胞增多",
                evidence=[f"EOS% {eos.value}↑"],
                confidence=0.72,
                note="提示过敏性疾病或寄生虫感染可能。",
            )
        )

    alt, ast, ggt, tbil = (
        _get(items, "ALT"),
        _get(items, "AST"),
        _get(items, "GGT"),
        _get(items, "TBIL"),
    )
    if _high(alt) or _high(ast):
        ratio = None
        if alt and ast and alt.value and ast.value:
            ratio = ast.value / alt.value
        results.append(
            Correlation(
                pattern="肝细胞损伤型肝功能异常",
                evidence=[f"ALT {alt.value}↑" if _high(alt) else "", f"AST {ast.value}↑" if _high(ast) else ""],
                confidence=0.8,
                note=(
                    f"AST/ALT={ratio:.2f}，>1 提示酒精性肝病或肝纤维化可能；"
                    f"<1 更倾向病毒性肝炎或脂肪肝。" if ratio else "建议结合病毒性肝炎标志物与肝脏超声。"
                ),
            )
        )
    if (_high(ggt) or _high(_get(items, "ALP"))) and _high(tbil):
        results.append(
            Correlation(
                pattern="胆汁淤积型肝功能异常",
                evidence=[f"GGT {ggt.value}↑" if _high(ggt) else "ALP↑", f"TBIL {tbil.value}↑"],
                confidence=0.83,
                note="GGT 与 ALP 同步升高提示胆道梗阻或胆汁淤积，建议腹部影像学检查。",
            )
        )

    cr, bun, ua = _get(items, "Cr"), _get(items, "BUN"), _get(items, "UA")
    if _high(cr):
        egfr = _calc_egfr(cr.value or 0, sex)
        results.append(
            Correlation(
                pattern="肾功能指标异常",
                evidence=[f"Cr {cr.value}{cr.unit or ''}↑", f"BUN {bun.value if bun else '-'}"],
                confidence=0.86,
                note=f"估算 eGFR ≈ {egfr:.0f} mL/min/1.73m²，建议肾内科随诊并复查。",
            )
        )
    if _high(ua):
        results.append(
            Correlation(
                pattern="高尿酸血症",
                evidence=[f"UA {ua.value}{ua.unit or ''}↑"],
                confidence=0.88,
                note="与痛风、代谢综合征相关，建议低嘌呤饮食并监测。",
            )
        )

    glu, tg, tc, hdl, ldl = (
        _get(items, "GLU"),
        _get(items, "TG"),
        _get(items, "TC"),
        _get(items, "HDL-C"),
        _get(items, "LDL-C"),
    )
    metabolic: list[str] = []
    if _high(glu):
        metabolic.append(f"GLU {glu.value}↑")
    if _high(tg):
        metabolic.append(f"TG {tg.value}↑")
    if hdl and hdl.flag == Flag.LOW:
        metabolic.append(f"HDL-C {hdl.value}↓")
    if _high(ldl) or _high(tc):
        metabolic.append((f"LDL-C {ldl.value}↑" if _high(ldl) else f"TC {tc.value}↑"))
    if len(metabolic) >= 2:
        results.append(
            Correlation(
                pattern="代谢综合征倾向",
                evidence=metabolic,
                confidence=0.8,
                note="多项糖脂代谢指标异常，建议内分泌科评估并干预生活方式。",
            )
        )

    logger.debug("关联推理命中 {} 条模式", len(results))
    return results


def _calc_egfr(creatinine: float, sex: Sex, age: int = 40) -> float:
    """CKD-EPI 简化估算（演示用，非临床诊断工具）。"""
    k = 0.9 if sex == Sex.MALE else 0.7
    alpha = -0.302 if sex == Sex.MALE else -0.241
    factor = 1.0 if sex == Sex.MALE else 1.012
    ratio = creatinine / k
    min_term = min(ratio, 1.0) ** alpha
    max_term = max(ratio, 1.0) ** -1.200
    return 142 * min_term * max_term * (0.9938**age) * factor
