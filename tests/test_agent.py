# -*- coding: utf-8 -*-
"""核心链路单测（标准库 unittest，无需 chromadb / API Key）。

    python -m unittest discover -s tests -v

覆盖点：
1. 报告解析与项目名归一化
2. 参考区间性别分层
3. 严重度分级（RDW 这类窄区间不能被误判危急值）
4. 危急值判定（血钾）
5. 交叉判读模式（小细胞低色素性贫血）
6. 淋巴细胞按绝对值而非百分比判定（历史漏判的回归测试）
7. 审校回退的轮次上限
8. 端到端跑通并输出免责声明
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import LabReportGraph  # noqa: E402
from src.llm import MockLLM  # noqa: E402
from src.models import Flag, LabItem, LabState, Severity, Sex  # noqa: E402
from src.parser import parse_by_rule  # noqa: E402
from src.rules import build_correlations, judge_items  # noqa: E402

SAMPLE_CBC = """性别：男  年龄：34
血常规
白细胞 WBC     6.8   10^9/L   3.5-9.5
红细胞 RBC     4.10  10^12/L  4.3-5.8   ↓
血红蛋白 HGB   98    g/L      130-175   ↓
红细胞比容 HCT  0.36  L/L      0.40-0.50 ↓
平均红细胞体积 MCV  76.5 fL    82-100    ↓
平均红细胞血红蛋白量 MCH  23.9 pg  27-34 ↓
平均红细胞血红蛋白浓度 MCHC 312 g/L 316-354 ↓
红细胞体积分布宽度 RDW  16.8 %  11.5-14.5 ↑
血小板 PLT     245   10^9/L   125-350
"""


def item(code: str, value: float, unit: str = "") -> LabItem:
    return LabItem(code=code, name=code, value=value, unit=unit)


class TestParser(unittest.TestCase):
    def test_parse_sample(self) -> None:
        items = parse_by_rule(SAMPLE_CBC)
        codes = {i.code for i in items}
        self.assertGreaterEqual(len(items), 8)
        # 中文名要归一化成标准 code
        self.assertIn("HGB", codes)
        self.assertIn("MCV", codes)

    def test_skip_unparsable_line(self) -> None:
        """表头、空行、纯文字不能被当成指标。"""
        items = parse_by_rule("血常规\n项目 结果 单位 参考范围\n")
        self.assertEqual(items, [])


class TestJudge(unittest.TestCase):
    def test_sex_stratified_range(self) -> None:
        """同一 HGB 120：男性偏低，女性正常。"""
        male = judge_items([item("HGB", 120)], Sex.MALE)[0]
        female = judge_items([item("HGB", 120)], Sex.FEMALE)[0]
        self.assertEqual(male.flag, Flag.LOW)
        self.assertEqual(female.flag, Flag.NORMAL)

    def test_narrow_range_not_critical(self) -> None:
        """RDW 11.5-14.5 跨度只有 3，轻微升高不能判危急值。"""
        rdw = judge_items([item("RDW", 16.8, "%")], Sex.MALE)[0]
        self.assertEqual(rdw.flag, Flag.HIGH)
        self.assertNotEqual(rdw.severity, Severity.CRITICAL)

    def test_critical_potassium(self) -> None:
        k = judge_items([item("K", 6.8, "mmol/L")], Sex.MALE)[0]
        self.assertEqual(k.severity, Severity.CRITICAL)

    def test_normal_item(self) -> None:
        wbc = judge_items([item("WBC", 6.8)], Sex.MALE)[0]
        self.assertEqual(wbc.flag, Flag.NORMAL)
        self.assertEqual(wbc.severity, Severity.NORMAL)

    def test_ref_text_filled(self) -> None:
        hgb = judge_items([item("HGB", 98, "g/L")], Sex.MALE)[0]
        self.assertEqual(hgb.ref, "130-175")


class TestCorrelation(unittest.TestCase):
    def test_microcytic_anemia(self) -> None:
        items = judge_items(
            [item("HGB", 98), item("MCV", 76.5), item("MCH", 23.9), item("RDW", 16.8)],
            Sex.MALE,
        )
        patterns = {c.pattern: c for c in build_correlations(items, Sex.MALE)}
        self.assertIn("小细胞低色素性贫血", patterns)
        # RDW 同时升高时置信度更高
        self.assertGreaterEqual(patterns["小细胞低色素性贫血"].confidence, 0.88)

    def test_lymph_count_by_absolute_value(self) -> None:
        """WBC 正常但淋巴比例 60%：绝对值 3.6 > 3.2，仍应判增多。

        这是历史漏判的回归测试：早期只判「WBC↑ 且 LYMPH%↑」，
        白细胞总数正常时整条模式被漏掉。
        """
        items = judge_items([item("WBC", 6.0), item("LYMPH%", 60)], Sex.MALE)
        patterns = [c.pattern for c in build_correlations(items, Sex.MALE)]
        self.assertIn("淋巴细胞增多", patterns)

    def test_lymph_below_threshold(self) -> None:
        """比例 45%（正常）且绝对值 1.8，不应报增多。"""
        items = judge_items([item("WBC", 4.0), item("LYMPH%", 45)], Sex.MALE)
        patterns = [c.pattern for c in build_correlations(items, Sex.MALE)]
        self.assertNotIn("淋巴细胞增多", patterns)

    def test_neutrophil_absolute(self) -> None:
        items = judge_items([item("WBC", 12.0), item("NEUT%", 80)], Sex.MALE)
        patterns = [c.pattern for c in build_correlations(items, Sex.MALE)]
        self.assertIn("中性粒细胞增多", patterns)


class TestGraph(unittest.TestCase):
    def test_review_rewrite_limit(self) -> None:
        """审校不过最多回退 2 轮，超过必须进 finalize，避免死循环。"""
        state = LabState(raw_text="x", review_notes=["依据不足"], review_round=0)
        self.assertEqual(LabReportGraph.should_rewrite(state), "rewrite")
        state.review_round = 2
        self.assertEqual(LabReportGraph.should_rewrite(state), "finalize")

    def test_end_to_end(self) -> None:
        graph = LabReportGraph(llm=MockLLM(), retriever=None)
        report = graph.run(SAMPLE_CBC)
        self.assertEqual(report.panel_type.value, "血常规")
        self.assertGreaterEqual(len(report.abnormal_items), 4)
        codes = {i.code for i in report.abnormal_items}
        self.assertIn("HGB", codes)
        self.assertTrue(report.disclaimer)
        self.assertTrue(report.summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
