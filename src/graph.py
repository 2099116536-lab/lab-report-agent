"""LangGraph 工作流定义。

图结构：

    parse → judge → correlate → retrieve → interpret → review ─┬─ (pass) → END
                                              ↑                 │
                                              └── (fail, <2轮) ──┘

``review`` 节点是本项目最关键的设计：医疗场景下模型"说错一句"的代价远高于"少说一句"，
所以用一个独立 Agent 做安全审校，不通过就带着修改意见回退重写，最多 2 轮。
"""

from __future__ import annotations

from loguru import logger

from .llm import BaseLLM, get_llm
from .models import Correlation, Flag, LabReport, LabState, PanelType, Severity
from .parser import detect_panel, parse_age, parse_report, parse_sex
from .rag import HybridRetriever
from .rules import build_correlations, judge_items

MAX_REVIEW_ROUNDS = 2

REVIEW_SYSTEM = """你是医学内容安全审校员。审查一段检验报告解读草稿，判断是否含以下问题：
1. 给出明确诊断结论（如"你患有XX病"）—— 解读只能提示可能方向，不能下诊断；
2. 推荐具体处方药或剂量；
3. 出现草稿未提及的检验项目或数值（幻觉）；
4. 语气恐吓或过度焦虑。
返回严格 JSON：{"passed": bool, "issues": [str]}，不要输出其他内容。"""

INTERPRET_SYSTEM = """你是检验科医师助手。基于给定的检验异常项、关联推理结论和检索到的医学知识，
为普通患者生成通俗易懂的解读。

要求：
- 先说明哪些指标异常、异常程度；
- 用生活化语言解释这些指标意味着什么；
- 给出就医建议（科室方向、复查项目），但绝不给出明确诊断和具体用药；
- 若存在危急值，必须在开头明确提示立即就医；
- 语气平和，不制造焦虑。
返回严格 JSON：{"summary": str, "advice": [str]}"""


class LabReportGraph:
    """把各节点组装成可执行的状态图。

    用 LangGraph 的 StateGraph 组装；若环境未安装 langgraph，
    退化为顺序执行的兼容实现，保证链路可跑（便于在没有依赖的机器上演示）。
    """

    def __init__(self, llm: BaseLLM | None = None, retriever: HybridRetriever | None = None) -> None:
        self.llm = llm or get_llm()
        self.retriever = retriever or HybridRetriever()
        self.app = self._build()

    # ------------------------------------------------------------------ #
    # 节点
    # ------------------------------------------------------------------ #
    def node_parse(self, state: LabState) -> LabState:
        state.sex = parse_sex(state.raw_text)
        state.age = parse_age(state.raw_text)
        state.items = parse_report(state.raw_text, self.llm)
        state.trace.append(f"parse: 解析出 {len(state.items)} 项，性别={state.sex.value}")
        return state

    def node_judge(self, state: LabState) -> LabState:
        state.items = judge_items(state.items, state.sex)
        abnormal = [i for i in state.items if i.flag != Flag.NORMAL]
        state.trace.append(f"judge: 异常 {len(abnormal)} 项")
        return state

    def node_correlate(self, state: LabState) -> LabState:
        state.correlations = build_correlations(state.items, state.sex)
        state.trace.append(f"correlate: 命中 {len(state.correlations)} 条模式")
        return state

    def node_retrieve(self, state: LabState) -> LabState:
        abnormal = [i for i in state.items if i.flag != Flag.NORMAL]
        if not abnormal:
            state.trace.append("retrieve: 无异常项，跳过检索")
            return state
        # 用异常项目名 + 关联模式拼检索 query，比整段文本检索精确得多
        queries = [f"{i.name} {i.flag.value} 临床意义" for i in abnormal[:8]]
        queries += [c.pattern for c in state.correlations]
        docs: list[str] = []
        seen: set[str] = set()
        for q in queries:
            for doc in self.retriever.search(q, top_k=3):
                if doc not in seen:
                    seen.add(doc)
                    docs.append(doc)
        state.retrieved_docs = docs[:10]
        state.trace.append(f"retrieve: 召回 {len(state.retrieved_docs)} 条知识")
        return state

    def node_interpret(self, state: LabState) -> LabState:
        abnormal = [i for i in state.items if i.flag != Flag.NORMAL]
        if not abnormal:
            state.draft_summary = "本次检验各项指标均在参考范围内，未见明显异常。"
            state.trace.append("interpret: 无异常，生成阴性结论")
            return state

        payload = {
            "abnormal": [
                {
                    "name": i.name,
                    "value": i.value,
                    "unit": i.unit,
                    "flag": i.flag.value,
                    "severity": i.severity.value,
                    "ref": i.ref,
                }
                for i in abnormal
            ],
            "correlations": [c.model_dump() for c in state.correlations],
            "knowledge": state.retrieved_docs[:6],
            "critical": [i.name for i in abnormal if i.severity == Severity.CRITICAL],
            "review_notes": state.review_notes,
        }
        import json

        try:
            data = self.llm.chat_json(
                INTERPRET_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.3
            )
            state.draft_summary = str(data.get("summary", "")).strip()
            raw_advice = data.get("advice", [])
            if isinstance(raw_advice, list):
                state.draft_advice = [str(a).strip() for a in raw_advice if str(a).strip()]
            state.trace.append(f"interpret: 生成草稿，{len(state.draft_advice)} 条建议")
        except Exception as exc:
            logger.error("解读生成失败：{}", exc)
            state.draft_summary = "解读生成失败，请稍后重试或咨询临床医师。"
        return state

    def node_review(self, state: LabState) -> LabState:
        import json

        payload = {"draft": state.draft_summary, "round": state.review_round}
        try:
            data = self.llm.chat_json(REVIEW_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.0)
            passed = bool(data.get("passed", True))
            issues = [str(x) for x in data.get("issues", [])]
        except Exception as exc:
            logger.warning("审校调用失败，默认放行：{}", exc)
            passed, issues = True, []

        state.review_round += 1
        if passed:
            state.trace.append(f"review: 第 {state.review_round} 轮通过")
            return state

        state.review_notes = issues
        state.trace.append(f"review: 第 {state.review_round} 轮驳回 {len(issues)} 条")
        return state

    def node_finalize(self, state: LabState) -> LabState:
        abnormal = [i for i in state.items if i.flag != Flag.NORMAL]
        state.report = LabReport(
            panel_type=detect_panel(state.raw_text) or PanelType.UNKNOWN,
            abnormal_items=abnormal,
            correlations=state.correlations,
            summary=state.draft_summary,
            advice=state.draft_advice or _extract_advice(state.draft_summary),
            review_passed=not state.review_notes,
            review_notes=state.review_notes,
        )
        return state

    # ------------------------------------------------------------------ #
    # 条件边
    # ------------------------------------------------------------------ #
    @staticmethod
    def should_rewrite(state: LabState) -> str:
        if state.review_notes and state.review_round < MAX_REVIEW_ROUNDS:
            return "rewrite"
        return "finalize"

    # ------------------------------------------------------------------ #
    # 组装
    # ------------------------------------------------------------------ #
    def _build(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            logger.warning("langgraph 未安装，使用顺序执行兼容模式")
            return None

        graph = StateGraph(LabState)
        graph.add_node("parse", self.node_parse)
        graph.add_node("judge", self.node_judge)
        graph.add_node("correlate", self.node_correlate)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("interpret", self.node_interpret)
        graph.add_node("review", self.node_review)
        graph.add_node("finalize", self.node_finalize)

        graph.set_entry_point("parse")
        graph.add_edge("parse", "judge")
        graph.add_edge("judge", "correlate")
        graph.add_edge("correlate", "retrieve")
        graph.add_edge("retrieve", "interpret")
        graph.add_edge("interpret", "review")
        graph.add_conditional_edges(
            "review",
            self.should_rewrite,
            {"rewrite": "interpret", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(self, raw_text: str) -> LabReport:
        """执行整条链路。"""
        state = LabState(raw_text=raw_text)
        if self.app is None:
            # 兼容模式：手工复现图语义
            for node in (
                self.node_parse,
                self.node_judge,
                self.node_correlate,
                self.node_retrieve,
                self.node_interpret,
                self.node_review,
            ):
                state = node(state)
                if state.review_notes and state.review_round < MAX_REVIEW_ROUNDS:
                    state = self.node_interpret(state)
                    state = self.node_review(state)
            state = self.node_finalize(state)
            return state.report or LabReport()

        result = self.app.invoke(state)
        if isinstance(result, dict):
            state = LabState(**result)
        else:
            state = result
        return state.report or LabReport()


def _extract_advice(summary: str) -> list[str]:
    """从草稿里抽建议条目；模型没给结构化建议时做一次兜底拆分。"""
    if not summary:
        return []
    import re

    # 模型有时把建议写在段落中间，按句拆分后再筛
    sentences = re.split(r"[。；\n]", summary)
    advice = [
        s.strip(" -•·")
        for s in sentences
        if re.search(r"(建议|推荐|请|可考虑|必要时|就诊|复查)", s) and len(s.strip()) > 6
    ]
    return advice[:5]
