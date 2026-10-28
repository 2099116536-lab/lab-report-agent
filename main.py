"""CLI 入口。

    python main.py --input data/samples/cbc_001.txt           # 自动选模型
    python main.py --input data/samples/cbc_001.txt --mock    # 强制离线 Mock
    python main.py --input data/samples/cbc_001.txt --trace   # 打印各节点 trace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from src.graph import LabReportGraph
from src.llm import MockLLM, get_llm
from src.rag import HybridRetriever

SAMPLE = """性别：男  年龄：34
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


def main() -> int:
    parser = argparse.ArgumentParser(description="检验报告智能解读 Agent")
    parser.add_argument("--input", type=str, help="检验单文本文件路径；不给则用内置样例")
    parser.add_argument("--mock", action="store_true", help="强制使用离线 Mock 模型")
    parser.add_argument("--trace", action="store_true", help="打印节点执行轨迹")
    args = parser.parse_args()

    if args.input:
        path = Path(args.input)
        if not path.exists():
            logger.error("文件不存在：{}", path)
            return 2
        text = path.read_text(encoding="utf-8")
    else:
        text = SAMPLE
        logger.info("未指定 --input，使用内置血常规样例")

    llm = MockLLM() if args.mock else get_llm()
    graph = LabReportGraph(llm=llm, retriever=HybridRetriever())
    logger.info("模型：{}", graph.llm.name)

    report = graph.run(text)

    if args.trace:
        print("\n--- trace ---")
        print("\n".join(graph_trace(graph, text)))

    print("\n--- report ---")
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def graph_trace(graph: LabReportGraph, text: str) -> list[str]:
    """单独跑一遍只为拿到 trace（演示用，生产应通过 callback 收集）。"""
    from src.models import LabState

    state = LabState(raw_text=text)
    for node in (
        graph.node_parse,
        graph.node_judge,
        graph.node_correlate,
        graph.node_retrieve,
        graph.node_interpret,
        graph.node_review,
        graph.node_finalize,
    ):
        state = node(state)
    return state.trace


if __name__ == "__main__":
    sys.exit(main())
