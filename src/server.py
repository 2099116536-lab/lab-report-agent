"""FastAPI 服务：全量 JSON 接口 + SSE 流式接口。

流式用 SSE 而非 WebSocket —— Agent 场景是单向的"服务端持续吐字"，
SSE 走普通 HTTP，穿透性好、实现简单、断线自动重连。
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .graph import LabReportGraph
from .llm import get_llm
from .rag import HybridRetriever

app = FastAPI(title="Lab Report Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_GRAPH: LabReportGraph | None = None


def get_graph() -> LabReportGraph:
    """懒加载：模型与向量库初始化较重，放到首次请求。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = LabReportGraph(llm=get_llm(), retriever=HybridRetriever())
    return _GRAPH


class ReportRequest(BaseModel):
    text: str = Field(..., description="检验单原文")
    mock: bool = False


class HealthResponse(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    graph = get_graph()
    return HealthResponse(status="ok", model=graph.llm.name)


@app.post("/report")
async def create_report(req: ReportRequest):
    """全量报告接口，返回结构化 JSON。"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="检验单内容为空")
    graph = get_graph()
    started = time.perf_counter()
    report = graph.run(req.text)
    elapsed = time.perf_counter() - started
    return {
        "data": report.model_dump(mode="json"),
        "elapsed_seconds": round(elapsed, 2),
        "model": graph.llm.name,
    }


@app.post("/report/stream")
async def stream_report(req: ReportRequest):
    """SSE 流式接口：先推各节点 trace，再逐字推送解读正文。"""

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        graph = get_graph()
        try:
            report = graph.run(req.text)
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)}, ensure_ascii=False)}
            return

        yield {"event": "meta", "data": json.dumps({"panel": report.panel_type.value}, ensure_ascii=False)}

        for item in report.abnormal_items:
            payload = {
                "name": item.name,
                "value": item.value,
                "unit": item.unit,
                "flag": item.flag.value,
                "severity": item.severity.value,
                "ref": item.ref,
            }
            yield {"event": "item", "data": json.dumps(payload, ensure_ascii=False)}

        for corr in report.correlations:
            yield {"event": "correlation", "data": corr.model_dump_json()}

        # 逐字推送，模拟打字机效果（真实场景替换为模型 streaming 回调）
        for char in report.summary:
            yield {"event": "token", "data": json.dumps({"t": char}, ensure_ascii=False)}

        yield {
            "event": "done",
            "data": json.dumps(
                {"advice": report.advice, "review_passed": report.review_passed},
                ensure_ascii=False,
            ),
        }

    return EventSourceResponse(event_generator())
