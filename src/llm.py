"""统一 LLM 接口层。

设计目标：
1. 业务代码只依赖 ``BaseLLM.chat``，切换厂商不改上层逻辑；
2. 提供 ``MockLLM`` —— 无 API Key 时也能端到端跑通整条链路，便于演示与 CI。
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger


class BaseLLM(ABC):
    """所有模型实现的公共接口。"""

    name: str = "base"

    @abstractmethod
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        """返回纯文本响应。"""

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> dict[str, Any]:
        """要求模型返回 JSON，并做一次容错解析。

        真实场景里模型经常在 JSON 外面裹 ```` ```json ```` 或加前后废话，
        这里统一处理，避免上层反复写解析逻辑。
        """
        raw = self.chat(system, user, temperature)
        cleaned = self._strip_fence(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 退一步：抓第一个 {...} 块
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                logger.error("模型未返回合法 JSON: {}", raw[:300])
                raise ValueError(f"模型输出无法解析为 JSON: {raw[:200]}")
            return json.loads(match.group(0))

    @staticmethod
    def _strip_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容接口：OpenAI / 通义千问 / DeepSeek / 本地 vLLM 均可接入。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # 延迟导入，Mock 模式下不需要装

        self.model = model or os.getenv("LLM_MODEL", "qwen-plus")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.name = self.model

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class MockLLM(BaseLLM):
    """离线 Mock 模型。

    不联网、不花钱，用于：
    - 演示整条 LangGraph 链路；
    - 单元测试 / CI；
    - 在没有 Key 的机器上排查非模型问题。
    """

    name = "mock-offline"

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        # 依据 system 提示判断调用方需要什么形态的返回
        if '"items"' in system or "结构化抽取" in system or "异常指标" in system:
            return self._mock_parse(user)
        if '"summary"' in system or "通俗" in system:
            return self._mock_interpret(user)
        if "关联" in system or "correlation" in system:
            return json.dumps({"correlations": []}, ensure_ascii=False)
        if "审校" in system or "review" in system:
            return json.dumps({"passed": True, "issues": []}, ensure_ascii=False)
        return json.dumps({"result": "mock response"}, ensure_ascii=False)

    def _mock_parse(self, user: str) -> str:
        """从输入里用正则抠出 `名称 数值 单位` 三元组，作为解析结果。"""
        items: list[dict[str, Any]] = []
        pattern = re.compile(
            r"(?P<name>[一-龥A-Za-z\*\(\)％%]+)\s*"
            r"(?P<value>[<>=≥≤]*\s*\d+\.?\d*)\s*"
            r"(?P<unit>[^\s\d]*)"
        )
        for m in pattern.finditer(user):
            name = m.group("name").strip()
            raw_value = m.group("value").replace(" ", "")
            if not name or len(name) < 2:
                continue
            try:
                value = float(re.sub(r"[<>=≥≤]", "", raw_value))
            except ValueError:
                continue
            items.append(
                {
                    "name": name,
                    "value": value,
                    "unit": m.group("unit").strip() or None,
                    "qualitative": None,
                }
            )
        return json.dumps({"items": items[:40]}, ensure_ascii=False)

    def _mock_interpret(self, user: str) -> str:
        """按解读节点的入参拼一段结构化白话解读（离线演示用）。"""
        import json as _json

        try:
            payload = _json.loads(user)
        except _json.JSONDecodeError:
            payload = {"abnormal": [], "correlations": [], "critical": []}

        abnormal = payload.get("abnormal", [])
        correlations = payload.get("correlations", [])
        critical = payload.get("critical", [])

        if not abnormal:
            return _json.dumps(
                {"summary": "本次检验各项指标均在参考范围内，未见明显异常。", "advice": []},
                ensure_ascii=False,
            )

        head = "⚠️ 本次检验存在需要立即就医的危急值，请尽快就诊。\n" if critical else ""
        lines = [head + f"本次共有 {len(abnormal)} 项指标超出参考范围："]
        for item in abnormal[:8]:
            arrow = "↑" if item.get("flag") == "high" else "↓"
            ref = item.get("ref") or "-"
            lines.append(
                f"- {item.get('name')} {item.get('value')}{item.get('unit') or ''}{arrow}"
                f"（参考 {ref}），程度：{item.get('severity')}"
            )

        for corr in correlations[:3]:
            lines.append(f"\n综合判断：{corr.get('pattern')} —— {corr.get('note', '')}")

        advice = [
            f"建议复查 {corr.get('pattern', '相关指标')} 对应的确诊项目" if correlations else "建议 1-2 周后复查异常指标",
            "携带本报告至相应专科门诊就诊",
            "解读结果仅供辅助参考，不作为诊断依据",
        ]
        return _json.dumps(
            {"summary": "\n".join(lines), "advice": advice}, ensure_ascii=False
        )


def get_llm(force_mock: bool = False) -> BaseLLM:
    """工厂方法：有 Key 用真模型，没 Key 自动降级到 Mock。"""
    if force_mock:
        return MockLLM()
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAICompatLLM()
        except Exception as exc:  # 依赖缺失时降级，保证链路可跑
            logger.warning("初始化远程模型失败，降级到 Mock：{}", exc)
            return MockLLM()
    logger.warning("未检测到 OPENAI_API_KEY，使用离线 Mock 模型")
    return MockLLM()
