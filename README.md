# Lab Report Agent · 检验报告智能解读与辅助分析 Agent

> 面向医学检验场景的垂直领域多智能体系统。输入血常规 / 生化 / 免疫 / 尿常规等检验单，自动完成
> **指标解析 → 参考区间比对与异常分级 → 医学知识检索（RAG）→ 通俗化解读生成 → 安全审校**，
> 输出结构化 JSON + 面向患者的白话报告。

一个检验科出身的人做的 Agent 项目 —— 领域知识不是从网上扒的，是本体专业。

---

## 为什么做这个

通用大模型解读检验单有三个硬伤：

1. **参考区间分层缺失** —— 同一个指标，男女、成人儿童的区间不同，模型经常张冠李戴；
2. **孤立看指标** —— 只看单个箭头，不会做指标间的关联推理（如 `HGB↓ + MCV↓ + RDW↑` 指向小细胞低色素性贫血）；
3. **幻觉与过度诊疗建议** —— 直接给"你得了 XX 病"，医疗场景下不可接受。

本项目用**确定性的规则引擎兜底 + LLM 负责表达**，把"异常判定"这类不容出错的环节从模型手里拿回来。

---

## 架构

```
                    ┌──────────────┐
   检验单(文本/PDF) ─▶  parse       │  OCR 文本 → 结构化 LabPanel
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  flag        │  规则引擎：参考区间比对 + 性别/年龄分层
                    └──────┬───────┘  异常分级 normal / borderline / critical
                           ▼
                    ┌──────────────┐
                    │  correlate   │  指标关联推理：交叉解读模式匹配
                    └──────┬───────┘  (贫血形态学 / 肝胆 / 肾功能 / 糖脂代谢)
                           ▼
                    ┌──────────────┐
                    │  retrieve    │  RAG：BM25 + 向量混合检索 → Rerank
                    └──────┬───────┘  召回检验项目临床意义知识条目
                           ▼
                    ┌──────────────┐
                    │  interpret   │  LLM 生成通俗解读 + 就医建议
                    └──────┬───────┘  Pydantic 结构化输出校验
                           ▼
                    ┌──────────────┐
                    │  review      │  安全审校 Agent：幻觉/越界/过度诊疗拦截
                    └──────┬───────┘  不通过则回退重写（最多 2 轮）
                           ▼
                      结构化报告
```

基于 **LangGraph** 的有状态图实现，节点间通过 `LabState` 传递，关键节点带
`human-in-the-loop` 检查点与重放能力。

---

## 技术要点

| 方向 | 做法 |
| --- | --- |
| 工作流编排 | LangGraph 状态图，条件边实现"审校不通过 → 回退重写"循环 |
| 检索 | BM25（稀疏）+ Chroma 向量（稠密）混合召回，RRF 融合，CrossEncoder 重排 |
| 结构化输出 | Pydantic v2 模型约束 + 输出校验重试，杜绝解析失败 |
| 工具调用 | OpenAI Function Calling 注册 `query_reference_range` / `calc_eGFR` / `search_guideline` |
| 服务化 | FastAPI + SSE 流式输出，打字机效果；`/report` 全量 JSON 接口 |
| 可观测 | LangChain Callback 埋点，记录每节点 token 消耗与耗时 |
| 评估 | 内置 40 例标注集，指标：异常识别 F1、解读召回率、审校拦截率 |
| 部署 | Docker 多阶段构建，镜像 < 400MB |

---

## 快速开始

```bash
git clone https://github.com/<your-id>/lab-report-agent.git
cd lab-report-agent
pip install -r requirements.txt

# 无 API Key 也能跑：内置离线 Mock 模型，走完整链路
python main.py --input data/samples/cbc_001.txt --mock

# 真实模型
export OPENAI_API_KEY=sk-xxx
python main.py --input data/samples/cbc_001.txt

# 启动服务（FastAPI + SSE 流式）
uvicorn src.server:app --port 8000
```

样例输出：

```json
{
  "panel_type": "血常规",
  "abnormal_items": [
    {"code": "HGB", "name": "血红蛋白", "value": 98, "unit": "g/L",
     "flag": "low", "severity": "moderate", "ref": "130-175"}
  ],
  "correlations": [
    {"pattern": "小细胞低色素性贫血",
     "evidence": ["HGB↓", "MCV↓", "MCH↓", "RDW↑"],
     "confidence": 0.86}
  ],
  "summary": "血红蛋白 98 g/L 低于成年男性参考区间（130-175），结合 MCV、MCH 同步降低，呈小细胞低色素改变，常见于缺铁性贫血……",
  "advice": ["建议复查铁代谢（血清铁、铁蛋白）", "消化内科就诊排查失血原因"],
  "disclaimer": "本报告为辅助解读，不构成诊断依据。"
}
```

---

## 目录结构

```
lab-report-agent/
├── README.md
├── requirements.txt
├── main.py                 # CLI 入口
├── src/
│   ├── llm.py              # 统一 LLM 接口（OpenAI / 通义 / 离线 Mock）
│   ├── models.py           # Pydantic 数据模型 + 检验项目参考区间库
│   ├── parser.py           # 检验单文本解析（正则 + LLM 兜底）
│   ├── rules.py            # 异常判定规则引擎 + 指标关联推理
│   ├── rag.py              # 混合检索 + Rerank
│   ├── graph.py            # LangGraph 状态图定义
│   ├── tools.py            # Function Calling 工具集
│   └── server.py           # FastAPI + SSE
├── data/
│   ├── knowledge/          # 检验项目临床意义知识库（Markdown）
│   └── samples/            # 样例检验单
├── eval/
│   ├── eval_set.jsonl      # 40 例标注评测集
│   └── run_eval.py         # 评测脚本
└── tests/
```

---

## 评估基线

在 40 例内部标注集上（离线 Mock 模型为下限参考，真实模型数据见 `eval/results.md`）：

| 模型 | 异常识别 F1 | 关联推理准确率 | 审校拦截率 | 端到端 P95 延迟 |
| --- | --- | --- | --- | --- |
| 规则引擎 only | 0.94 | 0.51 | — | 0.3 s |
| + RAG（无重排） | 0.94 | 0.68 | 0.71 | 6.2 s |
| + Rerank（本项目） | 0.95 | 0.83 | 0.89 | 7.8 s |

引入重排序后关联推理准确率 **+15 个百分点**，审校节点拦截掉 11% 的越界表述。

---

## 说明

- 知识库条目整理自公开检验医学教材与检验项目临床意义资料，仅用于工程演示；
- 参考区间采用成人通用区间，实际临床以检测机构报告标注为准；
- **本项目输出不构成任何诊断或治疗建议。**
