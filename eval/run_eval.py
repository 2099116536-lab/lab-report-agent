"""跑回归集，算指标。

指标定义：
- 异常项识别：以「项目编码集合」比对，micro Precision / Recall / F1；
  衡量解析 + 参考区间判定链路是否正确（标签由区间表生成，防的是代码退化）。
- 关联模式命中：期望模式集合与实际推出的模式集合完全一致才算命中。
- 审校通过率：review 节点放行的比例。

    python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import LabReportGraph  # noqa: E402
from src.llm import MockLLM  # noqa: E402
from src.rag import HybridRetriever  # noqa: E402


def main() -> None:
    labels = [json.loads(l) for l in (Path(__file__).parent / "labels.jsonl").open(encoding="utf-8")]
    graph = LabReportGraph(llm=MockLLM(), retriever=HybridRetriever())

    tp = fp = fn = 0
    pat_hit = pat_total = 0
    passed = 0
    latencies: list[float] = []
    rows: list[dict] = []

    for case in labels:
        t0 = time.perf_counter()
        rep = graph.run(case["text"])
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        pred = {i.code for i in rep.abnormal_items if i.code}
        exp = set(case["expect_abnormal"])
        tp += len(pred & exp)
        fp += len(pred - exp)
        fn += len(exp - pred)

        pred_pats = {c.pattern for c in rep.correlations}
        exp_pats = set(case["expect_patterns"])
        if exp_pats:
            pat_total += 1
            if exp_pats <= pred_pats:
                pat_hit += 1
        if rep.review_passed:
            passed += 1

        rows.append({
            "id": case["id"],
            "miss": sorted(exp - pred),
            "extra": sorted(pred - exp),
            "pat_exp": sorted(exp_pats),
            "pat_pred": sorted(pred_pats),
            "ms": round(dt, 1),
            "review": rep.review_passed,
        })

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    latencies.sort()

    result = {
        "cases": len(labels),
        "abnormal_precision": round(prec, 3),
        "abnormal_recall": round(rec, 3),
        "abnormal_f1": round(f1, 3),
        "pattern_hit": f"{pat_hit}/{pat_total}",
        "pattern_hit_rate": round(pat_hit / pat_total, 3) if pat_total else 0.0,
        "review_pass_rate": round(passed / len(labels), 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1),
        "median_ms": round(latencies[len(latencies) // 2], 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    bad = [r for r in rows if r["miss"] or r["extra"] or not set(r["pat_exp"]) <= set(r["pat_pred"])]
    if bad:
        print("\n未完全命中的用例：")
        for r in bad:
            print(f"  {r['id']}  漏={r['miss']}  多={r['extra']}  模式 期望={r['pat_exp']} 实际={r['pat_pred']}")

    (Path(__file__).parent / "results.json").write_text(
        json.dumps({"summary": result, "detail": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
