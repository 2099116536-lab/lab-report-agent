"""混合检索：BM25（稀疏） + Chroma（稠密） → RRF 融合 → 重排。

单路向量检索在检验场景的两个坑：
1. 缩写命中差 —— "MCV" 和 "平均红细胞体积" 语义相似但字面不同，纯 BM25 漏召回；
2. 数值敏感 —— "钾 6.2 危急值处理" 和 "钾 3.0 低钾处理" 向量很近，纯向量容易召错。

所以稀疏 + 稠密双路召回，RRF 融合后再用 CrossEncoder 重排。
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"


class HybridRetriever:
    """稀疏 + 稠密混合检索器。

    依赖（chromadb / rank_bm25）缺失时自动降级为纯 BM25，
    保证在没有网络或没装依赖的机器上依旧可跑通链路。
    """

    def __init__(self, docs: list[str] | None = None, persist_dir: str | None = None) -> None:
        self.docs: list[str] = docs or self._load_default_docs()
        self.persist_dir = persist_dir or os.getenv("CHROMA_DIR", ".chroma")
        self._bm25 = None
        self._collection = None
        self._embedder = None
        self._reranker = None
        self._init_sparse()
        self._init_dense()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def _load_default_docs(self) -> list[str]:
        if not KNOWLEDGE_DIR.exists():
            logger.warning("知识库目录不存在：{}，使用内置最小知识集", KNOWLEDGE_DIR)
            return self._builtin_docs()
        docs: list[str] = []
        for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
            for chunk in self.chunk_document(path.read_text(encoding="utf-8")):
                docs.append(chunk)
        return docs

    @staticmethod
    def _builtin_docs() -> list[str]:
        return [
            "血红蛋白(HGB)降低：成年男性<130 g/L、女性<115 g/L 为贫血。结合 MCV 可区分小细胞性、正细胞性、大细胞性贫血。",
            "平均红细胞体积(MCV)降低伴血红蛋白降低，提示小细胞低色素性贫血，最常见原因为缺铁性贫血，建议检测血清铁、铁蛋白、总铁结合力。",
            "MCV 升高伴血红蛋白降低，提示大细胞性贫血，常见于叶酸或维生素 B12 缺乏，建议检测血清叶酸、维生素 B12。",
            "白细胞(WBC)与中性粒细胞百分比同步升高，典型提示细菌感染，建议结合 CRP、PCT 及临床症状综合判断。",
            "淋巴细胞百分比升高伴白细胞升高，常见于病毒感染，必要时行外周血涂片及病毒血清学检查。",
            "嗜酸性粒细胞百分比升高，提示过敏性疾病或寄生虫感染可能。",
            "ALT/AST 升高提示肝细胞损伤。AST/ALT 比值 >1 倾向酒精性肝病或肝纤维化，<1 更倾向病毒性肝炎或脂肪肝。",
            "GGT 与 ALP 同步升高伴总胆红素升高，提示胆汁淤积或胆道梗阻，建议腹部超声或 MRCP 检查。",
            "血清肌酐升高提示肾功能受损，可用 CKD-EPI 公式估算 eGFR 进行分期，建议肾内科随诊。",
            "血尿酸升高为高尿酸血症，与痛风、代谢综合征、心血管疾病风险相关，建议低嘌呤饮食并定期监测。",
            "空腹血糖升高伴甘油三酯升高、HDL-C 降低，符合代谢综合征表现，建议内分泌科评估。",
            "血小板显著降低(<50×10^9/L)有出血风险，显著升高需排查骨髓增殖性疾病。",
            "血钾 >6.5 mmol/L 或 <2.5 mmol/L 属危急值，需立即临床干预并复查确认。",
        ]

    @staticmethod
    def chunk_document(text: str, size: int = 200, overlap: int = 20) -> list[str]:
        """按字符滑窗切分。中文按字符切比按 token 切更稳，且便于控制 overlap。"""
        chunks: list[str] = []
        step = max(size - overlap, 1)
        for start in range(0, len(text), step):
            piece = text[start : start + size].strip()
            if len(piece) > 30:  # 丢弃过碎片段，减少噪声
                chunks.append(piece)
        return chunks

    def _init_sparse(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 未安装，跳过稀疏检索")
            return
        tokenized = [self._tokenize(doc) for doc in self.docs]
        self._bm25 = BM25Okapi(tokenized)

    def _init_dense(self) -> None:
        try:
            import chromadb
        except ImportError:
            logger.warning("chromadb 未安装，跳过稠密检索")
            return
        try:
            client = chromadb.Client(chromadb.Settings(anonymized_telemetry=False))
            self._collection = client.get_or_create_collection("lab_knowledge")
            if self._collection.count() == 0 and self.docs:
                self._collection.add(
                    documents=self.docs,
                    ids=[f"doc_{i}" for i in range(len(self.docs))],
                )
        except Exception as exc:
            logger.warning("Chroma 初始化失败，降级为纯 BM25：{}", exc)
            self._collection = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中英混排分词：英文按词，中文按字 + 二字组。"""
        import re

        tokens: list[str] = []
        for eng in re.findall(r"[A-Za-z][A-Za-z\-]*", text):
            tokens.append(eng.lower())
        chinese = re.sub(r"[^一-龥]", "", text)
        tokens.extend(list(chinese))
        tokens.extend(chinese[i : i + 2] for i in range(len(chinese) - 1))
        return tokens

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5) -> list[str]:
        sparse_ids = self._sparse_search(query, top_k * 3)
        dense_ids = self._dense_search(query, top_k * 3)
        if not sparse_ids and not dense_ids:
            # 两路都不可用（依赖缺失 / 索引为空）时退回词面匹配，保证链路不断
            return self._lexical_search(query, top_k)
        fused = self._rrf([sparse_ids, dense_ids])[: top_k * 2]
        return self._rerank(query, fused, top_k)

    def _lexical_search(self, query: str, top_k: int) -> list[str]:
        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return self.docs[:top_k]
        scored = [
            (i, len(q_tokens & set(self._tokenize(doc))))
            for i, doc in enumerate(self.docs)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [self.docs[i] for i, score in scored[:top_k] if score > 0] or self.docs[:top_k]

    def _sparse_search(self, query: str, top_k: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:top_k]

    def _dense_search(self, query: str, top_k: int) -> list[int]:
        if self._collection is None:
            return []
        try:
            res = self._collection.query(query_texts=[query], n_results=top_k)
        except Exception as exc:
            logger.warning("向量检索失败：{}", exc)
            return []
        ids = res.get("ids", [[]])[0]
        return [int(i.split("_")[1]) for i in ids if i.startswith("doc_")]

    @staticmethod
    def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion：无需调权重的融合方式，对两路分数尺度不敏感。"""
        scores: dict[int, float] = {}
        for ranks in rank_lists:
            for position, doc_id in enumerate(ranks):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    def _rerank(self, query: str, doc_ids: list[int], top_k: int) -> list[str]:
        """重排：优先用 CrossEncoder，不可用时用启发式打分兜底。"""
        if not doc_ids:
            return []
        if self._reranker is None:
            self._reranker = self._load_reranker()
        if self._reranker is not None:
            try:
                pairs = [(query, self.docs[i]) for i in doc_ids]
                scores = self._reranker.predict(pairs)
                ranked = sorted(zip(doc_ids, scores), key=lambda t: t[1], reverse=True)
                return [self.docs[i] for i, _ in ranked[:top_k]]
            except Exception as exc:
                logger.warning("重排失败，回落启发式排序：{}", exc)

        # 启发式兜底：查询词在文档中的覆盖度
        q_tokens = set(self._tokenize(query))
        scored = []
        for i in doc_ids:
            overlap = len(q_tokens & set(self._tokenize(self.docs[i])))
            scored.append((i, overlap))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [self.docs[i] for i, _ in scored[:top_k]]

    @staticmethod
    def _load_reranker():
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder("BAAI/bge-reranker-base")
        except Exception:
            logger.debug("CrossEncoder 不可用，使用启发式重排")
            return None
