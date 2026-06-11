"""
Retrieval module: dense (vector), sparse (BM25), hybrid, and optional reranker.
"""

import logging
import math
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import TOP_K, RETRIEVAL_MODE, HYBRID_ALPHA, RRF_K
from src.config import RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_K
from src.chunking import Chunk
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BM25Index:
    """In-memory BM25 index built over chunk texts."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: list[Chunk] = []
        self.tokenized: list[list[str]] = []

    def build(self, chunks: list[Chunk]):
        """Build the BM25 index from a list of Chunks."""
        self.chunks = chunks
        self.tokenized = [list(jieba.cut(c.text)) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        logger.info("BM25 index built with %d documents", len(chunks))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search BM25 index, return top-k results with scores."""
        if self.bm25 is None:
            logger.warning("BM25 index not built yet — returning empty results.")
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "index": idx,
            })
        return results


class Retriever:
    """Main retriever supporting dense, hybrid, and optionally reranked search."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: Optional[BM25Index] = None,
        mode: str = RETRIEVAL_MODE,
        top_k: int = TOP_K,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.mode = mode
        self.top_k = top_k
        self._reranker = None

    # ── Dense retrieval ──

    def dense_search(self, query_embedding: list[float], top_k: Optional[int] = None) -> list[dict]:
        """Pure vector similarity search."""
        k = top_k or self.top_k
        return self.vector_store.similarity_search(query_embedding, top_k=k)

    # ── Hybrid retrieval ──

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: Optional[int] = None,
        alpha: float = HYBRID_ALPHA,
    ) -> list[dict]:
        """
        Fuse dense + sparse results via Reciprocal Rank Fusion (RRF).

        Args:
            query_embedding:  Vector for dense search.
            query_text:       Original text for BM25 search.
            top_k:            Number of final results.
            alpha:            Dense weight (0.0 = pure BM25, 1.0 = pure dense).

        Returns:
            List of dicts with keys: id, text, metadata, score (fused).
        """
        k = top_k or self.top_k
        dense_k = k * 4  # retrieve more candidates for fusion

        # Dense results
        dense_results = self.vector_store.similarity_search(query_embedding, top_k=dense_k)

        # Sparse results
        sparse_results = []
        if self.bm25_index is not None:
            sparse_results = self.bm25_index.search(query_text, top_k=dense_k)

        # RRF fusion
        if not sparse_results:
            return dense_results[:k]

        fused_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, res in enumerate(dense_results):
            doc_id = res["id"]
            fused_scores[doc_id] = alpha * (1.0 / (rank + RRF_K))
            result_map[doc_id] = res

        for rank, res in enumerate(sparse_results):
            chunk = res["chunk"]
            doc_id = f"{chunk.metadata.get('source', 'doc')}_{chunk.index}"
            rrf_score = (1.0 - alpha) * (1.0 / (rank + RRF_K))
            if doc_id in fused_scores:
                fused_scores[doc_id] += rrf_score
            else:
                fused_scores[doc_id] = rrf_score
                result_map[doc_id] = {
                    "id": doc_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": 0.0,
                }

        # Sort by fused score
        ranked = sorted(fused_scores.items(), key=lambda x: -x[1])
        results = []
        for doc_id, score in ranked:
            entry = result_map[doc_id]
            entry["score"] = score
            results.append(entry)
            if len(results) >= k:
                break

        return results

    # ── Rerank ──

    def _ensure_reranker(self):
        if self._reranker is not None:
            return
        if not RERANK_ENABLED:
            return
        try:
            from llama_index.postprocessor import SentenceTransformerRerank
            self._reranker = SentenceTransformerRerank(
                model=RERANK_MODEL,
                top_k=RERANK_TOP_K,
            )
            logger.info("Reranker loaded: %s", RERANK_MODEL)
        except Exception as e:
            logger.warning("Failed to load reranker (%s). Reranking disabled.", e)
            self._reranker = None

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank results using a cross-encoder model."""
        self._ensure_reranker()
        if self._reranker is None:
            return results

        # Convert to llama_index NodeWithScore format
        from llama_index.core.schema import TextNode, NodeWithScore

        nodes = []
        for res in results:
            node = TextNode(
                text=res["text"],
                metadata=res.get("metadata", {}),
            )
            nodes.append(NodeWithScore(node=node, score=res.get("score", 0.0)))

        reranked = self._reranker.postprocess_nodes(nodes, query_str=query)
        output = []
        for rn in reranked:
            output.append({
                "id": rn.node.node_id,
                "text": rn.node.text,
                "metadata": rn.node.metadata,
                "score": rn.score if rn.score is not None else 0.0,
            })
        return output

    # ── Unified retrieve ──

    def retrieve(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Unified retrieve: dispatches to dense, hybrid, or hybrid+rerank."""
        k = top_k or self.top_k

        if self.mode == "dense":
            results = self.dense_search(query_embedding, top_k=k)
        elif self.mode == "hybrid":
            results = self.hybrid_search(query_embedding, query_text, top_k=k)
        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")

        if RERANK_ENABLED and results:
            results = self.rerank(query_text, results)

        return results
