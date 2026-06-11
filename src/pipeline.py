"""
Pipeline orchestration: ties together loading, chunking, embedding, storage,
retrieval, and LLM querying into a single index() / query() interface.
"""

import logging
from typing import Optional

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY
from src.config import TOP_K
from src.loader import load_directory
from src.chunking import get_chunker, Chunk
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.retrieval import Retriever, BM25Index
from src.llm import LLM
from src.prompts import build_prompt

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: index documents and answer queries."""

    def __init__(self):
        self.embed_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.retriever = Retriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
        )
        self.llm = LLM()
        self._indexed_chunks: list[Chunk] = []

    # ── Indexing ──

    def index(self, data_dir: str = "data/raw") -> int:
        """
        Load documents from *data_dir*, chunk, embed, and store in Chroma.

        Returns:
            Number of chunks indexed.
        """
        logger.info("=== Indexing pipeline started ===")

        # 1. Load
        documents = load_directory(data_dir)
        if not documents:
            logger.warning("No documents found in '%s'. Nothing to index.", data_dir)
            return 0

        # 2. Chunk
        chunker = get_chunker(
            CHUNK_STRATEGY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = chunker.chunk(doc.text, metadata=doc.metadata)
            all_chunks.extend(chunks)

        logger.info("Created %d chunks from %d documents", len(all_chunks), len(documents))
        self._indexed_chunks = all_chunks

        # 3. Embed
        texts = [c.text for c in all_chunks]
        embeddings = self.embed_model.embed_texts(texts)

        # 4. Store (vectors)
        self.vector_store.clear()
        self.vector_store.add_chunks(all_chunks, embeddings)

        # 5. Build BM25 index (sparse)
        self.bm25_index.build(all_chunks)

        logger.info("=== Indexing complete: %d chunks indexed ===", len(all_chunks))
        return len(all_chunks)

    # ── Querying ──

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        template_name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            template_name: Prompt template to use (auto-detect if None).
            provider: LLM provider (default from config).

        Returns:
            Dict with keys: question, answer, sources (chunks used).
        """
        logger.info("=== Query: %s ===", question)

        k = top_k or TOP_K

        # 1. Embed query
        query_embedding = self.embed_model.embed_text(question)

        # 2. Retrieve
        results = self.retriever.retrieve(question, query_embedding, top_k=k)

        # 3. Assemble context
        context_parts = []
        sources = []
        for res in results:
            source = res["metadata"].get("source", "unknown")
            context_parts.append(f"[来源: {source}]\n{res['text']}")
            sources.append({
                "source": source,
                "score": round(res["score"], 4),
                "text_preview": res["text"][:200],
            })

        context = "\n\n".join(context_parts)

        # 4. Build prompt
        prompt = build_prompt(question, context, template_name=template_name)

        # 5. Query LLM
        answer = self.llm.query(prompt, provider=provider)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
