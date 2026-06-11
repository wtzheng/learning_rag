"""
Chroma vector store wrapper.

Handles persistence, adding chunks, similarity search, and clearing.
"""

import logging
from pathlib import Path

import chromadb
import chromadb.errors
from chromadb.config import Settings

from src.config import CHROMA_PATH, COLLECTION_NAME
from src.chunking import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Chroma-based vector store with a simplified interface."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = persist_dir or CHROMA_PATH
        self.collection_name = collection_name or COLLECTION_NAME
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ── Client initialisation ──

    def _ensure_client(self):
        if self._client is not None:
            return
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Lazy-initialised Chroma collection."""
        self._ensure_client()
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Write ──

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add chunks with their embeddings to the store.

        Returns the list of generated Chroma IDs.
        """
        if not chunks:
            return []

        ids = [f"{chunk.metadata.get('source', 'doc')}_{chunk.index}" for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to collection '%s'", len(chunks), self.collection_name)
        return ids

    # ── Read ──

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Return top-k results with text, metadata, and score."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        if not results["ids"]:
            return output

        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
            })
        return output

    # ── Lifecycle ──

    def clear(self):
        """Delete the entire collection (drops all data and rebuilds)."""
        self._ensure_client()
        try:
            self._client.delete_collection(self.collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
        self._collection = None
        logger.info("Cleared collection '%s'", self.collection_name)

    def count(self) -> int:
        """Number of chunks currently indexed."""
        return self.collection.count()
