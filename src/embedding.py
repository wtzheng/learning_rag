import logging
import os

import numpy as np

from src.config import EMBED_MODEL, EMBED_NORMALIZE, HF_ENDPOINT

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self.model_name = model_name or EMBED_MODEL
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s (device=%s)", self.model_name, self.device)
        if HF_ENDPOINT:
            os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded. Dim=%d", self.dim)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(texts, normalize_embeddings=EMBED_NORMALIZE)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @property
    def dim(self) -> int:
        self._load()
        # sentence-transformers >=4.0 renamed this method
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()
