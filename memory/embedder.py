"""
memory/embedder.py

Local semantic embeddings via sentence-transformers.

Configured via environment variables:
  EMBED_MODEL — HuggingFace model ID (default: google/embeddinggemma-300m)
                Set to empty string to disable semantic memory entirely.

The model is lazy-loaded on first use to avoid startup cost.
encode_query() is used for user messages; encode_document() for context entries.
EmbeddingGemma-300M does not support float16 — runs in float32.
"""

import asyncio
import logging
import math
import os
from functools import cached_property

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "google/embeddinggemma-300m"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class LocalEmbedder:
    """
    Async wrapper around a local sentence-transformers model.

    Uses encode_query() for user messages and encode_document() for context
    entries, matching EmbeddingGemma's task-specific prompt design.

    The underlying SentenceTransformer is loaded lazily on first call to
    avoid blocking startup. All encoding runs in a thread-pool executor so
    it does not block the async event loop.

    Usage:
        embedder = LocalEmbedder()
        vec  = await embedder.embed_query("what is the user's name?")
        vecs = await embedder.embed_documents(["Alice lives in Berlin", ...])
    """

    def __init__(self) -> None:
        self._model_name = os.getenv("EMBED_MODEL", _DEFAULT_MODEL).strip()
        self._enabled = bool(self._model_name)
        self._model = None  # lazy-loaded

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _load_model(self):
        """Load the SentenceTransformer model (called once, in executor)."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"LocalEmbedder: loading model '{self._model_name}' …")
            self._model = SentenceTransformer(self._model_name)
            logger.info("LocalEmbedder: model ready")
        except Exception as exc:
            logger.warning(f"LocalEmbedder: could not load model ({exc}); semantic memory disabled")
            self._enabled = False
            self._model = None
        return self._model

    async def embed_query(self, text: str) -> list[float] | None:
        """
        Embed a single query (user message) using the model's query encoder.

        Returns a float vector, or None if the embedder is unavailable.
        """
        if not self._enabled:
            return None
        try:
            loop = asyncio.get_event_loop()
            vec = await loop.run_in_executor(None, self._encode_query_sync, text)
            return vec
        except Exception as exc:
            logger.debug(f"LocalEmbedder.embed_query failed ({exc}); falling back to keyword scoring")
            return None

    async def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        """
        Embed a batch of documents (context entries) using the model's document encoder.

        Returns a list of float vectors (one per text), or None on failure.
        """
        if not self._enabled or not texts:
            return None
        try:
            loop = asyncio.get_event_loop()
            vecs = await loop.run_in_executor(None, self._encode_documents_sync, texts)
            return vecs
        except Exception as exc:
            logger.debug(f"LocalEmbedder.embed_documents failed ({exc}); falling back to keyword scoring")
            return None

    # ------------------------------------------------------------------
    # Sync helpers (run in executor)
    # ------------------------------------------------------------------

    def _encode_query_sync(self, text: str) -> list[float] | None:
        model = self._load_model()
        if model is None:
            return None
        vec = model.encode_query(text)
        return vec.tolist()

    def _encode_documents_sync(self, texts: list[str]) -> list[list[float]] | None:
        model = self._load_model()
        if model is None:
            return None
        vecs = model.encode_document(texts)
        return [v.tolist() for v in vecs]
