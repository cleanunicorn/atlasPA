"""
memory/embedding_cache.py

SHA-256-keyed persistent cache for document embeddings.

Stores embeddings in memory/embeddings.json so that restarting the agent
does not require re-embedding every context entry.

Only document embeddings are cached (queries are ephemeral).
"""

import hashlib
import json
import logging

from memory.embedder import LocalEmbedder
from paths import MEMORY_DIR

logger = logging.getLogger(__name__)

_CACHE_FILE = MEMORY_DIR / "embeddings.json"


class EmbeddingCache:
    """
    Persistent document-embedding cache backed by a JSON file.

    Keys are SHA-256 hashes of the input text so that identical content
    is never re-embedded across restarts.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_compute_batch(
        self, texts: list[str], embedder: LocalEmbedder
    ) -> list[list[float] | None]:
        """
        Return cached document embeddings for *texts*, computing missing ones.

        Only calls the embedder for texts not already cached, then persists
        in a single write.
        """
        keys = [_sha256(t) for t in texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in self._store]

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            vecs = await embedder.embed_documents(missing_texts)
            if vecs:
                for idx, vec in zip(missing_indices, vecs):
                    self._store[keys[idx]] = vec
                self._dirty = True

        return [self._store.get(k) for k in keys]

    def save(self) -> None:
        """Flush in-memory cache to disk (only writes when dirty)."""
        if not self._dirty:
            return
        try:
            _CACHE_FILE.write_text(json.dumps(self._store), encoding="utf-8")
            self._dirty = False
        except Exception as exc:
            logger.warning(f"EmbeddingCache: could not save ({exc})")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if _CACHE_FILE.exists():
            try:
                self._store = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"EmbeddingCache: could not load ({exc}), starting fresh")
                self._store = {}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
