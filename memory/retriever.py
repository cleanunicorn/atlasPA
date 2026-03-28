"""
memory/retriever.py

Relevance-based context entry selection.

When context.md has many entries, injects only the most relevant ones into
the system prompt rather than the full file. This keeps the context window
lean regardless of how much history has accumulated.

Algorithm: keyword overlap scoring by default; cosine similarity via a local
embedding model when LocalEmbedder and EmbeddingCache are provided.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.embedder import LocalEmbedder
    from memory.embedding_cache import EmbeddingCache


# Stop words excluded from keyword matching (too common to be meaningful)
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "me",
    "him",
    "us",
    "them",
    "that",
    "this",
    "these",
    "those",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "not",
    "no",
    "so",
    "if",
    "as",
    "up",
    "out",
    "about",
    "just",
    "also",
    "very",
    "more",
    "than",
    "then",
}


@dataclass
class ContextEntry:
    """A single parsed entry from context.md."""

    timestamp: str  # e.g. "2026-03-09 13:05"
    content: str  # The note text


def _tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase word tokens, filtering stop words."""
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def score_relevance(entry: ContextEntry, query_tokens: set[str]) -> float:
    """
    Score an entry's relevance to a query by Jaccard-style overlap.
    Returns a float in [0.0, 1.0]. Higher = more relevant.
    """
    if not query_tokens:
        return 0.0
    entry_tokens = _tokenize(entry.content)
    if not entry_tokens:
        return 0.0
    overlap = len(entry_tokens & query_tokens)
    # Normalise by query length so short queries don't dominate
    return overlap / len(query_tokens)


def select_relevant(
    entries: list[ContextEntry],
    query: str,
    top_k: int,
) -> list[ContextEntry]:
    """
    Return the top_k most relevant entries for a query.

    If entries <= top_k, returns all entries in original order (no selection needed).
    Otherwise, scores each entry and returns top_k, preserving chronological order
    within the selection.

    Args:
        entries: Parsed context entries (chronological).
        query:   The current user message (used for relevance scoring).
        top_k:   Maximum number of entries to return.

    Returns:
        Subset of entries, chronological order preserved.
    """
    if len(entries) <= top_k:
        return entries

    query_tokens = _tokenize(query)

    # Score all entries
    scored = [(score_relevance(e, query_tokens), i, e) for i, e in enumerate(entries)]

    # Pick top_k by score (tie-break: prefer more recent = higher index)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected_indices = {i for _, i, _ in scored[:top_k]}

    # Return in chronological order
    return [e for i, e in enumerate(entries) if i in selected_indices]


async def select_relevant_semantic(
    entries: list[ContextEntry],
    query: str,
    embedder: "LocalEmbedder",
    cache: "EmbeddingCache",
    top_k: int,
) -> list[ContextEntry]:
    """
    Semantic version of select_relevant using cosine similarity.

    Embeds the query via encode_query() and context entries via encode_document(),
    then returns the top_k entries by cosine similarity. Falls back to keyword
    scoring if the embedder is unavailable or returns None.

    Args:
        entries:  Parsed context entries (chronological).
        query:    The current user message.
        embedder: LocalEmbedder instance.
        cache:    EmbeddingCache for persistent storage of document vectors.
        top_k:    Maximum number of entries to return.

    Returns:
        Subset of entries, chronological order preserved.
    """
    if len(entries) <= top_k:
        return entries

    # Try semantic scoring
    from memory.embedder import cosine_similarity

    # Queries use encode_query (not cached — ephemeral)
    query_vec = await embedder.embed_query(query)
    if query_vec is None:
        # Embedder unavailable — fall back to keyword scoring
        return select_relevant(entries, query, top_k)

    entry_vecs = await cache.get_or_compute_batch(
        [e.content for e in entries], embedder
    )

    scored = []
    for i, (e, vec) in enumerate(zip(entries, entry_vecs)):
        sim = cosine_similarity(query_vec, vec) if vec is not None else 0.0
        scored.append((sim, i, e))

    cache.save()

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected_indices = {i for _, i, _ in scored[:top_k]}
    return [e for i, e in enumerate(entries) if i in selected_indices]
