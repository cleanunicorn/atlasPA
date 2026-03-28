"""
memory/summariser.py

LLM-assisted compression of old context entries.

When context.md exceeds CONTEXT_SUMMARY_THRESHOLD entries, the oldest half
is summarised into a compact "Background" block by the LLM. This keeps the
context window lean while preserving the gist of long-term memory.

Called automatically by the Brain after each `remember` tool call.
"""

import logging
import os

from providers.base import BaseLLMProvider, Message
from memory.store import MemoryStore

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLD = int(os.getenv("CONTEXT_SUMMARY_THRESHOLD", "20"))


async def maybe_summarise(store: MemoryStore, provider: BaseLLMProvider) -> bool:
    """
    Compress old context entries if the count exceeds SUMMARY_THRESHOLD.

    Args:
        store:    The MemoryStore to read from and write back to.
        provider: The LLM provider used to generate the summary.

    Returns:
        True if a summarisation was performed, False otherwise.
    """
    entries = store.parse_context_entries()
    if len(entries) <= SUMMARY_THRESHOLD:
        return False

    n_to_compress = len(entries) // 2
    old_entries = entries[:n_to_compress]
    recent_entries = entries[n_to_compress:]

    logger.info(
        f"Context has {len(entries)} entries (threshold: {SUMMARY_THRESHOLD}). "
        f"Compressing oldest {n_to_compress} entries..."
    )

    old_text = "\n\n".join(f"[{e.timestamp}] {e.content}" for e in old_entries)

    prompt = (
        "The following are memory notes about a user, ordered by time. "
        "Compress them into a single concise paragraph (max 150 words) that "
        "preserves all important facts. Output only the compressed paragraph, "
        "no headings or commentary.\n\n"
        f"{old_text}"
    )

    try:
        response = await provider.complete(
            messages=[Message(role="user", content=prompt)],
            system=(
                "You are a memory compressor. Your job is to distill factual notes "
                "into a dense, accurate summary. Never lose important facts. "
                "Output only the summary paragraph."
            ),
        )
        summary = (response.content or "").strip()
        if not summary:
            logger.warning("Summariser returned empty response — skipping compression")
            return False

        store.replace_context_entries(summary, recent_entries)
        logger.info(
            f"Summarised {n_to_compress} entries → {len(summary)} chars. "
            f"{len(recent_entries)} recent entries preserved."
        )
        return True

    except Exception as e:
        logger.error(f"Summarisation failed: {e} — context.md unchanged")
        return False
