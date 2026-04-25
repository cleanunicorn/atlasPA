"""
memory/summariser.py

LLM-assisted compression of old context entries and conversation history.

When context.md exceeds CONTEXT_SUMMARY_THRESHOLD entries, the oldest half
is summarised into a compact "Background" block by the LLM.

Additionally, when conversation history approaches the model's context limit,
the oldest messages are summarised to free up space.
"""

import logging
import os

from providers.base import BaseLLMProvider, Message
from memory.store import MemoryStore

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLD = int(os.getenv("CONTEXT_SUMMARY_THRESHOLD", "20"))
HISTORY_KEEP_RECENT = int(os.getenv("HISTORY_KEEP_RECENT", "6"))


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
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
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


async def maybe_summarise_history(
    messages: list[Message],
    provider: BaseLLMProvider,
    limit: int,
    extra_tokens: int = 0,
) -> list[Message]:
    """
    Summarise conversation history if it approaches the token limit.

    If total tokens (including extra_tokens like system prompt) exceed 80% of 'limit',
    the history (except for the most recent messages) is compressed into a single
    summary message.

    Args:
        messages:     The conversation history.
        provider:     The LLM provider for counting and summarising.
        limit:        The model's context window limit.
        extra_tokens: Additional tokens already in the context (system prompt, etc).

    Returns:
        The updated list of messages.
    """
    if not messages:
        return messages

    # Estimate tokens in history
    history_tokens = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        history_tokens += provider.count_tokens(content)
        if m.tool_calls:
            history_tokens += provider.count_tokens(str(m.tool_calls))

    total_usage = history_tokens + extra_tokens

    if total_usage < limit * 0.8:
        return messages

    logger.info(
        f"Context usage ({total_usage}) exceeds 80% of limit ({limit}). "
        f"Summarising history (extra_tokens={extra_tokens})..."
    )

    if len(messages) <= HISTORY_KEEP_RECENT:
        # We can't compress much more without losing the immediate turn context.
        # Just return and hope for the best, or consider dropping oldest.
        logger.warning("History is already at or below keep-recent limit; cannot compact further.")
        return messages

    n_to_compress = len(messages) - HISTORY_KEEP_RECENT
    old_msgs = messages[:n_to_compress]
    recent_msgs = messages[n_to_compress:]

    history_text = ""
    for m in old_msgs:
        content = m.content if isinstance(m.content, str) else "[multimodal content]"
        history_text += f"{m.role}: {content}\n"

    prompt = (
        "Summarize the following conversation history into a concise 'Background' "
        "that captures all key details, user preferences, and active goals. "
        "Output ONLY the summary paragraph, no intro or outro.\n\n"
        f"{history_text}"
    )

    try:
        response = await provider.complete(
            messages=[Message(role="user", content=prompt)],
            system="You are a conversation summarizer. Distill the history into a concise background block.",
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        )
        summary = (response.content or "").strip()
        if not summary:
            logger.warning("History summariser returned empty response — skipping")
            return messages

        summary_msg = Message(
            role="assistant", content=f"[Conversation Summary]: {summary}"
        )
        logger.info(f"Summarised {n_to_compress} messages into one summary message.")
        return [summary_msg] + recent_msgs

    except Exception as e:
        logger.error(f"History summarisation failed: {e}")
        return messages
