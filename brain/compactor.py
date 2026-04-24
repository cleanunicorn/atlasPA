"""
brain/compactor.py

Token-aware conversation history compaction.

When `system_prompt + conversation_history + query` is estimated to exceed
CONTEXT_COMPACTION_THRESHOLD * CONTEXT_MAX_TOKENS, the oldest half of the
history is LLM-summarised into one synthetic user message prefixed with
SUMMARY_MARKER. The recent tail (at least CONTEXT_COMPACTION_KEEP_RECENT
messages) is preserved verbatim. The cut never splits a tool-use/tool-result
pair. Best-effort: on provider failure the original list is returned.

Mirrors the shape of memory/summariser.py but operates on conversation history
instead of long-term context entries.
"""

import logging
import os

from providers.base import BaseLLMProvider, Message

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_MAX_TOKENS = 200_000
DEFAULT_COMPACTION_THRESHOLD = 0.8
DEFAULT_KEEP_RECENT = 10
SUMMARY_MARKER = "[Conversation summary of earlier messages]"


def _extract_text(content: str | list) -> str:
    """Flatten message content to plain text (mirrors brain.engine._extract_text)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def estimate_tokens(text: str) -> int:
    """Char-based heuristic: len(text) // 4. Conservative — undercounts dense
    inputs like code/JSON, which makes compaction trigger slightly early."""
    if not text:
        return 0
    return len(text) // 4


def estimate_history_tokens(messages: list[Message]) -> int:
    """Sum estimated tokens across all messages, handling multimodal content."""
    total = 0
    for m in messages:
        total += estimate_tokens(_extract_text(m.content))
        total += estimate_tokens(m.role)
    return total


def _is_already_compacted(messages: list[Message]) -> bool:
    """True iff the first message is a user message carrying a prior summary."""
    if not messages:
        return False
    first = messages[0]
    if first.role != "user":
        return False
    text = _extract_text(first.content)
    return text.startswith(SUMMARY_MARKER)


def _find_safe_cut(
    messages: list[Message], desired_cut: int, keep_recent: int
) -> int:
    """Expand cut forward past any tool-use/tool-result boundary.

    Ensures the post-summary tail does not start with a `role="tool"` message
    and is not preceded (in the original list) by an assistant message with
    tool_calls — which would leave a dangling tool_use without its tool_result
    after compaction.
    """
    cut = desired_cut
    upper = len(messages) - keep_recent
    while cut < upper:
        if messages[cut].role == "tool":
            cut += 1
            continue
        if cut - 1 >= 0 and messages[cut - 1].tool_calls:
            cut += 1
            continue
        break
    return min(cut, upper)


def _serialize_for_summary(messages: list[Message]) -> str:
    """Render messages as `role: text` lines for the summariser prompt."""
    lines = []
    for m in messages:
        text = _extract_text(m.content)
        if not text:
            continue
        lines.append(f"{m.role}: {text}")
    return "\n".join(lines)


async def maybe_compact_history(
    messages: list[Message],
    provider: BaseLLMProvider,
    *,
    system_prompt_tokens: int,
    query_tokens: int,
) -> tuple[list[Message], bool]:
    """
    Compact the oldest half of conversation history if the estimated total
    exceeds CONTEXT_COMPACTION_THRESHOLD * CONTEXT_MAX_TOKENS.

    Returns (messages, was_compacted). On any error, returns (messages, False).

    Env vars (read fresh on each call so tests can override):
        CONTEXT_MAX_TOKENS            — default 200000
        CONTEXT_COMPACTION_THRESHOLD  — default 0.8
        CONTEXT_COMPACTION_KEEP_RECENT — default 10
    """
    context_max = int(os.getenv("CONTEXT_MAX_TOKENS", str(DEFAULT_CONTEXT_MAX_TOKENS)))
    threshold = float(
        os.getenv("CONTEXT_COMPACTION_THRESHOLD", str(DEFAULT_COMPACTION_THRESHOLD))
    )
    keep_recent = int(
        os.getenv("CONTEXT_COMPACTION_KEEP_RECENT", str(DEFAULT_KEEP_RECENT))
    )

    if len(messages) < keep_recent + 2:
        return messages, False

    history_tokens = estimate_history_tokens(messages)
    total_tokens = system_prompt_tokens + query_tokens + history_tokens
    budget = int(threshold * context_max)
    if total_tokens < budget:
        return messages, False

    desired_cut = len(messages) // 2
    cut = _find_safe_cut(messages, desired_cut, keep_recent)
    if cut <= 0:
        return messages, False

    old = messages[:cut]
    recent = messages[cut:]
    old_text = _serialize_for_summary(old)
    if not old_text.strip():
        return messages, False

    system_prompt = (
        "You are a conversation compressor. Distill the following multi-turn "
        "dialogue into a compact narrative (max 400 words) preserving all "
        "facts, user preferences, decisions, file paths, names, and pending "
        "tasks. Output only the summary paragraph — no headings, no commentary."
    )

    logger.info(
        f"Compacting conversation: {len(old)} old + {len(recent)} recent "
        f"messages (~{history_tokens} heuristic tokens, budget {budget})"
    )

    try:
        response = await provider.complete(
            messages=[Message(role="user", content=old_text)],
            system=system_prompt,
            max_tokens=1024,
        )
        summary = (response.content or "").strip()
        if not summary:
            logger.warning("Compactor returned empty summary — keeping original history")
            return messages, False
    except Exception as e:
        logger.warning(f"Compactor failed: {e} — keeping original history")
        return messages, False

    summary_msg = Message(
        role="user",
        content=f"{SUMMARY_MARKER}: {summary}",
    )
    compacted = [summary_msg] + recent
    logger.info(
        f"Compressed conversation: {len(messages)} → {len(compacted)} messages "
        f"({len(summary)} char summary)"
    )
    return compacted, True
