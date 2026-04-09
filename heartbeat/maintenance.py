"""
heartbeat/maintenance.py

Daily maintenance task — cleans up expired jobs, consolidates memory,
and prunes stale data.

Context and awareness consolidation use direct LLM calls so the model
decides what to deduplicate, merge, and prune — rather than relying
on simple thresholds or time cutoffs.

Deterministic tasks (expired jobs, embedding cache) run without the LLM.

Usage:
    Called automatically by Heartbeat at MAINTENANCE_HOUR daily.
    Can also be triggered manually:
        await run_maintenance(memory, provider, on_jobs_changed=...)
"""

import hashlib
import json
import logging
import os
import re

from heartbeat.jobs import load_jobs, save_jobs
from memory.retriever import ContextEntry
from memory.store import MemoryStore
from providers.base import BaseLLMProvider, Message
from paths import MEMORY_DIR

logger = logging.getLogger(__name__)

# Minimum entries before bothering the LLM for consolidation
CONTEXT_CONSOLIDATION_THRESHOLD = int(os.getenv("CONTEXT_CONSOLIDATION_THRESHOLD", "5"))
AWARENESS_CONSOLIDATION_THRESHOLD = int(
    os.getenv("AWARENESS_CONSOLIDATION_THRESHOLD", "10")
)

AWARENESS_LOG_FILE = MEMORY_DIR / "awareness_log.json"
EMBEDDINGS_FILE = MEMORY_DIR / "embeddings.json"

MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def run_maintenance(
    memory: MemoryStore,
    provider: BaseLLMProvider | None = None,
    on_jobs_changed: "callable | None" = None,
    notify_callback: "callable | None" = None,
) -> str:
    """
    Run all daily maintenance tasks.

    Args:
        memory:           The MemoryStore instance.
        provider:         LLM provider for consolidation tasks.
        on_jobs_changed:  Called after expired jobs are removed so the scheduler
                          can reload.  Signature: on_jobs_changed() -> None
        notify_callback:  Optional async fn(text, files) to inform the user.

    Returns:
        Human-readable summary of what was done.
    """
    results: list[str] = []

    # 1. Remove expired one-time jobs (deterministic)
    expired = _cleanup_expired_jobs()
    if expired:
        results.append(f"Removed {len(expired)} expired job(s): {', '.join(expired)}")
        if on_jobs_changed:
            on_jobs_changed()

    # 2. Consolidate context entries via LLM
    if provider:
        try:
            before, after = await _consolidate_context(memory, provider)
            if before > 0:
                results.append(f"Consolidated context: {before} → {after} entries")
        except Exception as e:
            logger.error(f"Context consolidation failed: {e}")
            results.append(f"Context consolidation failed: {e}")

        # 3. Consolidate awareness log via LLM
        try:
            before, after = await _consolidate_awareness(provider)
            if before > 0:
                results.append(
                    f"Consolidated awareness log: {before} → {after} entries"
                )
        except Exception as e:
            logger.error(f"Awareness consolidation failed: {e}")
            results.append(f"Awareness consolidation failed: {e}")

    # 4. Prune stale embedding cache entries (deterministic)
    pruned_embeddings = _prune_embedding_cache(memory)
    if pruned_embeddings:
        results.append(f"Pruned {pruned_embeddings} stale embedding cache entries")

    summary = "; ".join(results) if results else "Nothing to clean up"
    logger.info(f"Daily maintenance complete: {summary}")

    if notify_callback and results:
        try:
            await notify_callback(f"🧹 Daily maintenance: {summary}", [])
        except Exception as e:
            logger.error(f"Maintenance notify failed: {e}")

    return summary


# ── Context consolidation (LLM) ────────────────────────────────────────────

_CONTEXT_SYSTEM = """\
You are a memory maintenance system for a personal AI agent.

Given timestamped context entries about a user, consolidate them:
1. Merge entries that cover the same topic into one comprehensive entry.
2. Remove facts that are repeated across multiple entries.
3. Preserve ALL unique facts — never discard information.
4. When merging, use the most recent timestamp of the merged entries.
5. Each output entry should cover a distinct topic or theme.

Output a JSON array of objects with "timestamp" and "content" keys.
Do not wrap the JSON in markdown code fences."""


def _strip_fences(text: str | None) -> str:
    """Remove markdown code fences if the LLM wrapped the output."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


async def _consolidate_context(
    memory: MemoryStore, provider: BaseLLMProvider
) -> tuple[int, int]:
    """
    Use LLM to intelligently consolidate context entries.

    Returns (before_count, after_count).  (0, 0) if skipped.
    """
    entries = memory.parse_context_entries()
    background = [e for e in entries if not e.timestamp]
    dated = [e for e in entries if e.timestamp]

    if len(dated) < CONTEXT_CONSOLIDATION_THRESHOLD:
        return 0, 0

    # Format entries for the LLM
    entry_lines = []
    for i, e in enumerate(dated):
        entry_lines.append(f"Entry {i} [{e.timestamp}]:\n{e.content}")
    entries_text = "\n\n".join(entry_lines)

    response = await provider.complete(
        messages=[Message(role="user", content=entries_text)],
        system=_CONTEXT_SYSTEM,
        max_tokens=MAX_TOKENS,
        json_mode=True,
    )

    raw = _strip_fences(response.content)
    if not raw:
        raise ValueError(
            "LLM returned empty/None for consolidation — "
            "model may not support structured output"
        )

    consolidated = json.loads(raw)

    if not isinstance(consolidated, list) or len(consolidated) == 0:
        raise ValueError(f"LLM returned invalid consolidation: {raw[:200]}")

    new_entries = []
    for item in consolidated:
        ts = item.get("timestamp", "")
        content = item.get("content", "").strip()
        if content:
            new_entries.append(ContextEntry(timestamp=ts, content=content))

    if not new_entries:
        raise ValueError("Consolidation produced zero entries — aborting")

    before_count = len(dated)
    memory._write_entries(background + new_entries)
    logger.info(
        f"Context consolidated: {before_count} → {len(new_entries)} entries "
        f"({before_count - len(new_entries)} removed)"
    )
    return before_count, len(new_entries)


# ── Awareness consolidation (LLM) ──────────────────────────────────────────

_AWARENESS_SYSTEM = """\
You are an awareness log analyzer for a personal AI agent.

Given awareness check log entries, decide which to retain:
1. Always keep entries where "triggered" is true (actual actions taken).
2. Keep recent "no action" entries (last 24 hours) for recency context.
3. Prune older "no action" entries — they add no value.
4. Keep entries that reveal useful patterns about user activity.

Output a JSON array of 0-based integer indices of entries to KEEP.
Do not wrap the JSON in markdown code fences."""


async def _consolidate_awareness(
    provider: BaseLLMProvider,
) -> tuple[int, int]:
    """
    Use LLM to analyze awareness log and prune low-value entries.

    Returns (before_count, after_count).  (0, 0) if skipped.
    """
    if not AWARENESS_LOG_FILE.exists():
        return 0, 0

    try:
        entries = json.loads(AWARENESS_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0

    if len(entries) < AWARENESS_CONSOLIDATION_THRESHOLD:
        return 0, 0

    response = await provider.complete(
        messages=[Message(role="user", content=json.dumps(entries, indent=2))],
        system=_AWARENESS_SYSTEM,
        max_tokens=MAX_TOKENS,
        json_mode=True,
    )

    raw = _strip_fences(response.content)
    if not raw:
        raise ValueError(
            "LLM returned empty/None for awareness pruning — "
            "model may not support structured output"
        )

    keep_indices = set(json.loads(raw))

    # Validate indices
    valid_range = set(range(len(entries)))
    keep_indices = keep_indices & valid_range

    kept = [e for i, e in enumerate(entries) if i in keep_indices]

    # Safety: always keep at least the most recent entry
    if not kept and entries:
        kept = [entries[-1]]

    before_count = len(entries)
    tmp = AWARENESS_LOG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    tmp.replace(AWARENESS_LOG_FILE)

    logger.info(
        f"Awareness log consolidated: {before_count} → {len(kept)} entries "
        f"({before_count - len(kept)} pruned)"
    )
    return before_count, len(kept)


# ── Expired jobs (deterministic) ─────────────────────────────────────────────


def _cleanup_expired_jobs() -> list[str]:
    """Remove one-time jobs whose ISO datetime schedule is in the past."""
    from datetime import datetime, timezone

    jobs = load_jobs()
    if not jobs:
        return []

    now = datetime.now(timezone.utc)
    expired_ids: list[str] = []
    remaining = []

    for job in jobs:
        try:
            dt = datetime.fromisoformat(job.schedule)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < now:
                expired_ids.append(job.id)
                logger.info(f"Expired job: '{job.id}' (scheduled {job.schedule})")
                continue
        except ValueError:
            pass  # Cron expression — keep it
        remaining.append(job)

    if expired_ids:
        save_jobs(remaining)

    return expired_ids


# ── Embedding cache pruning (deterministic) ──────────────────────────────────


def _prune_embedding_cache(memory: MemoryStore) -> int:
    """Remove embedding cache entries for context entries that no longer exist."""
    if not EMBEDDINGS_FILE.exists():
        return 0

    try:
        cache_data = json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return 0

    if not cache_data:
        return 0

    entries = memory.parse_context_entries()
    current_hashes = set()
    for entry in entries:
        h = hashlib.sha256(entry.content.encode("utf-8")).hexdigest()
        current_hashes.add(h)

    original_count = len(cache_data)
    pruned_cache = {k: v for k, v in cache_data.items() if k in current_hashes}
    pruned = original_count - len(pruned_cache)

    if pruned > 0:
        tmp = EMBEDDINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(pruned_cache), encoding="utf-8")
        tmp.replace(EMBEDDINGS_FILE)
        logger.info(f"Pruned {pruned} stale embedding cache entries")

    return pruned
