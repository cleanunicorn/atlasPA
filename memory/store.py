"""
memory/store.py

Persistent memory via local markdown files.

Files:
    soul.md     — The agent's identity and personality (mostly static)
    context.md  — Accumulated knowledge about the user (grows over time)

Phase 2 additions:
    - parse_context_entries()     — splits context.md into individual entries
    - forget_entry(note)          — removes the best-matching entry
    - replace_context_entries()   — for summariser: rebuild context from a summary + entries
    - build_system_prompt(query)  — injects only relevant entries when context is large
"""

import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from memory.retriever import ContextEntry, select_relevant, select_relevant_semantic
from paths import MEMORY_DIR

# If context has more than this many entries, use relevance filtering
CONTEXT_MAX_INJECTED = int(os.getenv("CONTEXT_MAX_INJECTED", "15"))


class MemoryStore:
    """Reads and writes the agent's persistent markdown memory files."""

    def __init__(self) -> None:
        self.soul_path = MEMORY_DIR / "soul.md"
        self.context_path = MEMORY_DIR / "context.md"
        self.location_path = MEMORY_DIR / "location.md"
        self._ensure_files()

        # Semantic memory (disable by setting EMBED_MODEL="" in env)
        from memory.embedder import LocalEmbedder
        from memory.embedding_cache import EmbeddingCache

        self._embedder = LocalEmbedder()
        self._cache = EmbeddingCache() if self._embedder.enabled else None

    def _ensure_files(self) -> None:
        """Create memory files with defaults if they don't exist."""
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.soul_path.exists():
            agent_name = os.getenv("AGENT_NAME", "Atlas")
            self.soul_path.write_text(
                f"# {agent_name}'s Identity\n\n"
                f"You are {agent_name}, a personal AI agent.\n"
                f"You are helpful, direct, and concise.\n"
                f"You remember context between conversations.\n"
                f"You take initiative when appropriate.\n"
            )

        if not self.context_path.exists():
            self.context_path.write_text(
                "# User Context\n\n"
                "_Nothing stored yet. This file grows as you learn about the user._\n"
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_soul(self) -> str:
        """Return the agent's core identity/personality."""
        return self.soul_path.read_text(encoding="utf-8")

    def load_context(self) -> str:
        """Return the full raw context.md content."""
        return self.context_path.read_text(encoding="utf-8")

    def parse_context_entries(self) -> list[ContextEntry]:
        """
        Parse context.md into individual dated entries.

        Each entry starts with a `## YYYY-MM-DD HH:MM` header.
        A leading "Background" block (from summarisation) is returned as
        a single entry with timestamp "" so it's always included.

        Returns:
            Ordered list of ContextEntry objects (chronological).
        """
        text = self.load_context()
        entries: list[ContextEntry] = []

        # Match sections starting with ## followed by a timestamp
        # Pattern: "## 2026-03-09 13:05"
        pattern = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*$", re.MULTILINE)
        splits = list(pattern.finditer(text))

        if not splits:
            # No dated entries — check for a plain background block
            # Strip the header and placeholder lines
            body = re.sub(r"^# .*$", "", text, flags=re.MULTILINE)
            body = body.replace(
                "_Nothing stored yet. This file grows as you learn about the user._", ""
            )
            body = body.strip()
            if body:
                entries.append(ContextEntry(timestamp="", content=body))
            return entries

        # Everything before the first dated entry is the "background" block
        preamble = text[: splits[0].start()].strip()
        # Strip the file header line
        preamble = re.sub(r"^# .*$", "", preamble, flags=re.MULTILINE).strip()
        preamble = preamble.replace(
            "_Nothing stored yet. This file grows as you learn about the user._", ""
        ).strip()
        if preamble:
            entries.append(ContextEntry(timestamp="", content=preamble))

        # Each dated section
        for i, match in enumerate(splits):
            timestamp = match.group(1)
            start = match.end()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            content = text[start:end].strip()
            if content:
                entries.append(ContextEntry(timestamp=timestamp, content=content))

        return entries

    # ── Write ─────────────────────────────────────────────────────────────────

    def append_context(self, note: str) -> None:
        """
        Append a new note to context.md with a timestamp header.
        Called by the Brain's `remember` tool.
        """
        tz_name = os.getenv("AGENT_TIMEZONE", "").strip()
        try:
            tz = ZoneInfo(tz_name) if tz_name else ZoneInfo("localtime")
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
        current = self.context_path.read_text(encoding="utf-8")
        current = current.replace(
            "_Nothing stored yet. This file grows as you learn about the user._\n", ""
        )
        self.context_path.write_text(
            current.rstrip() + f"\n\n## {timestamp}\n{note}\n",
            encoding="utf-8",
        )

    def set_current_location(self, location: str, timezone: str) -> None:
        """
        Persist the user's current location and timezone override.
        Pass empty strings to clear (reset to home).
        """
        if location and timezone:
            self.location_path.write_text(f"{location}\t{timezone}\n", encoding="utf-8")
        elif self.location_path.exists():
            self.location_path.unlink()

    def get_current_location(self) -> tuple[str, str] | None:
        """
        Return (location, timezone) if a travel override is active, else None.
        """
        if not self.location_path.exists():
            return None
        parts = self.location_path.read_text(encoding="utf-8").strip().split("\t", 1)
        if len(parts) == 2 and parts[1]:
            return parts[0], parts[1]
        return None

    def forget_entry(self, note: str) -> str:
        """
        Remove the context entry that best matches `note`.

        Uses token overlap to find the closest entry. Removes it from context.md
        and returns a confirmation message.

        Args:
            note: Description of the fact to forget (partial match is fine).

        Returns:
            Human-readable result message (never raises).
        """
        from memory.retriever import _tokenize, score_relevance

        entries = self.parse_context_entries()
        if not entries:
            return "No memories to forget."

        # Skip background block (timestamp == "") when searching — it's a summary
        dated = [e for e in entries if e.timestamp]
        if not dated:
            return "No individual memory entries to forget (only a background summary exists)."

        query_tokens = _tokenize(note)
        scored = [(score_relevance(e, query_tokens), e) for e in dated]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_entry = scored[0]

        if best_score == 0.0:
            return f"Could not find a memory matching: '{note}'"

        # Remove the matching entry
        remaining = [e for e in entries if e is not best_entry]
        self._write_entries(remaining)
        preview = best_entry.content[:80]
        return f"✅ Forgot: [{best_entry.timestamp}] {preview}"

    def replace_context_entries(
        self, summary: str, recent_entries: list[ContextEntry]
    ) -> None:
        """
        Rebuild context.md from a compressed summary + a list of recent entries.
        Called by the summariser after compressing old entries.
        """
        background = ContextEntry(timestamp="", content=summary)
        self._write_entries([background] + recent_entries)

    def _write_entries(self, entries: list[ContextEntry]) -> None:
        """Serialize a list of ContextEntry objects back to context.md."""
        lines = ["# User Context\n"]
        has_content = False

        for entry in entries:
            if not entry.content.strip():
                continue
            has_content = True
            if entry.timestamp:
                lines.append(f"\n## {entry.timestamp}\n{entry.content.strip()}\n")
            else:
                # Background block — no timestamp header
                lines.append(f"\n{entry.content.strip()}\n")

        if not has_content:
            lines.append(
                "\n_Nothing stored yet. This file grows as you learn about the user._\n"
            )

        self.context_path.write_text("".join(lines), encoding="utf-8")

    # ── System Prompt ─────────────────────────────────────────────────────────

    async def build_system_prompt(
        self, skills_summary: str = "", query: str = ""
    ) -> str:
        """
        Assemble the full system prompt from soul + context + skills.

        When context.md has many entries, only the most relevant ones
        (scored against `query`) are injected.

        Args:
            skills_summary: Compact skill index from SkillRegistry.
            query:          Current user message for relevance scoring.
                            Empty string → inject all entries (up to cap).
        """
        soul = self.load_soul()

        # Timezone priority: travel override > AGENT_TIMEZONE env > machine local
        current_loc = self.get_current_location()
        if current_loc:
            loc_label, tz_name = current_loc
        else:
            loc_label = None
            tz_name = os.getenv("AGENT_TIMEZONE", "").strip()
        try:
            tz = ZoneInfo(tz_name) if tz_name else ZoneInfo("localtime")
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")
            tz_name = "UTC"
        if not tz_name:
            tz_name = datetime.now(tz).strftime("%Z")
        now = datetime.now(tz).strftime("%A, %Y-%m-%d %H:%M")
        location_note = f" — currently in {loc_label}" if loc_label else ""

        # Build the context section
        entries = self.parse_context_entries()
        if entries:
            # Always keep the background block (timestamp == ""); filter dated ones
            background = [e for e in entries if not e.timestamp]
            dated = [e for e in entries if e.timestamp]

            if self._embedder.enabled and self._cache is not None:
                selected_dated = await select_relevant_semantic(
                    dated,
                    query or "",
                    self._embedder,
                    self._cache,
                    top_k=CONTEXT_MAX_INJECTED,
                )
            else:
                selected_dated = select_relevant(
                    dated, query or "", top_k=CONTEXT_MAX_INJECTED
                )

            selected = background + selected_dated
            if len(selected) < len(entries):
                context_text = (
                    "\n\n".join(
                        f"## {e.timestamp}\n{e.content}" if e.timestamp else e.content
                        for e in selected
                    )
                    + f"\n\n_(Showing {len(selected_dated)} of {len(dated)} memories "
                    f"most relevant to this message. Use the `remember` tool to add more.)_"
                )
            else:
                context_text = "\n\n".join(
                    f"## {e.timestamp}\n{e.content}" if e.timestamp else e.content
                    for e in selected
                )
        else:
            context_text = "_Nothing stored yet._"

        parts = [
            soul,
            f"\n---\n**Current time:** {now} ({tz_name}){location_note}\n",
            f"\n---\n# User Context\n\n{context_text}",
        ]

        if skills_summary:
            parts.append(f"\n---\n## Available Skills\n{skills_summary}")

        if os.getenv("CLAUDE_CODE_AVAILABLE", "").lower() == "true":
            parts.append(
                "\n---\n## Claude Code CLI\n"
                "The `claude` CLI (Claude Code) is available via the `run_claude` tool.\n"
                "Use it to generate high-quality code, explore and inspect APIs, "
                "and build new skills. Prefer `run_claude` over writing code yourself "
                "when the task involves complex code generation or API exploration.\n"
                "You can pass a working directory to scope the CLI to a specific project."
            )

        parts.append(
            "\n---\n"
            "When you learn something worth remembering about the user, call `remember`.\n"
            "To remove an outdated fact, call `forget`.\n"
            "Always respond in a conversational, helpful tone.\n"
            "Keep responses concise — this is a chat interface."
        )

        return "\n".join(parts)
