"""
memory/store.py

Persistent memory via local markdown files.

Files:
    soul.md     — The agent's identity and personality (mostly static)
    context.md  — Accumulated knowledge about the user and ongoing tasks (grows over time)

Design:
    - All memory is plain text / markdown so it's human-readable and editable
    - Memory is injected into the system prompt on every LLM call
    - Future phases can replace this with vector search for longer memories
"""

import os
from pathlib import Path
from datetime import datetime


MEMORY_DIR = Path(__file__).parent


class MemoryStore:
    def __init__(self):
        self.soul_path = MEMORY_DIR / "soul.md"
        self.context_path = MEMORY_DIR / "context.md"
        self._ensure_files()

    def _ensure_files(self):
        """Create memory files with defaults if they don't exist."""
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

    def load_soul(self) -> str:
        """Returns the agent's core identity/personality."""
        return self.soul_path.read_text()

    def load_context(self) -> str:
        """Returns accumulated knowledge about the user."""
        return self.context_path.read_text()

    def append_context(self, note: str):
        """
        Append a new note to the context file.
        Called by the brain when it decides something is worth remembering.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        current = self.context_path.read_text()
        # Remove the placeholder line if still present
        current = current.replace(
            "_Nothing stored yet. This file grows as you learn about the user._\n", ""
        )
        self.context_path.write_text(
            current.rstrip() + f"\n\n## {timestamp}\n{note}\n"
        )

    def build_system_prompt(self, skills_summary: str = "") -> str:
        """
        Assembles the full system prompt from soul + context + skills.
        This is injected on every LLM call.
        """
        soul = self.load_soul()
        context = self.load_context()
        tz = os.getenv("AGENT_TIMEZONE", "UTC")
        now = datetime.now().strftime("%A, %Y-%m-%d %H:%M")

        parts = [
            soul,
            f"\n---\n**Current time:** {now} ({tz})\n",
            f"\n---\n{context}",
        ]

        if skills_summary:
            parts.append(f"\n---\n## Available Skills\n{skills_summary}")

        parts.append(
            "\n---\n"
            "When you want to remember something important about the user, "
            "call the `remember` tool with a concise note.\n"
            "Always respond in a conversational, helpful tone.\n"
            "Keep responses concise — this is a chat interface."
        )

        return "\n".join(parts)
