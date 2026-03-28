"""
memory/history.py

Persistent per-user conversation history backed by JSON files.

Each user gets a file: memory/history/<user_id>.json
History survives agent restarts. Capped at HISTORY_MAX_MESSAGES (default 200).

Design:
    - Only user/assistant turns are persisted; internal tool call messages
      are included so the LLM can reconstruct its reasoning across sessions.
    - File writes happen after every message (atomic via temp file).
"""

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from providers.base import Message
from paths import HISTORY_DIR

logger = logging.getLogger(__name__)
MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "200"))


class ConversationHistory:
    """
    Persistent per-user conversation history.

    Usage:
        history = ConversationHistory()
        messages = history.load("user_123")
        # ... after brain.think() ...
        history.save("user_123", updated_messages)
        history.clear("user_123")
    """

    def __init__(self) -> None:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        # Sanitise user_id so it's safe as a filename
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(user_id))
        return HISTORY_DIR / f"{safe}.json"

    def load(self, user_id: str) -> list[Message]:
        """Load conversation history for a user. Returns [] if none saved yet."""
        path = self._path(user_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [Message(**m) for m in data]
        except Exception as e:
            logger.warning(f"Could not load history for {user_id}: {e} — starting fresh")
            return []

    @staticmethod
    def _strip_images(messages: list[Message]) -> list[Message]:
        """Replace base64 image data with a placeholder to keep history files small."""
        result = []
        for msg in messages:
            if isinstance(msg.content, list):
                stripped = []
                for block in msg.content:
                    if block.get("type") == "image":
                        stripped.append({"type": "text", "text": "[image attached]"})
                    else:
                        stripped.append(block)
                result.append(Message(
                    role=msg.role,
                    content=stripped,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=msg.tool_calls,
                ))
            else:
                result.append(msg)
        return result

    def save(self, user_id: str, messages: list[Message]) -> None:
        """
        Persist conversation history for a user.
        Caps at MAX_MESSAGES (drops oldest messages first).
        Uses a temp file + rename for atomic writes.
        """
        if len(messages) > MAX_MESSAGES:
            messages = messages[-MAX_MESSAGES:]

        messages = self._strip_images(messages)
        path = self._path(user_id)
        payload = json.dumps([asdict(m) for m in messages], indent=2, ensure_ascii=False)

        # Atomic write: write to temp file in same dir, then rename
        try:
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Failed to save history for {user_id}: {e}")

    def clear(self, user_id: str) -> None:
        """Delete the saved history for a user."""
        path = self._path(user_id)
        if path.exists():
            path.unlink()
            logger.info(f"Cleared history for user {user_id}")
