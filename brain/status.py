"""
brain/status.py

DSPy callback that pushes real-time status updates to channels while the
ReAct loop runs.

Architecture:
  - AtlasCallback is registered once via dspy.configure(callbacks=[...]).
  - Before each think() call, activate(queue, loop) binds it to an
    asyncio.Queue + event loop.  DSPy fires on_lm_start / on_tool_start
    in the worker thread; the callback bridges to the async world via
    loop.call_soon_threadsafe.
  - A drain coroutine consumes the queue and calls the channel's on_status.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

# ── Status update ────────────────────────────────────────────────────────────


@dataclass
class StatusUpdate:
    message: str  # Human-readable, e.g. "Searching the web..."
    phase: str  # "lm_start", "tool_start"
    tool_name: str = ""


# ── Tool → human label ──────────────────────────────────────────────────────

_TOOL_LABELS: dict[str, str | None] = {
    "remember": "Saving to memory",
    "forget": "Updating memory",
    "set_location": "Updating location",
    "send_file": "Preparing file",
    "schedule_job": "Scheduling job",
    "list_jobs": "Checking jobs",
    "delete_job": "Deleting job",
    "ask_user": None,  # suppress — user will see the question
    "create_plan": "Planning",
    "reflect": "Reflecting",
    "reload": "Restarting",
    "manage_skills": "Managing skills",
    "finish": "Wrapping up",  # DSPy internal
}


def _tool_status_message(tool_name: str) -> str | None:
    """Return a human-readable status string for a tool, or None to suppress."""
    if tool_name in _TOOL_LABELS:
        label = _TOOL_LABELS[tool_name]
        return f"{label}..." if label else None
    # skill_web_search → "Searching the web..."
    if tool_name.startswith("skill_"):
        pretty = tool_name[6:].replace("_", " ")
        return f"Using {pretty}..."
    return f"Using {tool_name.replace('_', ' ')}..."


# ── DSPy callback ───────────────────────────────────────────────────────────


class AtlasCallback(BaseCallback):
    """Bridges DSPy's sync callback hooks to an asyncio.Queue for status updates."""

    def __init__(self):
        self._queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def activate(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop

    def deactivate(self) -> None:
        self._queue = None
        self._loop = None

    def _put(self, update: StatusUpdate) -> None:
        if self._queue is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, update)
            except (RuntimeError, asyncio.QueueFull):
                pass  # loop closing or queue full — drop silently

    # DSPy hooks ──────────────────────────────────────────────────────────────

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._put(StatusUpdate("Thinking...", "lm_start"))

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        msg = _tool_status_message(getattr(instance, "name", ""))
        if msg:
            self._put(StatusUpdate(msg, "tool_start", getattr(instance, "name", "")))


# ── Drain coroutine ──────────────────────────────────────────────────────────


async def drain_status(queue: asyncio.Queue, on_status) -> None:
    """Consume StatusUpdate objects from the queue and forward to on_status.

    Stops when a None sentinel is received.  Exceptions in on_status are
    swallowed — status display must never crash the brain.
    """
    while True:
        update = await queue.get()
        if update is None:
            break
        try:
            await on_status(update.message)
        except Exception:
            pass
