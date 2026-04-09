"""
brain/status.py

Real-time status updates for channels while the ReAct loop runs.

The Brain calls emit_status() directly before LLM calls and tool executions.
Status messages are forwarded to the channel's on_status callback.
"""

import logging
from dataclasses import dataclass

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
}


def _tool_status_message(tool_name: str) -> str | None:
    """Return a human-readable status string for a tool, or None to suppress."""
    if tool_name in _TOOL_LABELS:
        label = _TOOL_LABELS[tool_name]
        return f"{label}..." if label else None
    # skill_web_search → "Using web search..."
    if tool_name.startswith("skill_"):
        pretty = tool_name[6:].replace("_", " ")
        return f"Using {pretty}..."
    return f"Using {tool_name.replace('_', ' ')}..."
