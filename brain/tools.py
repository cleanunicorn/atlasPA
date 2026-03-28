"""
brain/tools.py

Built-in tool factories and skill-to-dspy.Tool wrappers.

Each factory returns a dspy.Tool that closes over per-turn state, memory,
skill registry, or brain reference as needed.
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path

import dspy

from memory.store import MemoryStore
from skills.registry import SkillRegistry, Skill

logger = logging.getLogger(__name__)

_RESTART_DELAY = 2.0  # seconds before os.execv

_DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "input": {"type": "string", "description": "The input for this skill"}
    },
    "required": ["input"],
}


# ── Per-turn shared state ────────────────────────────────────────────────────


@dataclass
class _TurnState:
    """Mutable state shared between tool closures and think() for one turn."""

    clarification: str | None = None  # set by ask_user; stops loop early
    restart_requested: bool = False  # set by reload
    pending_files: list = field(default_factory=list)  # (Path, caption)
    current_plan: str | None = None


# ── Async skill bridge ───────────────────────────────────────────────────────


def _run_skill_sync(skill: Skill, kwargs: dict) -> str:
    """Run a skill's run() inside the calling thread (may be a worker thread)."""
    result = skill._module.run(**kwargs)
    if inspect.iscoroutine(result):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(result)
        finally:
            loop.close()
    return str(result)


# ── Skill → dspy.Tool ───────────────────────────────────────────────────────


def _make_skill_tool(skill: Skill) -> dspy.Tool:
    """Wrap a Skill as a dspy.Tool using its JSON Schema PARAMETERS."""
    schema = (
        getattr(skill._module, "PARAMETERS", _DEFAULT_SCHEMA)
        if skill._module
        else _DEFAULT_SCHEMA
    )
    props = schema.get("properties", {"input": {"type": "string"}})

    captured = skill

    def wrapper(**kwargs) -> str:
        return _run_skill_sync(captured, kwargs)

    wrapper.__name__ = f"skill_{skill.name}"

    return dspy.Tool(
        func=wrapper,
        name=f"skill_{skill.name}",
        desc=skill.description,
        args=props,
    )


# ── Built-in tool factories ─────────────────────────────────────────────────


def _make_remember(memory: MemoryStore) -> dspy.Tool:
    def remember(note: str) -> str:
        """Save an important fact or note about the user to long-term memory. Use this when you learn something about the user worth remembering across sessions."""
        memory.append_context(note)
        logger.info(f"Remembered: {note[:80]}")
        return f"✅ Remembered: {note}"

    return dspy.Tool(remember)


def _make_forget(memory: MemoryStore) -> dspy.Tool:
    def forget(note: str) -> str:
        """Remove an outdated or incorrect fact from long-term memory. Use when a previously remembered fact is no longer true or was wrong."""
        result = memory.forget_entry(note)
        logger.info(f"Forget: '{note[:80]}' → {result[:60]}")
        return result

    return dspy.Tool(forget)


def _make_set_location(memory: MemoryStore) -> dspy.Tool:
    def set_location(location: str, timezone: str) -> str:
        """Update the user's current location and timezone. Pass empty strings to reset to home."""
        memory.set_current_location(location, timezone)
        if location:
            logger.info(f"Location set: {location} ({timezone})")
            return f"✅ Location updated: {location} ({timezone})"
        logger.info("Location reset to home")
        return "✅ Location reset to home timezone."

    return dspy.Tool(set_location)


def _make_send_file(state: _TurnState) -> dspy.Tool:
    def send_file(path: str, caption: str = "") -> str:
        """Send a file (image, document, screenshot) to the user. Use after creating or saving a file the user should receive."""
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {p}"
        state.pending_files.append((p, caption))
        logger.info(f"File queued: {p.name}")
        return f"✅ File queued: {p.name} — it will be sent to the user."

    return dspy.Tool(send_file)


def _make_schedule_job(brain_ref) -> dspy.Tool:
    def schedule_job(job_id: str, schedule: str, prompt: str) -> str:
        """Schedule a recurring or one-time background task using a cron expression or ISO datetime."""
        from heartbeat.jobs import Job, upsert_job

        job = Job(id=job_id.strip(), schedule=schedule, prompt=prompt, enabled=True)
        if not job.id or not job.schedule or not job.prompt:
            return "Error: 'job_id', 'schedule', and 'prompt' are all required."
        upsert_job(job)
        if brain_ref.heartbeat:
            brain_ref.heartbeat.reload_jobs()
        logger.info(f"Scheduled job '{job.id}': {job.schedule}")
        return f"✅ Job '{job.id}' scheduled: {job.schedule}"

    return dspy.Tool(schedule_job)


def _make_list_jobs() -> dspy.Tool:
    def list_jobs() -> str:
        """List all scheduled background jobs and their status."""
        from heartbeat.jobs import load_jobs

        jobs = load_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines = ["Scheduled jobs:"]
        for j in jobs:
            status = "✅ enabled" if j.enabled else "⏸ disabled"
            lines.append(f"  • {j.id} [{status}]  schedule: {j.schedule}")
            lines.append(
                f"    prompt: {j.prompt[:80]}{'…' if len(j.prompt) > 80 else ''}"
            )
        return "\n".join(lines)

    return dspy.Tool(list_jobs)


def _make_delete_job(brain_ref) -> dspy.Tool:
    def delete_job(job_id: str) -> str:
        """Delete a scheduled background job. Use list_jobs first to find the exact job_id."""
        from heartbeat.jobs import remove_job

        removed = remove_job(job_id)
        if not removed:
            return f"No job found with job_id '{job_id}'."
        if brain_ref.heartbeat:
            brain_ref.heartbeat.reload_jobs()
        logger.info(f"Deleted job '{job_id}'")
        return f"✅ Job '{job_id}' deleted."

    return dspy.Tool(delete_job)


def _make_ask_user(state: _TurnState) -> dspy.Tool:
    def ask_user(question: str) -> str:
        """Ask the user a single focused clarifying question when their intent is ambiguous or you need a specific piece of information. Do not ask multiple questions."""
        state.clarification = question
        logger.info(f"Clarification requested: {question[:120]}")
        return "__ASK_USER__"

    return dspy.Tool(ask_user)


def _make_create_plan(state: _TurnState) -> dspy.Tool:
    def create_plan(title: str, steps: list) -> str:
        """Before tackling a complex multi-step task, lay out a structured plan. Call once at the start, then execute the steps using other tools."""
        if not isinstance(steps, list):
            steps = [str(steps)]
        plan_lines = [f"**{title}**"] + [f"{i + 1}. {s}" for i, s in enumerate(steps)]
        state.current_plan = "\n".join(plan_lines)
        logger.info(f"Plan created ({len(steps)} steps): {title}")
        return f"Plan recorded:\n{state.current_plan}"

    return dspy.Tool(create_plan)


def _make_reflect() -> dspy.Tool:
    def reflect(goal: str, accomplished: str, gaps: str = "none") -> str:
        """Verify you have fully addressed the user's request before giving a final answer. Identify any gaps and address them."""
        logger.info(f"Reflection — goal: {goal[:80]} | gaps: {gaps[:80]}")
        if gaps.lower() in ("none", "nothing", "n/a", ""):
            return (
                "Reflection complete: all steps done. Proceed with your final answer."
            )
        return f"Reflection: gaps identified — {gaps}. Address these before concluding."

    return dspy.Tool(reflect)


def _make_reload(state: _TurnState) -> dspy.Tool:
    def reload() -> str:
        """Restart the Atlas process to pick up code changes, newly installed dependencies, or updated configuration."""
        state.restart_requested = True
        logger.info("Reload requested")
        return "__RELOAD__"

    return dspy.Tool(reload)


def _make_manage_skills(skills: SkillRegistry) -> dspy.Tool:
    def manage_skills(
        action: str, name: str = "", skill_md: str = "", tool_py: str = ""
    ) -> str:
        """Install, uninstall, or list addon skills. Use 'install' to add a new skill, 'uninstall' to remove one, 'list' to see all."""
        if action == "list":
            return skills.get_skills_summary()
        if action == "install":
            if not name:
                return "Error: 'name' is required for install."
            if not tool_py:
                return "Error: 'tool_py' is required for install."
            result = skills.install(name, skill_md, tool_py)
            logger.info(f"manage_skills install '{name}': {result[:60]}")
            return result
        if action == "uninstall":
            if not name:
                return "Error: 'name' is required for uninstall."
            result = skills.uninstall(name)
            logger.info(f"manage_skills uninstall '{name}': {result[:60]}")
            return result
        return (
            f"Unknown manage_skills action '{action}'. Use: install, uninstall, list."
        )

    return dspy.Tool(manage_skills)
