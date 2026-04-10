"""
brain/tools.py

Built-in tool factories and skill wrappers.

Each factory returns a BrainTool that pairs a callable with the metadata
needed to build a ToolDefinition for the provider's complete() call.
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from providers.base import ToolDefinition
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


# ── BrainTool ───────────────────────────────────────────────────────────────


@dataclass
class BrainTool:
    """A tool the Brain can execute: metadata + callable."""

    name: str
    description: str
    parameters: dict  # Full JSON Schema {"type": "object", "properties": ...}
    func: Callable

    def __call__(self, **kwargs):
        return self.func(**kwargs)

    @property
    def args(self):
        """Properties dict — backward compat for tests that check .args."""
        return self.parameters.get("properties", {})

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


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


# ── Schema builder ──────────────────────────────────────────────────────────


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}


def _func_schema(func: Callable) -> dict:
    """Build a JSON Schema from a function's type-annotated signature."""
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        ann = param.annotation
        json_type = _TYPE_MAP.get(ann, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _tool(func: Callable, name: str | None = None) -> BrainTool:
    """Create a BrainTool from a function (uses name, docstring, signature)."""
    return BrainTool(
        name=name or func.__name__,
        description=func.__doc__ or "",
        parameters=_func_schema(func),
        func=func,
    )


# ── Skill → BrainTool ──────────────────────────────────────────────────────


def _make_skill_tool(skill: Skill) -> BrainTool:
    """Wrap a Skill as a BrainTool using its JSON Schema PARAMETERS."""
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

    return BrainTool(
        name=f"skill_{skill.name}",
        description=skill.description,
        parameters={
            "type": "object",
            "properties": props,
            "required": schema.get("required", []),
        },
        func=wrapper,
    )


# ── Built-in tool factories ─────────────────────────────────────────────────


def _make_remember(memory: MemoryStore) -> BrainTool:
    def remember(note: str) -> str:
        """Save an important fact or note about the user to long-term memory. Use this when you learn something about the user worth remembering across sessions."""
        memory.append_context(note)
        logger.info(f"Remembered: {note[:80]}")
        return f"✅ Remembered: {note}"

    return _tool(remember)


def _make_forget(memory: MemoryStore) -> BrainTool:
    def forget(note: str) -> str:
        """Remove an outdated or incorrect fact from long-term memory. Use when a previously remembered fact is no longer true or was wrong."""
        result = memory.forget_entry(note)
        logger.info(f"Forget: '{note[:80]}' → {result[:60]}")
        return result

    return _tool(forget)


def _make_set_location(memory: MemoryStore) -> BrainTool:
    def set_location(location: str, timezone: str) -> str:
        """Update the user's current location and timezone. Pass empty strings to reset to home."""
        memory.set_current_location(location, timezone)
        if location:
            logger.info(f"Location set: {location} ({timezone})")
            return f"✅ Location updated: {location} ({timezone})"
        logger.info("Location reset to home")
        return "✅ Location reset to home timezone."

    return _tool(set_location)


def _make_send_file(state: _TurnState) -> BrainTool:
    def send_file(path: str, caption: str = "") -> str:
        """Send a file (image, document, screenshot) to the user. Use after creating or saving a file the user should receive."""
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {p}"
        state.pending_files.append((p, caption))
        logger.info(f"File queued: {p.name}")
        return f"✅ File queued: {p.name} — it will be sent to the user."

    return _tool(send_file)


def _make_schedule_job(brain_ref) -> BrainTool:
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

    return _tool(schedule_job)


def _make_list_jobs() -> BrainTool:
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

    return _tool(list_jobs)


def _make_delete_job(brain_ref) -> BrainTool:
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

    return _tool(delete_job)


def _make_ask_user(state: _TurnState) -> BrainTool:
    def ask_user(question: str) -> str:
        """Ask the user a single focused clarifying question when their intent is ambiguous or you need a specific piece of information. Do not ask multiple questions."""
        state.clarification = question
        logger.info(f"Clarification requested: {question[:120]}")
        return "__ASK_USER__"

    return _tool(ask_user)


def _make_create_plan(state: _TurnState) -> BrainTool:
    def create_plan(title: str, steps: list) -> str:
        """Before tackling a complex multi-step task, lay out a structured plan. Call once at the start, then execute the steps using other tools."""
        if not isinstance(steps, list):
            steps = [str(steps)]
        plan_lines = [f"**{title}**"] + [f"{i + 1}. {s}" for i, s in enumerate(steps)]
        state.current_plan = "\n".join(plan_lines)
        logger.info(f"Plan created ({len(steps)} steps): {title}")
        return f"Plan recorded:\n{state.current_plan}"

    return _tool(create_plan)


def _make_reflect() -> BrainTool:
    def reflect(goal: str, accomplished: str, gaps: str = "none") -> str:
        """Verify you have fully addressed the user's request before giving a final answer. Identify any gaps and address them."""
        logger.info(f"Reflection — goal: {goal[:80]} | gaps: {gaps[:80]}")
        if gaps.lower() in ("none", "nothing", "n/a", ""):
            return (
                "Reflection complete: all steps done. Proceed with your final answer."
            )
        return f"Reflection: gaps identified — {gaps}. Address these before concluding."

    return _tool(reflect)


def _make_reload(state: _TurnState) -> BrainTool:
    def reload() -> str:
        """Restart the Atlas process to pick up code changes, newly installed dependencies, or updated configuration."""
        state.restart_requested = True
        logger.info("Reload requested")
        return "__RELOAD__"

    return _tool(reload)


def _make_run_claude() -> BrainTool:
    def run_claude(prompt: str, workdir: str = "", timeout: int = 120) -> str:
        """Run the Claude Code CLI with a prompt. Use this for code generation, API exploration, and building skills. Claude Code is a powerful coding assistant that can read files, search codebases, and write high-quality code."""
        import subprocess

        timeout = min(int(timeout), 300)
        cmd = ["claude", "-p", "--dangerously-skip-permissions", prompt]
        if workdir:
            cwd = Path(workdir).expanduser()
            if not cwd.is_dir():
                return f"Error: directory not found: {cwd}"
        else:
            cwd = None

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if not output.strip():
                output = f"(no output, exit code {result.returncode})"
            if len(output) > 16000:
                half = 8000
                output = (
                    output[:half]
                    + f"\n\n... [{len(output) - 16000} chars truncated] ...\n\n"
                    + output[-half:]
                )
            return output
        except subprocess.TimeoutExpired:
            return f"Error: claude CLI timed out after {timeout}s"
        except FileNotFoundError:
            return "Error: `claude` CLI not found on PATH. Is it installed?"
        except Exception as e:
            return f"Error running claude CLI: {e}"

    return _tool(run_claude)


def _make_update_self(brain_ref) -> BrainTool:
    async def update_self() -> str:
        """Update Atlas to the latest version from GitHub: pulls new code, reinstalls dependencies, and restarts the service. Ask the user for confirmation before calling this."""
        import os as _os
        import signal as _signal
        import subprocess as _subprocess

        root = Path(__file__).parent.parent.resolve()

        # Check whether an update is actually available
        if brain_ref.heartbeat:
            try:
                update_available, check_msg = await brain_ref.heartbeat.check_for_update()
                if not update_available:
                    return f"No update available — {check_msg}"
            except Exception as e:
                logger.warning(f"Pre-update check failed: {e}")

        # ── Step 1: git pull ────────────────────────────────────────────────────
        try:
            result = _subprocess.run(
                ["git", "pull"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=root,
            )
            pull_out = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                return f"❌ git pull failed:\n{pull_out}"
        except Exception as e:
            return f"❌ git pull failed: {e}"

        # ── Step 2: uv sync (install new / updated dependencies) ───────────────
        try:
            result = _subprocess.run(
                ["uv", "sync"],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=root,
            )
            sync_out = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                return f"❌ uv sync failed:\n{sync_out}\n\ngit pull output:\n{pull_out}"
        except FileNotFoundError:
            sync_out = "uv not found — skipping dependency sync"
            logger.warning(sync_out)
        except Exception as e:
            return f"❌ uv sync failed: {e}\n\ngit pull output:\n{pull_out}"

        logger.info(f"Update applied — git pull: {pull_out[:80]}")

        # ── Step 3: restart ─────────────────────────────────────────────────────
        # Try systemd first (the normal production deployment path).
        try:
            result = _subprocess.run(
                ["systemctl", "--user", "restart", "atlas"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return (
                    f"✅ Atlas updated and restarted via systemd.\n\n"
                    f"git pull: {pull_out}"
                )
        except Exception:
            pass  # systemd not available — fall back to in-process restart

        # Fall back: signal the main loop to restart (works in --watch / CLI mode)
        logger.info("Systemd restart unavailable — requesting in-process restart")
        _os.environ["_ATLAS_RESTART"] = "1"
        _os.kill(_os.getpid(), _signal.SIGTERM)
        return f"✅ Atlas updated — restarting now.\n\ngit pull: {pull_out}"

    return BrainTool(
        name="update_self",
        description=(
            "Update Atlas to the latest version from GitHub. "
            "Pulls new code, reinstalls dependencies, and restarts the service. "
            "Always confirm with the user before calling this tool."
        ),
        parameters={"type": "object", "properties": {}},
        func=update_self,
    )


def _make_manage_skills(skills: SkillRegistry) -> BrainTool:
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

    return _tool(manage_skills)
