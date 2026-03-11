"""
brain/engine.py

The Brain — implements the ReAct (Reason + Act) loop.

Loop:
    1. Build context: system prompt (soul + relevant memory + skills index)
    2. Call LLM with conversation history + available tools
    3. If LLM returns tool calls → execute them concurrently → append results → go to 2
    4. If LLM returns final text → return it
    5. Loop max MAX_ITERATIONS times to prevent runaway agents

Phase 6 additions:
    - Parallel tool execution via asyncio.gather
    - `ask_user` built-in: request clarification instead of guessing
    - `create_plan` built-in: structure complex multi-step tasks upfront
    - `reflect` built-in: self-assessment mid-task to catch missed steps
"""

import asyncio
import logging
import re
from collections.abc import Callable, Awaitable
from pathlib import Path
from providers.base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse
from memory.store import MemoryStore
from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10  # Safety cap on the ReAct loop

# ── Response sanitisation ─────────────────────────────────────────────────────

# Some local models (Ollama) leak raw XML tool-call syntax into the text output.
# Strip complete blocks first (including their content), then stray tags.
_TOOL_BLOCK_RE = re.compile(
    r'<(?:tool_call|function_calls|parameter)\b[^>]*>.*?</(?:tool_call|function_calls|parameter)>',
    re.DOTALL | re.IGNORECASE,
)
_TOOL_TAG_RE = re.compile(
    r'</?(?:tool_call|function_calls|function|parameter)\b[^>]*>',
    re.IGNORECASE,
)
# Blank lines left behind after tag removal
_MULTI_BLANK_RE = re.compile(r'\n{3,}')


def _clean_response(text: str) -> str:
    """Strip leaked tool-call XML fragments from the model's text output."""
    text = _TOOL_BLOCK_RE.sub("", text)
    text = _TOOL_TAG_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip() or "(no response)"


class _Clarification:
    """Sentinel returned by ask_user to stop the loop and surface a question."""
    def __init__(self, question: str):
        self.question = question


class _Restart:
    """
    Sentinel returned by the reload tool.
    After the current turn's response is delivered, the process re-execs itself.
    A short delay gives channels time to send the final message before the
    process image is replaced.
    """
    DELAY = 2.0  # seconds between response delivery and os.execv


class Brain:
    """
    ReAct reasoning engine.

    Receives a user message + conversation history, runs a tool-use loop
    until the LLM produces a final text response, and returns it.
    """

    def __init__(self, provider: BaseLLMProvider, memory: MemoryStore, skills: SkillRegistry):
        self.provider = provider
        self.memory = memory
        self.skills = skills
        self._pending_files: list[tuple[Path, str]] = []  # (path, caption) queued by send_file
        self._current_plan: str | None = None  # Set by create_plan, cleared each turn
        self.heartbeat = None  # Set by gateway after Heartbeat is created

        # Built-in tools — always available, regardless of installed skills
        self._builtin_tools = [
            ToolDefinition(
                name="set_location",
                description=(
                    "Update the user's current location. Call this when the user says they "
                    "are travelling to or are currently in a different place. The timezone "
                    "is used so that times and scheduling are always correct for where the "
                    "user actually is. Pass an empty string for both fields to reset to home."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country (e.g. 'Amsterdam, Netherlands'). Empty string to reset.",
                        },
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name (e.g. 'Europe/Amsterdam'). Empty string to reset.",
                        },
                    },
                    "required": ["location", "timezone"],
                },
            ),
            ToolDefinition(
                name="remember",
                description=(
                    "Save an important fact or note about the user to long-term memory. "
                    "Use this when you learn something about the user worth remembering "
                    "across sessions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "The fact or note to remember, written as a concise statement",
                        }
                    },
                    "required": ["note"],
                },
            ),
            ToolDefinition(
                name="forget",
                description=(
                    "Remove an outdated or incorrect fact from long-term memory. "
                    "Use when a previously remembered fact is no longer true or was wrong."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Description of the fact to forget (partial match is fine)",
                        }
                    },
                    "required": ["note"],
                },
            ),
            ToolDefinition(
                name="send_file",
                description=(
                    "Send a file (image, document, screenshot) to the user. "
                    "Use this after creating or saving a file that the user should receive. "
                    "Works for screenshots, exported files, generated images, etc."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to send",
                        },
                        "caption": {
                            "type": "string",
                            "description": "Optional caption to accompany the file",
                        },
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="schedule_job",
                description=(
                    "Schedule a recurring or one-time background task. "
                    "The job will run automatically and send you a message with the result. "
                    "Use a cron expression for recurring jobs (e.g. '0 8 * * *' for daily at 8am), "
                    "or an ISO datetime string for one-time jobs (e.g. '2026-03-10 15:00')."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for this job (e.g. 'morning_briefing')",
                        },
                        "schedule": {
                            "type": "string",
                            "description": "Cron expression ('0 8 * * 1' = Mondays at 8am) or ISO datetime for one-time",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The message to send to yourself when this job runs",
                        },
                    },
                    "required": ["id", "schedule", "prompt"],
                },
            ),
            ToolDefinition(
                name="list_jobs",
                description="List all scheduled background jobs and their status.",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="delete_job",
                description="Delete a scheduled background job by its id.",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The job id to delete",
                        }
                    },
                    "required": ["id"],
                },
            ),
            ToolDefinition(
                name="manage_skills",
                description=(
                    "Install, uninstall, or list addon skills. "
                    "Addon skills are saved to ~/agent-files/skills/ and persist across restarts. "
                    "Use 'install' to add a new skill from SKILL.md + tool.py content. "
                    "The new skill is immediately available after installation. "
                    "Use 'uninstall' to remove an addon skill. Core skills cannot be removed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["install", "uninstall", "list"],
                            "description": "Operation to perform.",
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Skill name (snake_case identifier). "
                                "Required for install and uninstall."
                            ),
                        },
                        "skill_md": {
                            "type": "string",
                            "description": (
                                "Full SKILL.md content for the skill (required for install). "
                                "Should describe what the skill does and how to use it."
                            ),
                        },
                        "tool_py": {
                            "type": "string",
                            "description": (
                                "Full tool.py Python source code (required for install). "
                                "Must define an async or sync `run(**kwargs) -> str` function "
                                "and optionally a PARAMETERS dict (JSON Schema)."
                            ),
                        },
                    },
                    "required": ["action"],
                },
            ),
            # ── Phase 6: advanced reasoning ──────────────────────────────────
            ToolDefinition(
                name="ask_user",
                description=(
                    "Ask the user a single focused clarifying question when their intent is "
                    "ambiguous or you need a specific piece of information to proceed "
                    "accurately. Use this instead of guessing. Do not ask multiple questions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "One clear, specific question for the user",
                        }
                    },
                    "required": ["question"],
                },
            ),
            ToolDefinition(
                name="create_plan",
                description=(
                    "Before tackling a complex multi-step task, lay out a structured plan. "
                    "Call this once at the start to organise your approach, then execute "
                    "the steps using other tools. Helps avoid missed steps on long tasks."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short descriptive title for the plan",
                        },
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ordered list of steps to execute",
                        },
                    },
                    "required": ["title", "steps"],
                },
            ),
            ToolDefinition(
                name="reflect",
                description=(
                    "Verify that you have fully addressed the user's request before "
                    "giving a final answer. Use after executing a plan or finishing a "
                    "complex multi-step task. Identify any gaps and address them."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "What the user originally asked for",
                        },
                        "accomplished": {
                            "type": "string",
                            "description": "Summary of what has been done so far",
                        },
                        "gaps": {
                            "type": "string",
                            "description": "Anything still needed, or 'none'",
                        },
                    },
                    "required": ["goal", "accomplished", "gaps"],
                },
            ),
            ToolDefinition(
                name="reload",
                description=(
                    "Restart the Atlas process to pick up code changes, newly installed "
                    "dependencies, or updated configuration. Use this after: installing a "
                    "skill that requires new packages, editing source files, or when the "
                    "user asks you to restart. The process will restart automatically "
                    "2 seconds after the response is delivered."
                ),
                parameters={"type": "object", "properties": {}},
            ),
        ]

    def _get_all_tools(self) -> list[ToolDefinition]:
        return self._builtin_tools + self.skills.get_tool_definitions()

    async def extract(
        self,
        text: str,
        schema: dict,
        instruction: str = "",
    ) -> dict:
        """
        Extract structured data from *text* as JSON.

        Uses the provider's json_mode so the response is guaranteed to be
        parseable JSON that matches *schema*.  Skills can call this when they
        need structured output from unstructured content (e.g. parsing a
        document, summarising into a fixed shape, etc.).

        Args:
            text:        The raw content to extract from.
            schema:      JSON Schema dict describing the expected output shape.
            instruction: Optional guidance on what to extract (prepended to text).

        Returns:
            Parsed dict.  Raises ValueError if the response is not valid JSON.
        """
        import json as _json

        system = (
            "You are a data-extraction assistant. "
            "Read the input and return ONLY a valid JSON object that matches this schema:\n"
            f"{_json.dumps(schema, indent=2)}\n"
            "No prose, no markdown fences — raw JSON only."
        )
        content = f"{instruction}\n\n{text}".strip() if instruction else text
        response = await self.provider.complete(
            messages=[Message(role="user", content=content)],
            system=system,
            json_mode=True,
        )
        raw = (response.content or "").strip()
        # Strip accidental markdown fences that some models add despite instructions
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"Model returned non-JSON output: {raw[:200]}") from exc

    def take_files(self) -> list[Path]:
        """
        Return and clear the list of files queued by send_file tool calls.
        Called by channels after think() to deliver files to the user.
        """
        files: list[tuple[Path, str]]
        files, self._pending_files = self._pending_files, []
        return files

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a string."""

        if tool_call.name == "set_location":
            location = tool_call.arguments.get("location", "").strip()
            timezone = tool_call.arguments.get("timezone", "").strip()
            self.memory.set_current_location(location, timezone)
            if location:
                logger.info(f"Location set: {location} ({timezone})")
                return f"✅ Location updated: {location} ({timezone})"
            else:
                logger.info("Location reset to home")
                return "✅ Location reset to home timezone."

        if tool_call.name == "remember":
            note = tool_call.arguments.get("note", "")
            self.memory.append_context(note)
            logger.info(f"Remembered: {note[:80]}")
            # Auto-summarise if context has grown large (fire-and-forget)
            try:
                from memory.summariser import maybe_summarise
                await maybe_summarise(self.memory, self.provider)
            except Exception as e:
                logger.warning(f"Summarisation skipped: {e}")
            return f"✅ Remembered: {note}"

        if tool_call.name == "forget":
            note = tool_call.arguments.get("note", "")
            result = self.memory.forget_entry(note)
            logger.info(f"Forget request: '{note[:80]}' → {result[:80]}")
            return result

        if tool_call.name == "send_file":
            path = Path(tool_call.arguments.get("path", "")).expanduser()
            caption = tool_call.arguments.get("caption", "")
            if not path.exists():
                return f"Error: file not found: {path}"
            self._pending_files.append((path, caption))
            logger.info(f"File queued for delivery: {path.name}")
            return f"✅ File queued: {path.name} — it will be sent to the user."

        if tool_call.name == "schedule_job":
            from heartbeat.jobs import Job, upsert_job
            job = Job(
                id=tool_call.arguments.get("id", "").strip(),
                schedule=tool_call.arguments.get("schedule", ""),
                prompt=tool_call.arguments.get("prompt", ""),
                enabled=True,
            )
            if not job.id or not job.schedule or not job.prompt:
                return "Error: 'id', 'schedule', and 'prompt' are all required."
            upsert_job(job)
            if self.heartbeat:
                self.heartbeat.reload_jobs()
            logger.info(f"Scheduled job '{job.id}': {job.schedule}")
            return f"✅ Job '{job.id}' scheduled: {job.schedule}"

        if tool_call.name == "list_jobs":
            from heartbeat.jobs import load_jobs
            jobs = load_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = ["Scheduled jobs:"]
            for j in jobs:
                status = "✅ enabled" if j.enabled else "⏸ disabled"
                lines.append(f"  • {j.id} [{status}]  schedule: {j.schedule}")
                lines.append(f"    prompt: {j.prompt[:80]}{'…' if len(j.prompt) > 80 else ''}")
            return "\n".join(lines)

        if tool_call.name == "delete_job":
            from heartbeat.jobs import remove_job
            job_id = tool_call.arguments.get("id", "")
            removed = remove_job(job_id)
            if not removed:
                return f"No job found with id '{job_id}'."
            if self.heartbeat:
                self.heartbeat.reload_jobs()
            logger.info(f"Deleted job '{job_id}'")
            return f"✅ Job '{job_id}' deleted."

        if tool_call.name == "ask_user":
            question = tool_call.arguments.get("question", "Could you clarify your request?")
            logger.info(f"Clarification requested: {question[:120]}")
            return _Clarification(question)

        if tool_call.name == "create_plan":
            title = tool_call.arguments.get("title", "Plan")
            steps = tool_call.arguments.get("steps", [])
            if not isinstance(steps, list):
                steps = [str(steps)]
            plan_lines = [f"**{title}**"] + [f"{i + 1}. {s}" for i, s in enumerate(steps)]
            self._current_plan = "\n".join(plan_lines)
            logger.info(f"Plan created ({len(steps)} steps): {title}")
            return f"Plan recorded:\n{self._current_plan}"

        if tool_call.name == "reflect":
            goal = tool_call.arguments.get("goal", "")
            accomplished = tool_call.arguments.get("accomplished", "")
            gaps = tool_call.arguments.get("gaps", "none").strip()
            logger.info(f"Reflection — goal: {goal[:80]} | gaps: {gaps[:80]}")
            if gaps.lower() in ("none", "nothing", "n/a", ""):
                return "Reflection complete: all steps done. Proceed with your final answer."
            return f"Reflection: gaps identified — {gaps}. Address these before concluding."

        if tool_call.name == "reload":
            logger.info("Reload requested — process will restart after response is delivered")
            return _Restart()

        if tool_call.name == "manage_skills":
            action = tool_call.arguments.get("action", "")
            if action == "list":
                return self.skills.get_skills_summary()
            elif action == "install":
                name = tool_call.arguments.get("name", "")
                skill_md = tool_call.arguments.get("skill_md", "")
                tool_py = tool_call.arguments.get("tool_py", "")
                if not name:
                    return "Error: 'name' is required for install."
                if not tool_py:
                    return "Error: 'tool_py' is required for install."
                result = self.skills.install(name, skill_md, tool_py)
                logger.info(f"manage_skills install '{name}': {result[:60]}")
                return result
            elif action == "uninstall":
                name = tool_call.arguments.get("name", "")
                if not name:
                    return "Error: 'name' is required for uninstall."
                result = self.skills.uninstall(name)
                logger.info(f"manage_skills uninstall '{name}': {result[:60]}")
                return result
            else:
                return f"Unknown manage_skills action '{action}'. Use: install, uninstall, list."

        if tool_call.name.startswith("skill_"):
            skill_name = tool_call.name[len("skill_"):]
            skill = self.skills.get_skill(skill_name)
            if skill:
                logger.info(f"Running skill: {skill_name} with {tool_call.arguments}")
                return await skill.run(**tool_call.arguments)
            return f"Unknown skill: {skill_name}"

        return f"Unknown tool: {tool_call.name}"

    async def think(
        self,
        user_message: str | list,
        conversation_history: list[Message],
        on_token: Callable[[str], Awaitable[None]] | None = None,
        system_suffix: str = "",
    ) -> tuple[str, list[Message]]:
        """
        Run the ReAct loop for a single user message.

        Args:
            user_message:           The user's new message. May be a plain string
                                    or a list of content blocks (text + images).
            conversation_history:   Previous messages in this conversation.
            on_token:               Optional async callback called with each text
                                    token of the final response as it streams in.
                                    Only fires on the last (non-tool-call) iteration.
            system_suffix:          Optional text appended to the system prompt.
                                    Used by the scheduler to inject "always fetch
                                    fresh data" instructions for background jobs.

        Returns:
            (final_response_text, updated_history)
        """
        self._pending_files = []   # Clear any leftover files from previous turn
        self._current_plan = None  # Reset plan each conversation turn

        # Extract plain text for relevance-based memory filtering
        if isinstance(user_message, list):
            query_text = " ".join(
                b.get("text", "") for b in user_message if b.get("type") == "text"
            )
        else:
            query_text = user_message

        # Build system prompt — pass query for relevance-filtered context injection
        skills_summary = self.skills.get_skills_summary()
        system = self.memory.build_system_prompt(skills_summary, query=query_text)
        if system_suffix:
            system = system + "\n\n---\n" + system_suffix

        messages = list(conversation_history) + [
            Message(role="user", content=user_message)
        ]

        for iteration in range(MAX_ITERATIONS):
            logger.debug(f"ReAct iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Rebuild tool list each iteration so newly installed addon skills
            # are immediately callable within the same conversation.
            all_tools = self._get_all_tools()

            # Use streaming on every call; on_token is only invoked when the
            # response is final text (no tool calls), so it's a no-op for
            # tool-use iterations.
            response: LLMResponse = await self.provider.stream(
                messages=messages,
                tools=all_tools,
                system=system,
                on_token=on_token,
            )

            if response.tool_calls:
                names = [tc.name for tc in response.tool_calls]
                logger.info(f"Tool calls (parallel): {names}")

                messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                ))

                # Execute all tool calls concurrently; capture exceptions per-call
                raw_results = await asyncio.gather(
                    *[self._execute_tool(tc) for tc in response.tool_calls],
                    return_exceptions=True,
                )

                clarification: str | None = None
                restart: _Restart | None = None
                for tc, raw in zip(response.tool_calls, raw_results):
                    if isinstance(raw, _Clarification):
                        clarification = raw.question
                        result_text = f"[Asking user: {raw.question}]"
                    elif isinstance(raw, _Restart):
                        restart = raw
                        result_text = "[Restart scheduled — will restart after this response]"
                    elif isinstance(raw, Exception):
                        result_text = f"Error in {tc.name}: {raw}"
                        logger.error(f"Tool {tc.name} raised: {raw}")
                    else:
                        result_text = raw
                    messages.append(Message(
                        role="tool",
                        content=result_text,
                        tool_call_id=tc.id,
                    ))

                # If ask_user was called, surface the question and stop this turn
                if clarification:
                    messages.append(Message(role="assistant", content=clarification))
                    logger.info("Returning clarification question to user.")
                    return clarification, messages

                # If reload was called, let the LLM produce a final response then restart
                if restart:
                    # One more LLM call so the agent can say "restarting now…"
                    all_tools = self._get_all_tools()
                    response = await self.provider.stream(
                        messages=messages,
                        tools=all_tools,
                        system=system,
                        on_token=on_token,
                    )
                    final_text = _clean_response(response.content or "Restarting…")
                    messages.append(Message(role="assistant", content=final_text))
                    logger.info("Reload confirmed — scheduling process restart")
                    import threading, os as _os, sys as _sys
                    def _do_restart():
                        import time
                        time.sleep(_Restart.DELAY)
                        _os.execv(_sys.executable, [_sys.executable] + _sys.argv)
                    threading.Thread(target=_do_restart, daemon=True, name="atlas-reload").start()
                    return final_text, messages

                continue

            else:
                final_text = _clean_response(response.content or "")
                messages.append(Message(role="assistant", content=final_text))
                logger.info(
                    f"Brain finished in {iteration + 1} iteration(s). "
                    f"Provider: {self.provider.model_name}"
                )
                return final_text, messages

        fallback = "I got stuck in a reasoning loop. Please try rephrasing your request."
        messages.append(Message(role="assistant", content=fallback))
        return fallback, messages
