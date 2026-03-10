"""
brain/engine.py

The Brain — implements the ReAct (Reason + Act) loop.

Loop:
    1. Build context: system prompt (soul + relevant memory + skills index)
    2. Call LLM with conversation history + available tools
    3. If LLM returns tool calls → execute them → append results → go to 2
    4. If LLM returns final text → return it
    5. Loop max MAX_ITERATIONS times to prevent runaway agents

Phase 2 additions:
    - `forget` built-in tool: remove a memory entry by content match
    - Relevance-aware system prompt: passes current query to MemoryStore
    - Auto-summarisation: compresses context.md after `remember` if needed
"""

import logging
import re
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
        self.heartbeat = None  # Set by gateway after Heartbeat is created

        # Built-in tools — always available, regardless of installed skills
        self._builtin_tools = [
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
        ]

    def _get_all_tools(self) -> list[ToolDefinition]:
        return self._builtin_tools + self.skills.get_tool_definitions()

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
        user_message: str,
        conversation_history: list[Message],
    ) -> tuple[str, list[Message]]:
        """
        Run the ReAct loop for a single user message.

        Args:
            user_message:           The user's new message.
            conversation_history:   Previous messages in this conversation.

        Returns:
            (final_response_text, updated_history)
        """
        self._pending_files = []  # Clear any leftover files from previous turn

        # Build system prompt — pass query for relevance-filtered context injection
        skills_summary = self.skills.get_skills_summary()
        system = self.memory.build_system_prompt(skills_summary, query=user_message)

        messages = list(conversation_history) + [
            Message(role="user", content=user_message)
        ]

        for iteration in range(MAX_ITERATIONS):
            logger.debug(f"ReAct iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Rebuild tool list each iteration so newly installed addon skills
            # are immediately callable within the same conversation.
            all_tools = self._get_all_tools()

            response: LLMResponse = await self.provider.complete(
                messages=messages,
                tools=all_tools,
                system=system,
            )

            if response.tool_calls:
                logger.info(f"Tool calls: {[tc.name for tc in response.tool_calls]}")

                messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                ))

                for tool_call in response.tool_calls:
                    result = await self._execute_tool(tool_call)
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                    ))

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
