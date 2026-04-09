"""
brain/engine.py

The Brain — implements a ReAct (Reason + Act) loop using the provider's
native tool-calling API.

No DSPy dependency.  The loop calls provider.complete() iteratively,
executing tool calls until the model produces a final text response.

Key design notes:
  - Natively async — no thread bridging needed.
  - Tools are BrainTool instances (brain/tools.py) converted to
    ToolDefinition for the provider.
  - Async skill run() functions are bridged with a fresh event loop
    in a worker thread (via asyncio.to_thread).
  - Built-in tools close over a per-turn _TurnState to propagate
    side-effects (file queuing, clarifications, plan tracking).
"""

import asyncio
import json as _json
import logging
import os
import re
from pathlib import Path
from collections.abc import Callable, Awaitable

from providers.base import BaseLLMProvider, Message, ToolCall
from memory.store import MemoryStore
from skills.registry import SkillRegistry

from brain.status import _tool_status_message
from brain.tools import (  # noqa: F401 — re-exported for tests
    BrainTool,
    _TurnState,
    _run_skill_sync,
    _make_skill_tool,
    _make_remember,
    _make_forget,
    _make_set_location,
    _make_send_file,
    _make_schedule_job,
    _make_list_jobs,
    _make_delete_job,
    _make_ask_user,
    _make_create_plan,
    _make_reflect,
    _make_reload,
    _make_manage_skills,
    _make_run_claude,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 30
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ── Response sanitisation ────────────────────────────────────────────────────

_TOOL_BLOCK_RE = re.compile(
    r"<(?:tool_call|function_calls|parameter)\b[^>]*>.*?</(?:tool_call|function_calls|parameter)>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_TAG_RE = re.compile(
    r"</?(?:tool_call|function_calls|function|parameter)\b[^>]*>",
    re.IGNORECASE,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _clean_response(text: str) -> str:
    """Strip leaked tool-call XML fragments from the model's text output."""
    text = _TOOL_BLOCK_RE.sub("", text)
    text = _TOOL_TAG_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


# ── Helper: extract text from multimodal content ────────────────────────────


def _extract_text(content: str | list) -> str:
    """Extract plain text from a message content (handles multimodal blocks)."""
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if b.get("type") == "text")


# ── Helper: execute a tool call ─────────────────────────────────────────────


async def _execute_tool(tool_map: dict[str, BrainTool], tc: ToolCall) -> str:
    """Execute a tool call and return the result string."""
    tool = tool_map.get(tc.name)
    if tool is None:
        return f"Error: unknown tool '{tc.name}'"

    try:
        result = tool.func(**tc.arguments)
        # Handle async tool functions
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)
    except Exception as e:
        logger.warning(f"Tool {tc.name} raised: {e}")
        return f"Error in {tc.name}: {e}"


# ── Brain ────────────────────────────────────────────────────────────────────


class Brain:
    """
    ReAct reasoning engine using native tool calling.

    Wraps provider.complete() with Atlas's skill registry, memory,
    and built-in tools.  Public interface is identical to the previous engine.
    """

    def __init__(
        self, provider: BaseLLMProvider, memory: MemoryStore, skills: SkillRegistry
    ):
        self.provider = provider
        self.memory = memory
        self.skills = skills
        self._pending_files: list[tuple[Path, str]] = []
        self._current_plan: str | None = None
        self.heartbeat = None
        self._session_tokens: dict[str, int] = {"input": 0, "output": 0}

        logger.info(f"Brain ready: {provider.model_name}")

    # ── Tool construction ────────────────────────────────────────────────────

    def _build_tools(self, state: _TurnState) -> list[BrainTool]:
        """Build the full tool list for one turn: built-ins + skills."""
        tools = [
            _make_remember(self.memory),
            _make_forget(self.memory),
            _make_set_location(self.memory),
            _make_send_file(state),
            _make_schedule_job(self),
            _make_list_jobs(),
            _make_delete_job(self),
            _make_ask_user(state),
            _make_create_plan(state),
            _make_reflect(),
            _make_reload(state),
            _make_manage_skills(self.skills),
        ]
        if os.getenv("CLAUDE_CODE_AVAILABLE", "").lower() == "true":
            tools.append(_make_run_claude())
        # Append skill tools (rebuilt each call to pick up newly installed addons)
        for skill in self.skills._skills.values():
            try:
                tools.append(_make_skill_tool(skill))
            except Exception as e:
                logger.warning(f"Could not wrap skill '{skill.name}': {e}")
        return tools

    # ── ReAct loop ──────────────────────────────────────────────────────────

    async def _react_loop(
        self,
        system: str,
        messages: list[Message],
        tools: list[BrainTool],
        state: _TurnState,
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Run the ReAct loop: call LLM, execute tools, repeat until text response.

        Returns the model's final text answer.
        """
        tool_map = {t.name: t for t in tools}
        tool_defs = [t.to_definition() for t in tools]

        for i in range(MAX_ITERATIONS):
            if on_status:
                try:
                    await on_status("Thinking...")
                except Exception:
                    pass

            response = await self.provider.complete(
                messages=messages,
                tools=tool_defs,
                system=system,
                max_tokens=MAX_TOKENS,
            )

            # Track token usage
            if response.usage:
                self._session_tokens["input"] += response.usage.get("input_tokens", 0)
                self._session_tokens["output"] += response.usage.get(
                    "output_tokens", 0
                )

            # No tool calls → final answer
            if not response.tool_calls:
                return response.content or "Done."

            # Append assistant message with tool calls
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=[
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in response.tool_calls
                    ],
                )
            )

            # Execute each tool call
            for tc in response.tool_calls:
                if on_status:
                    status_msg = _tool_status_message(tc.name)
                    if status_msg:
                        try:
                            await on_status(status_msg)
                        except Exception:
                            pass

                result = await _execute_tool(tool_map, tc)

                messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                    )
                )

                # Early exit: ask_user or reload
                if state.clarification or state.restart_requested:
                    return response.content or ""

            logger.debug(f"ReAct iteration {i + 1}/{MAX_ITERATIONS} complete")

        logger.warning("ReAct loop hit max iterations")
        return response.content or "Done."

    # ── Public interface ─────────────────────────────────────────────────────

    async def think(
        self,
        user_message: str | list,
        conversation_history: list[Message],
        on_status: Callable[[str], Awaitable[None]] | None = None,
        system_suffix: str = "",
    ) -> tuple[str, list[Message]]:
        """
        Run the ReAct loop for a single user message.

        on_status receives human-readable progress strings ("Thinking...",
        "Using web search...") while the loop runs.  Safe to ignore.
        """
        self._pending_files = []
        self._current_plan = None
        state = _TurnState()

        query_text = _extract_text(user_message)

        skills_summary = self.skills.get_skills_summary()
        system = await self.memory.build_system_prompt(skills_summary, query=query_text)
        if system_suffix:
            system += "\n\n---\n" + system_suffix

        messages = list(conversation_history) + [
            Message(role="user", content=user_message)
        ]

        tools = self._build_tools(state)

        logger.info(
            f"ReAct starting ({len(tools)} tools, max_iters={MAX_ITERATIONS})"
        )

        answer = await self._react_loop(system, messages, tools, state, on_status)

        # Sync per-turn state back to Brain
        self._pending_files = state.pending_files
        self._current_plan = state.current_plan

        # ask_user was called — surface the question and stop
        if state.clarification:
            messages.append(Message(role="assistant", content=state.clarification))
            logger.info("Returning clarification question to user.")
            return state.clarification, messages

        # reload was called — signal the main process to restart cleanly
        if state.restart_requested:
            final_text = _clean_response(answer or "Restarting…")
            messages.append(Message(role="assistant", content=final_text))
            logger.info("Reload confirmed — sending SIGTERM for clean restart")

            import signal

            os.environ["_ATLAS_RESTART"] = "1"
            os.kill(os.getpid(), signal.SIGTERM)
            return final_text, messages

        final_text = _clean_response(answer or "✅ Done.")
        messages.append(Message(role="assistant", content=final_text))
        logger.info(f"Brain finished. Provider: {self.provider.model_name}")
        return final_text, messages

    def take_files(self) -> list[tuple[Path, str]]:
        """Return and clear queued files. Called by channels after think()."""
        files, self._pending_files = self._pending_files, []
        return files

    async def extract(self, text: str, schema: dict, instruction: str = "") -> dict:
        """
        Extract structured data from text as JSON.

        Bypasses the ReAct loop; calls the provider directly with json_mode=True.
        """
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
            max_tokens=MAX_TOKENS,
            json_mode=True,
        )
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"Model returned non-JSON output: {raw[:200]}") from exc

    @property
    def session_tokens(self) -> dict[str, int]:
        """Cumulative token usage for the current session."""
        return dict(self._session_tokens)

    def reset_session_tokens(self) -> None:
        self._session_tokens = {"input": 0, "output": 0}
