"""
brain/engine.py

The Brain — implements the ReAct (Reason + Act) loop via DSPy.

DSPy's dspy.ReAct module handles the reasoning/acting cycle. Atlas skills
and built-in tools are wrapped as dspy.Tool instances and passed in each turn.

Key design notes:
  - DSPy is synchronous; think() bridges via asyncio.to_thread().
  - Async skill run() functions are bridged with a fresh event loop per call.
  - Built-in tools close over a per-turn _TurnState to propagate side-effects
    (file queuing, clarifications, plan tracking) back to the async caller.
  - on_token streaming is not supported by DSPy; the parameter is accepted for
    interface compatibility but never called.
"""

import asyncio
import json as _json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from collections.abc import Callable, Awaitable

import dspy
from dspy.utils.exceptions import AdapterParseError

from providers.base import BaseLLMProvider, Message
from memory.store import MemoryStore
from skills.registry import SkillRegistry

from brain.status import AtlasCallback, drain_status
from brain.tools import (  # noqa: F401 — re-exported for tests
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
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 30
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
_RESTART_DELAY = 2.0  # seconds before os.execv

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


# ── Resilient ReAct subclass ─────────────────────────────────────────────────


class AtlasReAct(dspy.ReAct):
    """ReAct subclass that gracefully handles AdapterParseError.

    DSPy's ReAct.forward() only catches ValueError when the LLM produces a
    malformed response.  AdapterParseError (raised when the LLM outputs raw
    tool args instead of the expected {next_thought, next_tool_name,
    next_tool_args} structure) extends Exception directly, so it crashes the
    entire loop.  This subclass widens the catch so the loop can break
    gracefully and still produce a final answer from the accumulated trajectory.

    The extract step (final answer generation) can also fail with
    AdapterParseError when the LLM returns a plain-text answer instead of
    the expected JSON {reasoning, answer}.  In that case the raw LLM text
    is used directly as the answer.
    """

    def forward(self, **input_args):
        from dspy.predict.react import _fmt_exc

        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
            except (ValueError, AdapterParseError) as err:
                logger.warning(
                    "Ending the trajectory: Agent failed to select a valid "
                    f"tool: {_fmt_exc(err)}"
                )
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[
                    pred.next_tool_name
                ](**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = (
                    f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"
                )

            if pred.next_tool_name == "finish":
                break

        try:
            extract = self._call_with_potential_trajectory_truncation(
                self.extract, trajectory, **input_args
            )
            return dspy.Prediction(trajectory=trajectory, **extract)
        except AdapterParseError as err:
            logger.warning(
                "Extract step failed to parse LLM response as JSON; "
                "using raw LLM text as answer."
            )
            raw = err.lm_response or ""
            output_keys = list(self.signature.output_fields.keys())
            fallback = {k: "" for k in output_keys}
            fallback[output_keys[-1]] = raw
            return dspy.Prediction(trajectory=trajectory, **fallback)


# ── DSPy Signature ───────────────────────────────────────────────────────────


class AtlasSignature(dspy.Signature):
    """You are Atlas, a personal AI assistant. Reason carefully, use tools as
    needed, and give a helpful final answer.
    If ask_user returns '__ASK_USER__', immediately call finish and set the
    answer field to the clarifying question verbatim."""

    system_prompt: str = dspy.InputField(
        desc="Your identity, long-term memory about the user, available skills, and current time."
    )
    conversation_history: str = dspy.InputField(
        desc="Prior conversation context as 'Role: content' lines."
    )
    user_message: str = dspy.InputField(desc="The user's current message.")
    answer: str = dspy.OutputField(desc="Your final response to the user.")


# ── LM configuration ────────────────────────────────────────────────────────


def _build_dspy_lm() -> dspy.LM:
    """Map Atlas env vars to a DSPy LM (via LiteLLM)."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    max_tokens = MAX_TOKENS
    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return dspy.LM(f"anthropic/{model}", max_tokens=max_tokens)
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return dspy.LM(f"openai/{model}", max_tokens=max_tokens)
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return dspy.LM(f"ollama_chat/{model}", api_base=base, max_tokens=max_tokens)
    if provider == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
        return dspy.LM(f"openrouter/{model}", max_tokens=max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


# ── History serialisation ────────────────────────────────────────────────────


def _extract_text(content: str | list) -> str:
    """Extract plain text from a message content (handles multimodal blocks)."""
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if b.get("type") == "text")


def _serialize_history(messages: list[Message]) -> str:
    """Flatten conversation history to a plain string for DSPy context."""
    lines = []
    for msg in messages:
        if msg.role == "user":
            lines.append(f"User: {_extract_text(msg.content)}")
        elif msg.role == "assistant" and not msg.tool_calls:
            lines.append(f"Assistant: {msg.content or ''}")
        elif msg.role == "assistant" and msg.tool_calls:
            names = [tc["name"] for tc in (msg.tool_calls or [])]
            lines.append(f"Assistant: [used tools: {', '.join(names)}]")
        # tool result messages omitted — DSPy manages its own trajectory
    return "\n".join(lines) or "(no previous conversation)"


# ── Brain ────────────────────────────────────────────────────────────────────


class Brain:
    """
    DSPy-based ReAct reasoning engine.

    Wraps dspy.ReAct with Atlas's skill registry, memory, and built-in tools.
    Public interface is identical to the previous hand-rolled engine.
    """

    def __init__(
        self, provider: BaseLLMProvider, memory: MemoryStore, skills: SkillRegistry
    ):
        self.provider = provider  # kept for extract() which calls provider directly
        self.memory = memory
        self.skills = skills
        self._pending_files: list[tuple[Path, str]] = []
        self._current_plan: str | None = None
        self.heartbeat = None
        self._session_tokens: dict[str, int] = {"input": 0, "output": 0}

        self._status_callback = AtlasCallback()
        lm = _build_dspy_lm()
        dspy.configure(lm=lm, callbacks=[self._status_callback])
        logger.info(f"DSPy configured: {lm.model}")

    # ── Tool construction ────────────────────────────────────────────────────

    def _build_tools(self, state: _TurnState) -> list[dspy.Tool]:
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
        # Append skill tools (rebuilt each call to pick up newly installed addons)
        for skill in self.skills._skills.values():
            try:
                tools.append(_make_skill_tool(skill))
            except Exception as e:
                logger.warning(f"Could not wrap skill '{skill.name}' as DSPy tool: {e}")
        return tools

    # ── Public interface ─────────────────────────────────────────────────────

    async def think(
        self,
        user_message: str | list,
        conversation_history: list[Message],
        on_status: Callable[[str], Awaitable[None]] | None = None,
        system_suffix: str = "",
    ) -> tuple[str, list[Message]]:
        """
        Run the DSPy ReAct loop for a single user message.

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

        history_str = _serialize_history(conversation_history)
        messages = list(conversation_history) + [
            Message(role="user", content=user_message)
        ]

        tools = self._build_tools(state)
        react = AtlasReAct(AtlasSignature, tools=tools, max_iters=MAX_ITERATIONS)

        logger.info(
            f"DSPy ReAct starting ({len(tools)} tools, max_iters={MAX_ITERATIONS})"
        )

        # Set up status drain if channel wants progress updates
        drain_task = None
        if on_status:
            status_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
            self._status_callback.activate(status_queue, asyncio.get_running_loop())
            drain_task = asyncio.create_task(drain_status(status_queue, on_status))

        try:
            prediction = await asyncio.to_thread(
                react,
                system_prompt=system,
                conversation_history=history_str,
                user_message=query_text,
            )
        finally:
            if drain_task is not None:
                self._status_callback.deactivate()
                status_queue.put_nowait(None)  # sentinel to stop drain
                await drain_task

        # Sync per-turn state back to Brain
        self._pending_files = state.pending_files
        self._current_plan = state.current_plan

        # ask_user was called — surface the question and stop
        if state.clarification:
            messages.append(Message(role="assistant", content=state.clarification))
            logger.info("Returning clarification question to user.")
            return state.clarification, messages

        # reload was called — schedule restart after delivering response
        if state.restart_requested:
            final_text = _clean_response(prediction.answer or "Restarting…")
            messages.append(Message(role="assistant", content=final_text))
            logger.info("Reload confirmed — scheduling process restart")

            def _do_restart():
                import time

                time.sleep(_RESTART_DELAY)
                os.execv(sys.executable, [sys.executable] + sys.argv)

            threading.Thread(
                target=_do_restart, daemon=True, name="atlas-reload"
            ).start()
            return final_text, messages

        final_text = _clean_response(prediction.answer or "✅ Done.")
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
        """Cumulative token usage since last reset. Zero in DSPy mode."""
        return dict(self._session_tokens)

    def reset_session_tokens(self) -> None:
        self._session_tokens = {"input": 0, "output": 0}
