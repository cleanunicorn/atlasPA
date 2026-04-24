"""
brain/engine.py

The Brain — implements a ReAct (Reason + Act) loop using DSPy.
"""

import json as _json
import logging
import os
import re
import asyncio
import datetime
from pathlib import Path
from collections.abc import Callable, Awaitable

import dspy
from brain.dspy_adapter import AtlasLM, brain_tool_to_dspy, AtlasSignature, ToolSelectionSignature
from providers.base import BaseLLMProvider, Message
from memory.store import MemoryStore
from skills.registry import SkillRegistry

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
    _make_update_self,
    _make_request_skills,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 30
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
TOOL_SELECTION_THRESHOLD = 5
TRACE_DIR = Path("logs/traces")

# Tools that are always available regardless of selection
_ALWAYS_INCLUDED_TOOLS = frozenset({
    "ask_user",
    "request_skills",
})

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


# ── Brain ────────────────────────────────────────────────────────────────────


class Brain:
    """
    ReAct reasoning engine using DSPy.
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
        self._selected_tools: list[str] | None = None  # cached tool/skill selection

        # DSPy setup
        self.lm = AtlasLM(provider)

        if not TRACE_DIR.exists():
            TRACE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Brain ready (DSPy): {provider.model_name}")

    # ── Tool construction ────────────────────────────────────────────────────

    def _all_candidate_tools(
        self, state: _TurnState
    ) -> list[BrainTool]:
        """Build every possible tool (built-ins + all skills). Unfiltered."""
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
            _make_update_self(self),
            _make_request_skills(state, self.skills),
        ]
        if os.getenv("CLAUDE_CODE_AVAILABLE", "").lower() == "true":
            tools.append(_make_run_claude())
        for skill in self.skills._skills.values():
            try:
                tools.append(_make_skill_tool(skill))
            except Exception as e:
                logger.warning(f"Could not wrap skill '{skill.name}': {e}")
        return tools

    def _build_tools(
        self, state: _TurnState, selected_tools: list[str]
    ) -> list[BrainTool]:
        """Build the filtered tool list for one turn."""
        selected = set(selected_tools) | _ALWAYS_INCLUDED_TOOLS
        return [
            t for t in self._all_candidate_tools(state)
            if t.name in selected
        ]

    # ── Tool selection ──────────────────────────────────────────────────────

    def _tool_catalog(self) -> tuple[list[str], str]:
        """Build a catalog of all available tools (names + descriptions).

        Returns (all_names, catalog_text) where catalog_text is a compact
        summary suitable for the selection LLM call.
        """
        state = _TurnState()
        candidates = self._all_candidate_tools(state)
        # Exclude always-included tools from the catalog — they're implicit
        filterable = [t for t in candidates if t.name not in _ALWAYS_INCLUDED_TOOLS]
        all_names = [t.name for t in filterable]
        lines = []
        for t in filterable:
            # Truncate long descriptions to keep the prompt lean
            desc = (t.description or "").split("\n")[0][:200]
            lines.append(f"  - **{t.name}**: {desc}")
        return all_names, "\n".join(lines)

    async def _select_tools(self, query_text: str) -> list[str]:
        """Pick relevant tools and skills for a query using DSPy."""
        all_names, catalog = self._tool_catalog()
        if not all_names:
            return []

        if len(all_names) < TOOL_SELECTION_THRESHOLD:
            logger.debug(
                f"Skipping tool selection ({len(all_names)} < {TOOL_SELECTION_THRESHOLD})"
            )
            return all_names

        try:
            predictor = dspy.Predict(ToolSelectionSignature)
            # Use JSONAdapter for structured output of tools list
            with dspy.settings.context(lm=self.lm, adapter=dspy.JSONAdapter()):
                result = await predictor.aforward(request=query_text, catalog=catalog)
                selected = result.tools

            if not isinstance(selected, list):
                # Fallback if LM returns string representation of list
                if isinstance(selected, str):
                    try:
                        selected = _json.loads(selected)
                    except Exception:
                        selected = []
                else:
                    selected = []

            valid = set(all_names)
            filtered = [s for s in selected if s in valid]
            logger.info(f"Tool selector chose {len(filtered)}/{len(all_names)}: {filtered}")
            return filtered
        except Exception as e:
            logger.warning(f"Tool selection failed, using all tools: {e}")
            return all_names

    # ── ReAct loop ──────────────────────────────────────────────────────────

    def _save_trace(self, lm: AtlasLM):
        """Save DSPy LM history as a trace for future compilation."""
        if not lm.history:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_file = TRACE_DIR / f"trace_{timestamp}.json"

        try:
            with open(trace_file, "w") as f:
                _json.dump(lm.history, f, indent=2)
            logger.debug(f"Saved DSPy trace to {trace_file}")
        except Exception as e:
            logger.warning(f"Could not save DSPy trace: {e}")

    async def _run_react(
        self,
        system_prompt: str,
        history_str: str,
        query_text: str,
        state: _TurnState,
        selected_tools: list[str],
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Run the ReAct loop using DSPy.
        Supports hot-loading skills mid-loop while preserving trajectory.
        """
        trajectory = []

        for _ in range(MAX_ITERATIONS):
            tools = self._build_tools(state, selected_tools)
            dspy_tools = [brain_tool_to_dspy(t, on_status) for t in tools]

            with dspy.settings.context(lm=self.lm, adapter=dspy.ChatAdapter()):
                react = dspy.ReAct(AtlasSignature, tools=dspy_tools, max_iters=MAX_ITERATIONS)

                # To preserve trajectory, we'd ideally pass it to ReAct.
                # Since dspy.ReAct doesn't expose it, we simulate it by updating the 'history' field.
                combined_history = history_str
                if trajectory:
                    combined_history += "\n--- PREVIOUS STEPS ---\n" + "\n".join(trajectory)

                prediction = await react.aforward(
                    context=system_prompt,
                    history=combined_history,
                    question=query_text
                )

            # ask_user stops the loop early
            if state.clarification or state.restart_requested:
                 self._save_trace(self.lm)
                 return prediction.answer or ""

            # Check if skills were requested
            if state.requested_skills:
                new_skills = []
                for skill_name in state.requested_skills:
                    name = f"skill_{skill_name}"
                    if name not in selected_tools:
                        selected_tools.append(name)
                        new_skills.append(skill_name)
                state.requested_skills.clear()

                if new_skills:
                    logger.info(f"Hot-loading skills: {new_skills}. Re-running ReAct loop.")
                    if hasattr(prediction, 'trajectory'):
                         trajectory = prediction.trajectory
                    continue

            self._save_trace(self.lm)
            return prediction.answer

        self._save_trace(self.lm)
        return "Loop limit reached."

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
        """
        self._pending_files = []
        self._current_plan = None
        state = _TurnState()

        query_text = _extract_text(user_message)

        # Choose the right list of tools for this query
        self._selected_tools = await self._select_tools(query_text)

        # Build system prompt with only the selected skills
        selected_skill_names = [
            n for n in self._selected_tools
            if n.startswith("skill_")
        ]
        skill_names = [n[len("skill_"):] for n in selected_skill_names]
        filtered_summary = self.skills.get_skills_summary(only=skill_names)

        system_prompt = await self.memory.build_system_prompt(
            filtered_summary, query=query_text
        )
        if system_suffix:
            system_prompt += "\n\n---\n" + system_suffix

        history_str = ""
        for m in conversation_history:
            history_str += f"{m.role}: {m.content}\n"

        try:
            answer = await self._run_react(
                system_prompt, history_str, query_text,
                state, self._selected_tools, on_status
            )
        except Exception as e:
            logger.error(f"DSPy ReAct failed: {e}")
            answer = f"I encountered an error while thinking: {e}"

        self._pending_files = state.pending_files
        self._current_plan = state.current_plan

        messages = list(conversation_history) + [
            Message(role="user", content=user_message)
        ]

        if state.clarification:
            messages.append(Message(role="assistant", content=state.clarification))
            logger.info("Returning clarification question to user.")
            return state.clarification, messages

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
        Extract structured data from text using DSPy.
        """
        class ExtractionSignature(dspy.Signature):
            """Extract structured data."""
            text = dspy.InputField()
            instruction = dspy.InputField()
            output = dspy.OutputField(desc="JSON data matching requested schema")

        try:
            with dspy.settings.context(lm=self.lm, adapter=dspy.JSONAdapter()):
                predictor = dspy.Predict(ExtractionSignature)
                full_instruction = f"{instruction}\n\nReturn ONLY a valid JSON object matching this schema: {_json.dumps(schema)}"
                result = await predictor.aforward(text=text, instruction=full_instruction)
                raw = result.output
                self._save_trace(self.lm)
        except Exception as e:
            logger.warning(f"DSPy extraction failed, falling back: {e}")
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
            raw = response.content or ""

        raw = raw.strip()
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
