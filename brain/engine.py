"""
brain/engine.py

The Brain — implements a ReAct (Reason + Act) loop using DSPy.
"""

import json as _json
import logging
import os
import re
import datetime
from pathlib import Path
from collections.abc import Callable, Awaitable

import dspy
from brain.dspy_adapter import AtlasLM, brain_tool_to_dspy, AtlasSignature
from brain.compactor import maybe_compact_history, estimate_tokens
from providers.base import BaseLLMProvider, Message
from memory.store import MemoryStore
from memory.summariser import maybe_summarise_history
from skills.registry import SkillRegistry
from brain.tools import (
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
    _make_update_self,
    _make_request_skills,
    _TurnState,
    BrainTool,
)

logger = logging.getLogger(__name__)

# ReAct parameters
MAX_ITERATIONS = 8
TRACE_DIR = Path("logs/traces")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# ── Internals ───────────────────────────────────────────────────────────────


def _extract_text(content: str | list) -> str:
    """Extract plain text from multimodal content blocks."""
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if b.get("type") == "text")


def _clean_response(text: str) -> str:
    """Strip 'Answer:' prefix or similar if DSPy leaves it in."""
    text = re.sub(r"^(Answer|Response):\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


class Brain:
    """
    Atlas reasoning engine. Handles tool selection and the ReAct loop.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        self.provider = provider
        self.memory = memory
        self.skills = skills
        self.lm = AtlasLM(provider)
        self.heartbeat = None  # Injected by main.py

        self._pending_files: list[tuple[Path, str]] = []
        self._current_plan: str | None = None
        self._session_tokens = {"input": 0, "output": 0}

    def _build_tools(self, state: _TurnState, selected_tools: list[str]) -> list:
        """
        Build the list of tool instances for the current turn.
        Only includes tools explicitly selected or requested.
        """
        all_tools = self._create_tools(state)

        # Filter to only the selected ones
        return [t for t in all_tools if t.name in selected_tools]

    def _create_tools(
        self, state: _TurnState
    ) -> list[BrainTool]:
        """Build every possible tool (built-ins + all skills). Unfiltered."""
        tools = [
            _make_remember(self.memory, self.provider),
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

        # Add addon skills
        for skill_name in self.skills.all_skill_names():
            tools.append(self.skills.get_skill_as_tool(skill_name))

        return tools

    def _tool_catalog(self) -> tuple[list[str], str]:
        """Get names and descriptions of all available tools."""
        state = _TurnState()
        tools = self._create_tools(state)

        names = [t.name for t in tools]
        catalog_lines = [f"- {t.name}: {t.description}" for t in tools]

        return names, "\n".join(catalog_lines)

    async def _select_tools(self, query_text: str) -> list[str]:
        """
        Ask the LLM to pick which tools are relevant for the current query.
        This keeps the context window clean by only injecting relevant skill docs.
        """
        all_names, catalog = self._tool_catalog()

        system = (
            "You are a tool router. Given the user's request and a list of "
            "available tools, return a JSON object with a single key \"tools\" "
            "containing an array of tool names that are relevant to fulfilling "
            "the request. Include only tools that are directly useful. "
            "If none are relevant, return an empty array."
        )
        user_content = (
            f"User request: {query_text}\n\n"
            f"Available tools:\n{catalog}"
        )

        try:
            response = await self.provider.complete(
                messages=[Message(role="user", content=user_content)],
                system=system,
                max_tokens=2048,
                json_mode=True,
            )
            raw = (response.content or "").strip()
            # Extract JSON from markdown fences, even when surrounded by text
            fence_match = re.search(r"```[a-z]*\n?(.*?)\n?```", raw, re.DOTALL)
            if fence_match:
                raw = fence_match.group(1).strip()
            logger.debug(f"Tool selector raw response: {raw}")
            data = _json.loads(raw)
            selected = data.get("tools", [])
            if not isinstance(selected, list):
                raise ValueError(f"Expected list, got {type(selected)}")
            # Validate against actual tool names
            valid = set(all_names)
            filtered = [s for s in selected if s in valid]
            dropped = [s for s in selected if s not in valid]
            if dropped:
                logger.warning(f"Tool selector returned unknown tools: {dropped}")
            logger.info(
                f"Tool selector chose {len(filtered)}/{len(all_names)}: {filtered}"
            )
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

        for i in range(MAX_ITERATIONS):
            tools = self._build_tools(state, selected_tools)
            dspy_tools = [brain_tool_to_dspy(t, on_status) for t in tools]

            with dspy.settings.context(lm=self.lm, adapter=dspy.ChatAdapter()):
                react = dspy.ReAct(AtlasSignature, tools=dspy_tools, max_iters=MAX_ITERATIONS)

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

        conversation_history, _ = await maybe_compact_history(
            conversation_history,
            self.provider,
            system_prompt_tokens=estimate_tokens(system_prompt),
            query_tokens=estimate_tokens(query_text),
        )

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
                max_tokens=4096,
                json_mode=True,
            )
            raw = response.content or ""

        raw = raw.strip()
        if raw.startswith("```"):
            fence_match = re.search(r"```[a-z]*\n?(.*?)\n?```", raw, re.DOTALL)
            if fence_match:
                raw = fence_match.group(1).strip()
            else:
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
