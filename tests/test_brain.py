"""
tests/test_brain.py

Tests for memory, history, and summariser — components that don't touch the
ReAct loop directly.  Brain-specific ReAct tests live in test_dspy_engine.py.

Also exports shared test helpers (MockProvider, make_brain, fixtures) used by
test_channels.py and test_heartbeat.py.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch

from providers.base import BaseLLMProvider, Message, LLMResponse
from memory.store import MemoryStore
from skills.registry import SkillRegistry
from brain.engine import Brain, _TurnState, _clean_response


# ── Minimal provider mock ─────────────────────────────────────────────────────


class MockProvider(BaseLLMProvider):
    """Controllable mock LLM provider used by summariser and extract() tests."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages,
        tools=None,
        system=None,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self._call_count >= len(self._responses):
            return LLMResponse(content="(no more mock responses)", tool_calls=[])
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_memory(tmp_path: Path) -> MemoryStore:
    """A MemoryStore backed by a temporary directory."""
    with patch("memory.store.MEMORY_DIR", tmp_path):
        store = MemoryStore()
    yield store


@pytest.fixture
def empty_skills(tmp_path: Path) -> SkillRegistry:
    """A SkillRegistry with no skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    with (
        patch("skills.registry.CORE_SKILLS_DIR", skills_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon_skills"),
    ):
        yield SkillRegistry()


# ── make_brain — test helper ────────────────────────────────────────────────


def make_brain(
    responses: list[LLMResponse],
    memory: MemoryStore,
    skills: SkillRegistry,
) -> tuple[Brain, MockProvider]:
    """
    Create a Brain for testing without real API calls.

    Replaces brain.think() with a simulated loop that:
      - Iterates through `responses` in order
      - For tool-call responses: executes the tool closure, appends tool messages
      - For the first text response: returns it as the final answer

    This preserves the old test API while working with the new engine.
    """

    provider = MockProvider(responses)
    brain = Brain(provider=provider, memory=memory, skills=skills)

    # Replace think() with a simulation that drives the tool closures directly,
    # reproducing the ReAct-loop behaviour without a real LLM.
    _responses = list(responses)

    async def simulated_think(
        user_message,
        conversation_history,
        on_status=None,
        system_suffix="",
    ):
        from brain.engine import _extract_text

        state = _TurnState()
        tools = brain._build_tools(state)
        tool_map = {t.name: t for t in tools}

        text = (
            _extract_text(user_message)
            if isinstance(user_message, list)
            else user_message
        )
        messages = list(conversation_history) + [Message(role="user", content=text)]

        for resp in _responses:
            if resp.tool_calls:
                messages.append(
                    Message(
                        role="assistant",
                        content=resp.content or "",
                        tool_calls=[
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in resp.tool_calls
                        ],
                    )
                )
                for tc in resp.tool_calls:
                    tool = tool_map.get(tc.name)
                    if tool is not None:
                        try:
                            result = tool(**tc.arguments)
                            if asyncio.iscoroutine(result):
                                result = await result
                        except Exception as e:
                            result = f"Error in {tc.name}: {e}"
                    elif tc.name.startswith("skill_"):
                        skill = skills.get_skill(tc.name[len("skill_") :])
                        result = (
                            await skill.run(**tc.arguments)
                            if skill
                            else f"Unknown skill: {tc.name}"
                        )
                    else:
                        result = f"Unknown tool: {tc.name}"
                    messages.append(
                        Message(role="tool", content=str(result), tool_call_id=tc.id)
                    )

                # ask_user or reload — stop immediately
                if state.clarification:
                    messages.append(
                        Message(role="assistant", content=state.clarification)
                    )
                    brain._pending_files = state.pending_files
                    brain._current_plan = state.current_plan
                    return state.clarification, messages
                if state.restart_requested:
                    brain._pending_files = state.pending_files
                    brain._current_plan = state.current_plan
                    return "Restarting…", messages
            else:
                final = _clean_response(resp.content or "✅ Done.")
                brain._pending_files = state.pending_files
                brain._current_plan = state.current_plan
                messages.append(Message(role="assistant", content=final))
                return final, messages

        # Fallback if responses list ran out
        fallback = "✅ Done."
        messages.append(Message(role="assistant", content=fallback))
        return fallback, messages

    brain.think = simulated_think
    return brain, provider


# ── Memory tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_relevance_filtering_in_system_prompt(tmp_memory):
    """build_system_prompt passes the query so only relevant entries are injected."""
    from memory.store import CONTEXT_MAX_INJECTED

    for i in range(CONTEXT_MAX_INJECTED + 5):
        tmp_memory.append_context(f"Generic fact number {i} about something unrelated.")
    tmp_memory.append_context("User's favourite colour is electric blue.")

    system = await tmp_memory.build_system_prompt(
        query="What is the user's favourite colour?"
    )
    assert "electric blue" in system.lower()


@pytest.mark.asyncio
async def test_context_entry_parsing(tmp_memory):
    """parse_context_entries correctly splits context.md into dated entries."""
    tmp_memory.append_context("User likes hiking.")
    tmp_memory.append_context("User has a dog named Max.")

    entries = tmp_memory.parse_context_entries()
    assert len(entries) == 2
    assert any("hiking" in e.content for e in entries)
    assert any("Max" in e.content for e in entries)
    assert all(e.timestamp for e in entries)


# ── History tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persistent_history_survives_reload(tmp_path):
    """ConversationHistory saves and loads correctly from disk."""
    from memory.history import ConversationHistory

    with patch("memory.history.HISTORY_DIR", tmp_path / "history"):
        hist = ConversationHistory()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        hist.save("user_42", messages)
        loaded = hist.load("user_42")

    assert len(loaded) == 2
    assert loaded[0].role == "user"
    assert loaded[0].content == "Hello"
    assert loaded[1].role == "assistant"


@pytest.mark.asyncio
async def test_persistent_history_clear(tmp_path):
    """ConversationHistory.clear() removes the saved file."""
    from memory.history import ConversationHistory

    with patch("memory.history.HISTORY_DIR", tmp_path / "history"):
        hist = ConversationHistory()
        hist.save("user_99", [Message(role="user", content="test")])
        hist.clear("user_99")
        loaded = hist.load("user_99")

    assert loaded == []


# ── Summariser tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summariser_compresses_old_entries(tmp_memory):
    """Summariser compresses old entries when count exceeds threshold."""
    from memory.summariser import maybe_summarise, SUMMARY_THRESHOLD

    for i in range(SUMMARY_THRESHOLD + 2):
        tmp_memory.append_context(f"Memory entry {i}: user did something on day {i}.")

    entries_before = len(tmp_memory.parse_context_entries())
    assert entries_before > SUMMARY_THRESHOLD

    provider = MockProvider(
        [
            LLMResponse(
                content="User did various things across many days.", tool_calls=[]
            )
        ]
    )
    result = await maybe_summarise(tmp_memory, provider)

    assert result is True
    entries_after = tmp_memory.parse_context_entries()
    assert len(entries_after) < entries_before
    background = [e for e in entries_after if not e.timestamp]
    assert len(background) == 1
    assert "various" in background[0].content
