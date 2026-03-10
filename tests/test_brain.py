"""
tests/test_brain.py

Integration tests for the Brain (ReAct loop) — Phase 1 & 2.

All tests use a mock LLM provider to avoid real API calls.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from providers.base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse
from brain.engine import Brain
from memory.store import MemoryStore
from skills.registry import SkillRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────────


class MockProvider(BaseLLMProvider):
    """
    Controllable mock LLM provider.
    Configure `responses` — returned in order on each call to complete().
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self._call_count >= len(self._responses):
            return LLMResponse(content="(no more mock responses)", tool_calls=[])
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


@pytest.fixture
def tmp_memory(tmp_path: Path) -> MemoryStore:
    """A MemoryStore backed by a temporary directory."""
    with patch("memory.store.MEMORY_DIR", tmp_path):
        store = MemoryStore()
    return store


@pytest.fixture
def empty_skills(tmp_path: Path) -> SkillRegistry:
    """A SkillRegistry with no skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    with (
        patch("skills.registry.CORE_SKILLS_DIR", skills_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon_skills"),
    ):
        registry = SkillRegistry()
    return registry


def make_brain(
    responses: list[LLMResponse],
    memory: MemoryStore,
    skills: SkillRegistry,
) -> tuple[Brain, MockProvider]:
    provider = MockProvider(responses)
    brain = Brain(provider=provider, memory=memory, skills=skills)
    return brain, provider


# ── Phase 1 tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_turn_no_tools(tmp_memory, empty_skills):
    """Brain returns a plain text response without any tool calls."""
    brain, provider = make_brain(
        [LLMResponse(content="Hello! How can I help you today?", tool_calls=[])],
        tmp_memory, empty_skills,
    )
    response, history = await brain.think("Hi", conversation_history=[])

    assert response == "Hello! How can I help you today?"
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert provider._call_count == 1


@pytest.mark.asyncio
async def test_multi_turn_conversation(tmp_memory, empty_skills):
    """Brain carries conversation history correctly across multiple think() calls."""
    brain, _ = make_brain(
        [
            LLMResponse(content="Nice to meet you, Alice!", tool_calls=[]),
            LLMResponse(content="You told me your name is Alice.", tool_calls=[]),
        ],
        tmp_memory, empty_skills,
    )
    _, history = await brain.think("My name is Alice", conversation_history=[])
    response, history = await brain.think("What's my name?", conversation_history=history)

    assert "Alice" in response
    assert len(history) == 4
    assert history[0].content == "My name is Alice"
    assert history[2].content == "What's my name?"


@pytest.mark.asyncio
async def test_tool_call_then_final_response(tmp_memory, empty_skills):
    """Brain executes a tool call and feeds the result back to the LLM."""
    mock_skill = MagicMock()
    mock_skill.run = AsyncMock(return_value="The capital of France is Paris.")
    empty_skills._skills["geo"] = mock_skill

    brain, provider = make_brain(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_001", name="skill_geo", arguments={"query": "capital of France"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="The capital of France is Paris.", tool_calls=[]),
        ],
        tmp_memory, empty_skills,
    )
    response, history = await brain.think("What is the capital of France?", conversation_history=[])

    assert "Paris" in response
    assert provider._call_count == 2
    assert "tool" in [m.role for m in history]
    mock_skill.run.assert_called_once_with(query="capital of France")


@pytest.mark.asyncio
async def test_remember_tool_writes_to_context(tmp_memory, empty_skills):
    """The `remember` tool appends a note to memory/context.md."""
    brain, provider = make_brain(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_002", name="remember", arguments={"note": "User likes Python."})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Got it, I'll remember that!", tool_calls=[]),
        ],
        tmp_memory, empty_skills,
    )
    response, _ = await brain.think("I love Python.", conversation_history=[])

    assert "User likes Python." in tmp_memory.load_context()
    assert "Got it" in response
    assert provider._call_count == 2


@pytest.mark.asyncio
async def test_max_iterations_safety_cap(tmp_memory, empty_skills):
    """Brain returns a fallback message when MAX_ITERATIONS is reached."""
    from brain.engine import MAX_ITERATIONS

    infinite = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc_loop", name="remember", arguments={"note": "loop"})],
        stop_reason="tool_use",
    )
    brain, provider = make_brain(
        [infinite] * (MAX_ITERATIONS + 5), tmp_memory, empty_skills,
    )
    response, _ = await brain.think("Loop forever!", conversation_history=[])

    assert "loop" in response.lower() or "stuck" in response.lower()
    assert provider._call_count == MAX_ITERATIONS


@pytest.mark.asyncio
async def test_empty_response_handled(tmp_memory, empty_skills):
    """Brain handles an LLM response with no content and no tool calls."""
    brain, _ = make_brain([LLMResponse(content=None, tool_calls=[])], tmp_memory, empty_skills)
    response, _ = await brain.think("Hello", conversation_history=[])
    assert response == "(no response)"


@pytest.mark.asyncio
async def test_tool_execution_error_surfaces_in_result(tmp_memory, empty_skills):
    """A skill error is returned as the tool result, not as an unhandled exception."""
    mock_skill = MagicMock()
    mock_skill.run = AsyncMock(return_value="Error running skill 'broken': Something went wrong")
    empty_skills._skills["broken"] = mock_skill

    brain, provider = make_brain(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_err", name="skill_broken", arguments={})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="The skill encountered an error.", tool_calls=[]),
        ],
        tmp_memory, empty_skills,
    )
    response, history = await brain.think("Run the broken skill", conversation_history=[])

    assert provider._call_count == 2
    tool_msgs = [m for m in history if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Error" in tool_msgs[0].content


# ── Phase 2 tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_tool_removes_entry(tmp_memory, empty_skills):
    """The `forget` tool removes a matching entry from context.md."""
    # Seed context with an entry
    tmp_memory.append_context("User prefers dark mode in all apps.")

    brain, provider = make_brain(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc_forget", name="forget", arguments={"note": "dark mode"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="I've forgotten that about you.", tool_calls=[]),
        ],
        tmp_memory, empty_skills,
    )
    response, history = await brain.think("Forget the dark mode preference.", conversation_history=[])

    assert "dark mode" not in tmp_memory.load_context()
    assert provider._call_count == 2
    tool_msgs = [m for m in history if m.role == "tool"]
    assert "Forgot" in tool_msgs[0].content or "forgot" in tool_msgs[0].content.lower()


@pytest.mark.asyncio
async def test_relevance_filtering_in_system_prompt(tmp_memory, empty_skills):
    """build_system_prompt passes the query so only relevant entries are injected."""
    # Add many entries to exceed CONTEXT_MAX_INJECTED
    from memory.store import CONTEXT_MAX_INJECTED

    for i in range(CONTEXT_MAX_INJECTED + 5):
        tmp_memory.append_context(f"Generic fact number {i} about something unrelated.")
    tmp_memory.append_context("User's favourite colour is electric blue.")

    brain, _ = make_brain(
        [LLMResponse(content="Electric blue!", tool_calls=[])],
        tmp_memory, empty_skills,
    )
    system = tmp_memory.build_system_prompt(query="What is the user's favourite colour?")

    # The relevant entry should be included
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


@pytest.mark.asyncio
async def test_summariser_compresses_old_entries(tmp_memory, empty_skills):
    """Summariser compresses old entries when count exceeds threshold."""
    from memory.summariser import maybe_summarise, SUMMARY_THRESHOLD

    # Add entries beyond the threshold
    for i in range(SUMMARY_THRESHOLD + 2):
        tmp_memory.append_context(f"Memory entry {i}: user did something on day {i}.")

    entries_before = len(tmp_memory.parse_context_entries())
    assert entries_before > SUMMARY_THRESHOLD

    # Summariser LLM call returns a compact summary
    provider = MockProvider([LLMResponse(content="User did various things across many days.", tool_calls=[])])
    result = await maybe_summarise(tmp_memory, provider)

    assert result is True
    entries_after = tmp_memory.parse_context_entries()
    # After summarisation: 1 background + remaining recent entries
    assert len(entries_after) < entries_before
    # The background entry contains the summary
    background = [e for e in entries_after if not e.timestamp]
    assert len(background) == 1
    assert "various" in background[0].content
