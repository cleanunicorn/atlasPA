"""
tests/test_brain.py

Integration tests for the Brain (ReAct loop).

Tests use a mock LLM provider to avoid real API calls.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from providers.base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse
from brain.engine import Brain
from memory.store import MemoryStore
from skills.registry import SkillRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────────


class MockProvider(BaseLLMProvider):
    """
    A controllable mock LLM provider.

    Configure responses via `responses` — a list of LLMResponse objects
    that are returned in order on each call to `complete()`.
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[dict] = []  # recorded calls for assertions

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
    """A MemoryStore backed by temporary files that are cleaned up after the test."""
    # Patch MEMORY_DIR to point at tmp_path
    with patch("memory.store.MEMORY_DIR", tmp_path):
        store = MemoryStore()
    return store


@pytest.fixture
def empty_skills() -> SkillRegistry:
    """A SkillRegistry with no skills (empty temp directory)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("skills.registry.SKILLS_DIR", Path(tmp_dir)):
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


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_turn_no_tools(tmp_memory, empty_skills):
    """Brain returns a plain text response without any tool calls."""
    responses = [
        LLMResponse(content="Hello! How can I help you today?", tool_calls=[])
    ]
    brain, provider = make_brain(responses, tmp_memory, empty_skills)

    response, history = await brain.think("Hi", conversation_history=[])

    assert response == "Hello! How can I help you today?"
    assert len(history) == 2  # user + assistant
    assert history[0].role == "user"
    assert history[0].content == "Hi"
    assert history[1].role == "assistant"
    assert history[1].content == "Hello! How can I help you today?"
    assert provider._call_count == 1


@pytest.mark.asyncio
async def test_multi_turn_conversation(tmp_memory, empty_skills):
    """Brain correctly carries conversation history across multiple calls to think()."""
    brain, provider = make_brain(
        responses=[
            LLMResponse(content="Nice to meet you, Alice!", tool_calls=[]),
            LLMResponse(content="You told me your name is Alice.", tool_calls=[]),
        ],
        memory=tmp_memory,
        skills=empty_skills,
    )

    _, history = await brain.think("My name is Alice", conversation_history=[])
    response, history = await brain.think("What's my name?", conversation_history=history)

    assert "Alice" in response
    # History: user, assistant, user, assistant
    assert len(history) == 4
    assert history[0].content == "My name is Alice"
    assert history[2].content == "What's my name?"


@pytest.mark.asyncio
async def test_tool_call_then_final_response(tmp_memory, empty_skills):
    """
    Brain executes a tool call and feeds the result back to the LLM,
    which then returns a final response.
    """
    # Create a mock skill
    mock_skill = MagicMock()
    mock_skill.run = AsyncMock(return_value="The capital of France is Paris.")
    empty_skills._skills["geo"] = mock_skill

    responses = [
        # First call: LLM requests a tool
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc_001", name="skill_geo", arguments={"query": "capital of France"})],
            stop_reason="tool_use",
        ),
        # Second call: LLM synthesises the tool result
        LLMResponse(content="The capital of France is Paris.", tool_calls=[]),
    ]
    brain, provider = make_brain(responses, tmp_memory, empty_skills)

    response, history = await brain.think("What is the capital of France?", conversation_history=[])

    assert "Paris" in response
    assert provider._call_count == 2

    # Verify tool result was injected into messages
    roles = [m.role for m in history]
    assert "tool" in roles

    # Verify the skill was called with the correct arguments
    mock_skill.run.assert_called_once_with(query="capital of France")


@pytest.mark.asyncio
async def test_remember_tool_writes_to_context(tmp_memory, empty_skills):
    """
    When the LLM calls the built-in `remember` tool, the note is appended
    to memory/context.md.
    """
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc_002", name="remember", arguments={"note": "User likes Python."})],
            stop_reason="tool_use",
        ),
        LLMResponse(content="Got it, I'll remember that!", tool_calls=[]),
    ]
    brain, provider = make_brain(responses, tmp_memory, empty_skills)

    response, history = await brain.think("I love Python.", conversation_history=[])

    # Check context.md was updated
    context = tmp_memory.load_context()
    assert "User likes Python." in context

    assert "Got it" in response
    assert provider._call_count == 2


@pytest.mark.asyncio
async def test_max_iterations_safety_cap(tmp_memory, empty_skills):
    """
    If the LLM keeps returning tool calls beyond MAX_ITERATIONS,
    the Brain returns a fallback message and doesn't loop forever.
    """
    from brain.engine import MAX_ITERATIONS

    # Always return a tool call — never a final text response
    infinite_tool_call = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc_loop", name="remember", arguments={"note": "loop"})],
        stop_reason="tool_use",
    )

    brain, provider = make_brain(
        responses=[infinite_tool_call] * (MAX_ITERATIONS + 5),
        memory=tmp_memory,
        skills=empty_skills,
    )

    response, history = await brain.think("Loop forever!", conversation_history=[])

    assert "loop" in response.lower() or "stuck" in response.lower()
    assert provider._call_count == MAX_ITERATIONS


@pytest.mark.asyncio
async def test_empty_response_handled(tmp_memory, empty_skills):
    """Brain handles an LLM response with no content and no tool calls gracefully."""
    responses = [LLMResponse(content=None, tool_calls=[])]
    brain, _ = make_brain(responses, tmp_memory, empty_skills)

    response, history = await brain.think("Hello", conversation_history=[])

    assert response == "(no response)"


@pytest.mark.asyncio
async def test_tool_execution_error_surfaces_in_result(tmp_memory, empty_skills):
    """
    If a skill raises an exception, the error message is returned as the tool
    result (not propagated as an unhandled exception).
    """
    mock_skill = MagicMock()
    mock_skill.run = AsyncMock(return_value="Error running skill 'broken': Something went wrong")
    empty_skills._skills["broken"] = mock_skill

    responses = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc_err", name="skill_broken", arguments={})],
            stop_reason="tool_use",
        ),
        LLMResponse(content="The skill encountered an error.", tool_calls=[]),
    ]
    brain, provider = make_brain(responses, tmp_memory, empty_skills)

    # Should not raise
    response, history = await brain.think("Run the broken skill", conversation_history=[])

    assert provider._call_count == 2
    # The error should appear in the tool result message
    tool_msgs = [m for m in history if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Error" in tool_msgs[0].content
