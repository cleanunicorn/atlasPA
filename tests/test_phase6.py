"""
tests/test_phase6.py

Tests for advanced reasoning features:
  - ask_user → clarification returned to caller
  - create_plan → plan stored on brain
  - reflect → gap detection
  - Brain.extract() → structured JSON output

Note: parallel tool execution was a feature of the old hand-rolled ReAct loop.
DSPy's ReAct runs tools sequentially; that behaviour is exercised in integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from providers.base import LLMResponse, ToolCall
from brain.engine import _make_reflect, _make_ask_user, _TurnState


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_skills(tmp_path):
    from skills.registry import SkillRegistry

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon"),
    ):
        (tmp_path / "core").mkdir()
        (tmp_path / "addon").mkdir()
        yield SkillRegistry()


@pytest.fixture
def tmp_memory(tmp_path):
    from memory.store import MemoryStore

    with patch("memory.store.MEMORY_DIR", tmp_path):
        yield MemoryStore()


@pytest.fixture
def brain(tmp_memory, empty_skills):
    """Brain with DSPy mocked out for tests that only call extract()."""
    import types
    from brain.engine import Brain

    mock_lm = types.SimpleNamespace(model="mock/model")
    with (
        patch("brain.engine._build_dspy_lm", return_value=mock_lm),
        patch("brain.engine.dspy.configure"),
    ):
        provider = MagicMock()
        provider.model_name = "mock"
        yield Brain(provider=provider, memory=tmp_memory, skills=empty_skills)


# ── ask_user / clarification ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_user_returns_question_immediately(tmp_memory, empty_skills):
    """When ask_user is called, think() stops and returns the question."""
    from tests.test_brain import make_brain

    brain, _ = make_brain(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="q1",
                        name="ask_user",
                        arguments={"question": "Which project?"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="fallback", tool_calls=[]),
        ],
        tmp_memory,
        empty_skills,
    )

    text, messages = await brain.think("do the thing", [])
    assert text == "Which project?"
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Which project?"


def test_ask_user_tool_sets_clarification():
    """ask_user tool closure sets state.clarification and returns sentinel."""
    state = _TurnState()
    tool = _make_ask_user(state)
    result = tool(question="Are you sure?")
    assert state.clarification == "Are you sure?"
    assert result == "__ASK_USER__"


# ── create_plan ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_plan_stores_plan(tmp_memory, empty_skills):
    """create_plan saves the plan to brain._current_plan."""
    from tests.test_brain import make_brain

    brain, _ = make_brain(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name="create_plan",
                        arguments={
                            "title": "My Plan",
                            "steps": ["Step 1", "Step 2", "Step 3"],
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Plan executed.", tool_calls=[]),
        ],
        tmp_memory,
        empty_skills,
    )

    await brain.think("complex task", [])
    assert brain._current_plan is not None
    assert "My Plan" in brain._current_plan
    assert "Step 1" in brain._current_plan
    assert "1." in brain._current_plan


@pytest.mark.asyncio
async def test_create_plan_result_included_in_tool_messages(tmp_memory, empty_skills):
    """The plan text is returned as the tool result message."""
    from tests.test_brain import make_brain

    brain, _ = make_brain(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name="create_plan",
                        arguments={
                            "title": "Task Plan",
                            "steps": ["Do A", "Do B"],
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="ok", tool_calls=[]),
        ],
        tmp_memory,
        empty_skills,
    )

    _, messages = await brain.think("task", [])
    tool_msgs = [m.content for m in messages if m.role == "tool"]
    assert any("Task Plan" in m for m in tool_msgs)


# ── reflect ───────────────────────────────────────────────────────────────────


def test_reflect_no_gaps():
    """reflect with 'none' gaps returns a 'proceed' message."""
    tool = _make_reflect()
    result = tool(goal="write a report", accomplished="wrote it", gaps="none")
    assert "proceed" in result.lower() or "done" in result.lower()


def test_reflect_with_gaps():
    """reflect with real gaps returns an 'address these' message."""
    tool = _make_reflect()
    result = tool(
        goal="write and send a report",
        accomplished="wrote report",
        gaps="still need to send the email",
    )
    assert "still need to send the email" in result


# ── Brain.extract() ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_parses_json(brain):
    """extract() returns a parsed dict from a JSON-mode provider response."""
    brain.provider.complete = AsyncMock(
        return_value=LLMResponse(
            content='{"name": "Alice", "age": 30}',
            stop_reason="end_turn",
        )
    )

    result = await brain.extract(
        text="Alice is 30 years old.",
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        },
    )
    assert result == {"name": "Alice", "age": 30}
    _, kwargs = brain.provider.complete.call_args
    assert kwargs.get("json_mode") is True


@pytest.mark.asyncio
async def test_extract_strips_markdown_fences(brain):
    """extract() handles models that wrap JSON in ```json fences."""
    brain.provider.complete = AsyncMock(
        return_value=LLMResponse(
            content='```json\n{"key": "value"}\n```',
            stop_reason="end_turn",
        )
    )

    result = await brain.extract("some text", {"type": "object"})
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_extract_raises_on_invalid_json(brain):
    """extract() raises ValueError if the model returns non-JSON."""
    brain.provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="Here is the data: not json",
            stop_reason="end_turn",
        )
    )

    with pytest.raises(ValueError, match="non-JSON"):
        await brain.extract("text", {"type": "object"})
