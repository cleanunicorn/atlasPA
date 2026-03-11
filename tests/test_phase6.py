"""
tests/test_phase6.py

Tests for Phase 6 — advanced reasoning features:
  - Parallel tool execution
  - ask_user → clarification returned to caller
  - create_plan → plan stored on brain
  - reflect → gap detection
  - Brain.extract() → structured JSON output
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from providers.base import Message, LLMResponse, ToolCall


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_tool_response(*tool_names: str, content: str = "") -> LLMResponse:
    """LLMResponse that calls one or more tools."""
    return LLMResponse(
        content=content,
        tool_calls=[
            ToolCall(id=f"id_{n}", name=n, arguments={})
            for n in tool_names
        ],
        stop_reason="tool_use",
    )


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], stop_reason="end_turn")


@pytest.fixture
def brain(empty_skills):
    from unittest.mock import MagicMock
    from memory.store import MemoryStore
    from brain.engine import Brain

    provider = MagicMock()
    provider.model_name = "mock"
    memory = MagicMock(spec=MemoryStore)
    memory.build_system_prompt.return_value = "sys"
    return Brain(provider=provider, memory=memory, skills=empty_skills)


@pytest.fixture
def empty_skills(tmp_path):
    from skills.registry import SkillRegistry
    with patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"), \
         patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon"):
        (tmp_path / "core").mkdir()
        (tmp_path / "addon").mkdir()
        yield SkillRegistry()


# ── Parallel tool execution ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_tools_both_executed(brain):
    """Two tool calls in one response are both executed (via gather)."""
    execution_order = []

    async def slow_tool(tc):
        execution_order.append(tc.name)
        await asyncio.sleep(0)
        return f"result:{tc.name}"

    brain._execute_tool = slow_tool

    brain.provider.complete = AsyncMock(side_effect=[
        _make_tool_response("skill_a", "skill_b"),
        _make_text_response("done"),
    ])

    text, _ = await brain.think("go", [])
    assert "done" in text
    assert set(execution_order) == {"skill_a", "skill_b"}


@pytest.mark.asyncio
async def test_tool_exception_captured_as_error_message(brain):
    """If a tool raises, the error is captured as a tool result string."""
    async def boom(tc):
        raise RuntimeError("exploded")

    brain._execute_tool = boom

    brain.provider.complete = AsyncMock(side_effect=[
        _make_tool_response("bad_tool"),
        _make_text_response("handled"),
    ])

    text, messages = await brain.think("run", [])
    assert "handled" in text
    tool_results = [m.content for m in messages if m.role == "tool"]
    assert any("exploded" in r for r in tool_results)


# ── ask_user / clarification ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_user_returns_question_immediately(brain):
    """When ask_user is called, the loop stops and returns the question."""
    brain.provider.complete = AsyncMock(return_value=LLMResponse(
        content="",
        tool_calls=[ToolCall(id="q1", name="ask_user", arguments={"question": "Which project?"})],
        stop_reason="tool_use",
    ))

    text, messages = await brain.think("do the thing", [])
    assert text == "Which project?"
    # Final assistant message should be the question
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Which project?"


@pytest.mark.asyncio
async def test_ask_user_alongside_other_tools(brain):
    """ask_user wins even when called alongside other tools."""
    called = []

    original = brain._execute_tool

    async def tracking(tc):
        called.append(tc.name)
        return await original(tc)

    brain._execute_tool = tracking

    brain.provider.complete = AsyncMock(return_value=LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="r1", name="remember", arguments={"note": "x"}),
            ToolCall(id="q1", name="ask_user", arguments={"question": "Are you sure?"}),
        ],
        stop_reason="tool_use",
    ))

    text, _ = await brain.think("do it", [])
    assert text == "Are you sure?"
    assert "remember" in called
    assert "ask_user" in called


# ── create_plan ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_plan_stores_plan(brain):
    """create_plan saves the plan to brain._current_plan."""
    brain.provider.complete = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="p1", name="create_plan", arguments={
                "title": "My Plan",
                "steps": ["Step 1", "Step 2", "Step 3"],
            })],
            stop_reason="tool_use",
        ),
        _make_text_response("Plan executed."),
    ])

    await brain.think("complex task", [])
    assert brain._current_plan is not None
    assert "My Plan" in brain._current_plan
    assert "Step 1" in brain._current_plan
    assert "1." in brain._current_plan


@pytest.mark.asyncio
async def test_create_plan_result_included_in_tool_messages(brain):
    """The plan text is returned as the tool result so the LLM sees it."""
    brain.provider.complete = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="p1", name="create_plan", arguments={
                "title": "Task Plan",
                "steps": ["Do A", "Do B"],
            })],
            stop_reason="tool_use",
        ),
        _make_text_response("ok"),
    ])

    _, messages = await brain.think("task", [])
    tool_msgs = [m.content for m in messages if m.role == "tool"]
    assert any("Task Plan" in m for m in tool_msgs)


# ── reflect ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflect_no_gaps(brain):
    """reflect with 'none' gaps returns a 'proceed' message."""
    result = await brain._execute_tool(
        ToolCall(id="r1", name="reflect", arguments={
            "goal": "write a report",
            "accomplished": "wrote it",
            "gaps": "none",
        })
    )
    assert "proceed" in result.lower() or "done" in result.lower()


@pytest.mark.asyncio
async def test_reflect_with_gaps(brain):
    """reflect with real gaps returns an 'address these' message."""
    result = await brain._execute_tool(
        ToolCall(id="r1", name="reflect", arguments={
            "goal": "write and send a report",
            "accomplished": "wrote report",
            "gaps": "still need to send the email",
        })
    )
    assert "still need to send the email" in result


# ── Brain.extract() ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_parses_json(brain):
    """extract() returns a parsed dict from a JSON-mode provider response."""
    brain.provider.complete = AsyncMock(return_value=LLMResponse(
        content='{"name": "Alice", "age": 30}',
        stop_reason="end_turn",
    ))

    result = await brain.extract(
        text="Alice is 30 years old.",
        schema={"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
    )
    assert result == {"name": "Alice", "age": 30}
    # Verify json_mode=True was passed to provider
    _, kwargs = brain.provider.complete.call_args
    assert kwargs.get("json_mode") is True


@pytest.mark.asyncio
async def test_extract_strips_markdown_fences(brain):
    """extract() handles models that wrap JSON in ```json fences."""
    brain.provider.complete = AsyncMock(return_value=LLMResponse(
        content='```json\n{"key": "value"}\n```',
        stop_reason="end_turn",
    ))

    result = await brain.extract("some text", {"type": "object"})
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_extract_raises_on_invalid_json(brain):
    """extract() raises ValueError if the model returns non-JSON."""
    brain.provider.complete = AsyncMock(return_value=LLMResponse(
        content="Here is the data: not json",
        stop_reason="end_turn",
    ))

    with pytest.raises(ValueError, match="non-JSON"):
        await brain.extract("text", {"type": "object"})
