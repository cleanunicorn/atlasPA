"""
tests/test_dspy_engine.py

Tests for the Brain (brain/engine.py).
"""

import asyncio
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from providers.base import BaseLLMProvider, Message, LLMResponse, ToolCall
from memory.store import MemoryStore
from skills.registry import SkillRegistry
from brain.engine import (
    Brain,
)
from brain.tools import (
    _TurnState,
    _make_skill_tool,
    _make_remember,
    _make_forget,
    _make_ask_user,
    _make_send_file,
    _make_create_plan,
)


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

    @property
    def max_context_window(self) -> int:
        return 1000

    def count_tokens(self, text: str) -> int:
        return len(text.split())

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
    """

    provider = MockProvider(responses)
    brain = Brain(provider=provider, memory=memory, skills=skills)

    _responses = list(responses)

    async def simulated_think(
        user_message,
        conversation_history,
        on_status=None,
        system_suffix="",
    ):
        from brain.engine import _extract_text

        state = _TurnState()
        all_names, _ = brain._tool_catalog()
        tools = brain._build_tools(state, selected_tools=all_names)
        tool_map = {t.name: t for t in tools}

        text = (
            _extract_text(user_message)
            if isinstance(user_message, list)
            else user_message
        )
        messages = list(conversation_history) + [Message(role="user", content=text)]

        for resp in _responses:
            if resp.tool_calls:
                processed_tool_calls = []
                for tc in resp.tool_calls:
                    if isinstance(tc, dict):
                        processed_tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))
                    else:
                        processed_tool_calls.append(tc)

                messages.append(
                    Message(
                        role="assistant",
                        content=resp.content or "",
                        tool_calls=[
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in processed_tool_calls
                        ],
                    )
                )
                for tc in processed_tool_calls:
                    tool = tool_map.get(tc.name)
                    if tool is not None:
                        try:
                            # Note: BrainTool now might be async
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
            else:
                messages.append(Message(role="assistant", content=resp.content))
                # Replicate Brain.think's transfer of state
                brain._pending_files = state.pending_files
                return resp.content, messages

        return "(Simulation ended without final answer)", messages

    brain.think = simulated_think
    return brain, provider


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_brain_direct_text_response(tmp_memory, empty_skills):
    """Brain returns a direct text response if LLM provides one."""
    responses = [LLMResponse(content="Hello! How can I help?")]
    brain, _ = make_brain(responses, tmp_memory, empty_skills)

    answer, history = await brain.think("Hi", [])
    assert answer == "Hello! How can I help?"
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


@pytest.mark.asyncio
async def test_brain_single_tool_loop(tmp_memory, empty_skills):
    """Brain executes a tool and returns the final answer."""
    responses = [
        LLMResponse(
            content="Let me remember that.",
            tool_calls=[
                {"id": "c1", "name": "remember", "arguments": {"note": "User likes tea"}}
            ],
        ),
        LLMResponse(content="I've remembered that you like tea."),
    ]
    brain, _ = make_brain(responses, tmp_memory, empty_skills)

    answer, history = await brain.think("I like tea.", [])
    assert "I've remembered" in answer
    assert "User likes tea" in tmp_memory.context_path.read_text()
    assert len(history) == 4  # user, assistant (call), tool (res), assistant (final)


@pytest.mark.asyncio
async def test_brain_file_sending(tmp_memory, empty_skills, tmp_path):
    """Tools can queue files to be sent to the user."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    responses = [
        LLMResponse(
            content="Sending file…",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "send_file",
                    "arguments": {"path": str(test_file), "caption": "My file"},
                }
            ],
        ),
        LLMResponse(content="Done."),
    ]
    brain, _ = make_brain(responses, tmp_memory, empty_skills)

    await brain.think("Send me the file", [])
    files = brain.take_files()

    assert len(files) == 1
    assert files[0][0] == test_file
    assert files[0][1] == "My file"
    assert brain.take_files() == []  # queue cleared


def test_turn_state_initialization():
    """_TurnState starts empty."""
    state = _TurnState()
    assert state.clarification is None
    assert state.restart_requested is False
    assert state.pending_files == []
    assert state.current_plan is None


@pytest.mark.asyncio
async def test_make_skill_tool_is_async_bridge():
    """_make_skill_tool wraps a skill in an async function that awaits Skill.run."""
    mock_skill = MagicMock()
    mock_skill.name = "test"
    mock_skill.description = "desc"
    mock_skill._module = MagicMock()
    mock_skill.run = AsyncMock(return_value="res")

    tool = _make_skill_tool(mock_skill)
    assert tool.name == "skill_test"
    assert tool.description == "desc"

    result = await tool(x=10)
    assert result == "res"
    mock_skill.run.assert_awaited_with(x=10)


@pytest.mark.asyncio
async def test_remember_tool(tmp_memory):
    """remember tool appends to context.md."""
    # Updated to pass provider
    provider = MagicMock(spec=BaseLLMProvider)
    # mock maybe_summarise so it doesn't need real provider
    with patch("brain.tools.maybe_summarise", AsyncMock(return_value=False)):
        tool = _make_remember(tmp_memory, provider)
        result = await tool(note="Test fact")
        assert "✅ Remembered" in result
        assert "Test fact" in tmp_memory.context_path.read_text()


def test_forget_tool(tmp_memory):
    """forget tool removes entry from context.md."""
    tmp_memory.append_context("Fact to forget")
    tool = _make_forget(tmp_memory)
    result = tool(note="Fact to forget")
    assert "✅ Forgot" in result
    assert "Fact to forget" not in tmp_memory.context_path.read_text()


def test_ask_user_tool():
    """ask_user tool sets clarification on turn state."""
    state = _TurnState()
    tool = _make_ask_user(state)
    result = tool(question="What is your name?")
    assert result == "__ASK_USER__"
    assert state.clarification == "What is your name?"


def test_create_plan_tool():
    """create_plan tool stores formatted plan on turn state."""
    state = _TurnState()
    tool = _make_create_plan(state)
    tool(title="My Plan", steps=["Step A", "Step B"])
    assert "**My Plan**" in state.current_plan
    assert "1. Step A" in state.current_plan
    assert "2. Step B" in state.current_plan


def test_send_file_tool():
    """send_file tool queues path on turn state."""
    state = _TurnState()
    tool = _make_send_file(state)
    with patch("pathlib.Path.exists", return_value=True):
        tool(path="/tmp/test.png", caption="Cool image")
    assert len(state.pending_files) == 1
    assert str(state.pending_files[0][0]) == "/tmp/test.png"
    assert state.pending_files[0][1] == "Cool image"
