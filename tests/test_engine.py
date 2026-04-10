"""
tests/test_dspy_engine.py

Tests for the Brain (brain/engine.py).

Structure:
  - Unit tests for helper functions (_clean_response, _run_skill_sync, etc.)
  - Unit tests for individual tool factory functions
  - Integration tests for Brain.think() with provider mocked
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from providers.base import BaseLLMProvider, Message, LLMResponse, ToolCall
from memory.store import MemoryStore
from skills.registry import SkillRegistry
from brain.engine import (
    Brain,
    _TurnState,
    _run_skill_sync,
    _make_skill_tool,
    _make_remember,
    _make_forget,
    _make_ask_user,
    _make_send_file,
    _make_create_plan,
    _make_reflect,
    _make_reload,
    _make_request_skills,
    _clean_response,
    TOOL_SELECTION_THRESHOLD,
)


# ── Minimal provider ────────────────────────────────────────────────────────


class MockProvider(BaseLLMProvider):
    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages,
        tools=None,
        system=None,
        max_tokens=4096,
        **kwargs,
    ) -> LLMResponse:
        if self._call_count >= len(self._responses):
            return LLMResponse(content="(no more mock responses)", tool_calls=[])
        r = self._responses[self._call_count]
        self._call_count += 1
        return r


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_memory(tmp_path: Path) -> MemoryStore:
    with patch("memory.store.MEMORY_DIR", tmp_path):
        store = MemoryStore()
    yield store


@pytest.fixture
def empty_skills(tmp_path: Path) -> SkillRegistry:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    with (
        patch("skills.registry.CORE_SKILLS_DIR", skills_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon_skills"),
    ):
        yield SkillRegistry()


@pytest.fixture
def brain(tmp_memory, empty_skills):
    """Brain instance with a mock provider."""
    yield Brain(
        provider=MockProvider(),
        memory=tmp_memory,
        skills=empty_skills,
    )


# ── Unit: _clean_response ─────────────────────────────────────────────────────


def test_clean_response_strips_xml():
    dirty = "Here is the answer.<tool_call>something</tool_call> Done."
    assert "<tool_call>" not in _clean_response(dirty)
    assert "Here is the answer." in _clean_response(dirty)


def test_clean_response_collapses_blank_lines():
    result = _clean_response("Line one\n\n\n\n\nLine two")
    assert "\n\n\n" not in result


# ── Unit: _run_skill_sync ─────────────────────────────────────────────────────


def test_run_skill_sync_with_sync_skill():
    skill = MagicMock()
    skill._module.run = lambda query, **kwargs: f"result: {query}"
    assert _run_skill_sync(skill, {"query": "test"}) == "result: test"


@pytest.mark.asyncio
async def test_run_skill_sync_with_async_skill():
    skill = MagicMock()

    async def async_run(**kwargs):
        return f"async: {kwargs.get('query', '')}"

    skill._module.run = async_run
    result = _run_skill_sync(skill, {"query": "hello"})
    # _run_skill_sync returns the coroutine for the caller to await
    assert asyncio.iscoroutine(result)
    assert await result == "async: hello"


def test_run_skill_sync_returns_string():
    """Non-string return values are coerced to str."""
    skill = MagicMock()
    skill._module.run = lambda **kwargs: 42
    assert _run_skill_sync(skill, {}) == "42"


# ── Unit: _make_skill_tool ────────────────────────────────────────────────────


def test_skill_tool_wrapping_name_and_desc():
    skill = MagicMock()
    skill.name = "web_search"
    skill.description = "Search the web with DuckDuckGo."
    skill._module = MagicMock()
    skill._module.PARAMETERS = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }
    skill._module.run = lambda query, **kwargs: f"results for {query}"

    tool = _make_skill_tool(skill)

    assert tool.name == "skill_web_search"
    assert "query" in tool.args


def test_skill_tool_arg_type_mapping():
    """JSON Schema types are correctly mapped to Python types."""
    skill = MagicMock()
    skill.name = "multi_arg"
    skill.description = "Multi-arg skill"
    skill._module = MagicMock()
    skill._module.PARAMETERS = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
        },
        "required": ["text"],
    }
    skill._module.run = lambda **kwargs: "ok"

    tool = _make_skill_tool(skill)
    assert tool.args["text"]["type"] == "string"
    assert tool.args["count"]["type"] == "integer"


def test_skill_tool_callable():
    """The wrapped tool can be invoked and returns the skill's output."""
    skill = MagicMock()
    skill.name = "echo"
    skill.description = "Echo tool"
    skill._module = MagicMock()
    skill._module.PARAMETERS = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }
    skill._module.run = lambda msg, **kwargs: f"echo: {msg}"

    tool = _make_skill_tool(skill)
    result = tool.func(msg="hello")
    assert result == "echo: hello"


# ── Unit: built-in tool factories ────────────────────────────────────────────


def test_remember_tool_writes_to_memory(tmp_memory):
    tool = _make_remember(tmp_memory)
    result = tool.func(note="User likes tea.")
    assert "✅" in result
    assert "User likes tea." in tmp_memory.load_context()


def test_forget_tool_removes_from_memory(tmp_memory):
    tmp_memory.append_context("User dislikes coffee.")
    tool = _make_forget(tmp_memory)
    tool.func(note="coffee")
    assert "coffee" not in tmp_memory.load_context()


def test_ask_user_sets_clarification():
    state = _TurnState()
    tool = _make_ask_user(state)
    result = tool.func(question="What format do you prefer?")
    assert state.clarification == "What format do you prefer?"
    assert result == "__ASK_USER__"


def test_send_file_queues_path(tmp_path):
    state = _TurnState()
    f = tmp_path / "report.pdf"
    f.write_bytes(b"content")
    tool = _make_send_file(state)
    result = tool.func(path=str(f), caption="Your report")
    assert len(state.pending_files) == 1
    assert state.pending_files[0][0] == f
    assert "✅" in result


def test_send_file_missing_file(tmp_path):
    state = _TurnState()
    tool = _make_send_file(state)
    result = tool.func(path=str(tmp_path / "nonexistent.pdf"))
    assert "Error" in result
    assert len(state.pending_files) == 0


def test_create_plan_updates_state():
    state = _TurnState()
    tool = _make_create_plan(state)
    result = tool.func(title="My Plan", steps=["Step one", "Step two"])
    assert state.current_plan is not None
    assert "My Plan" in state.current_plan
    assert "Step one" in state.current_plan
    assert "Plan recorded" in result


def test_reflect_no_gaps():
    tool = _make_reflect()
    result = tool.func(
        goal="answer the question", accomplished="answered it", gaps="none"
    )
    assert "all steps done" in result.lower()


def test_reflect_with_gaps():
    tool = _make_reflect()
    result = tool.func(
        goal="search and summarise", accomplished="searched", gaps="summary missing"
    )
    assert "summary missing" in result


def test_reload_sets_flag():
    state = _TurnState()
    tool = _make_reload(state)
    result = tool.func()
    assert state.restart_requested is True
    assert result == "__RELOAD__"


# ── Integration: Brain.think() with mocked provider ────────────────────────


def _precache_tools(brain):
    """Pre-cache tool selection so think() doesn't make a selection LLM call."""
    all_names, _ = brain._tool_catalog()
    brain._selected_tools = all_names


@pytest.mark.asyncio
async def test_basic_response(tmp_memory, empty_skills):
    """Brain.think() returns the answer from the LLM."""
    provider = MockProvider(
        [
            LLMResponse(content="Hello there!", tool_calls=[]),
        ]
    )
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)
    _precache_tools(brain)
    response, history = await brain.think("Hi", [])

    assert response == "Hello there!"
    assert history[-1].role == "assistant"
    assert history[-1].content == "Hello there!"
    assert history[0].role == "user"


@pytest.mark.asyncio
async def test_history_passed_through(tmp_memory, empty_skills):
    """Existing conversation history is prepended to messages."""
    prior = [
        Message(role="user", content="Earlier message"),
        Message(role="assistant", content="Earlier reply"),
    ]
    provider = MockProvider(
        [
            LLMResponse(content="Follow-up answer", tool_calls=[]),
        ]
    )
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)
    _precache_tools(brain)
    response, history = await brain.think("Follow-up", prior)

    assert history[0].content == "Earlier message"
    assert history[2].content == "Follow-up"
    assert history[-1].content == "Follow-up answer"


@pytest.mark.asyncio
async def test_tool_call_and_response(tmp_memory, empty_skills):
    """Brain executes tool calls and returns the final text answer."""
    provider = MockProvider(
        [
            # First response: tool call
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1", name="remember", arguments={"note": "User likes cats"}
                    )
                ],
                stop_reason="tool_use",
            ),
            # Second response: final answer
            LLMResponse(content="Got it, I'll remember that!", tool_calls=[]),
        ]
    )
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)
    _precache_tools(brain)
    response, history = await brain.think("I like cats", [])

    assert response == "Got it, I'll remember that!"
    assert "User likes cats" in tmp_memory.load_context()


@pytest.mark.asyncio
async def test_ask_user_stops_loop_and_returns_question(tmp_memory, empty_skills):
    """When ask_user tool is called, think() returns the clarification question."""
    provider = MockProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="q1",
                        name="ask_user",
                        arguments={"question": "What date do you mean?"},
                    )
                ],
                stop_reason="tool_use",
            ),
        ]
    )
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)
    _precache_tools(brain)
    response, history = await brain.think("Schedule something", [])

    assert response == "What date do you mean?"
    assert history[-1].content == "What date do you mean?"


@pytest.mark.asyncio
async def test_remember_tool_invoked_writes_memory(brain, tmp_memory):
    """The remember tool closure writes to memory when called during think()."""
    state = _TurnState()
    tools = brain._build_tools(state, selected_tools=["remember"])
    remember_tool = next(t for t in tools if t.name == "remember")
    remember_tool.func(note="User loves hiking.")
    assert "User loves hiking." in tmp_memory.load_context()


@pytest.mark.asyncio
async def test_send_file_delivered_after_think(brain, tmp_path):
    """Files queued by send_file during think() are returned by take_files()."""
    f = tmp_path / "chart.png"
    f.write_bytes(b"\x89PNG")

    state = _TurnState()
    tools = brain._build_tools(state, selected_tools=["send_file"])
    send_file_tool = next(t for t in tools if t.name == "send_file")
    send_file_tool.func(path=str(f), caption="Your chart")

    # Simulate think() syncing state back
    brain._pending_files = state.pending_files
    files = brain.take_files()

    assert len(files) == 1
    assert files[0][0] == f
    assert files[0][1] == "Your chart"


@pytest.mark.asyncio
async def test_skill_tool_in_brain(tmp_path, tmp_memory):
    """A skill registered in SkillRegistry is available as a tool in think()."""
    skill_dir = tmp_path / "skills" / "greeter"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("Greet the user by name.")
    (skill_dir / "tool.py").write_text(
        "PARAMETERS = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}\n"
        "def run(name='world', **kwargs): return f'Hello, {name}!'\n"
    )

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "skills"),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon_skills"),
    ):
        skills = SkillRegistry()
        b = Brain(provider=MockProvider(), memory=tmp_memory, skills=skills)

    state = _TurnState()
    all_names, _ = b._tool_catalog()
    tools = b._build_tools(state, selected_tools=all_names)
    greeter = next((t for t in tools if t.name == "skill_greeter"), None)
    assert greeter is not None
    assert greeter.func(name="Atlas") == "Hello, Atlas!"


@pytest.mark.asyncio
async def test_take_files_clears_queue(brain, tmp_path):
    """take_files() returns queued files and clears the internal list."""
    f = tmp_path / "doc.txt"
    f.write_text("content")
    brain._pending_files = [(f, "caption")]

    files = brain.take_files()
    assert len(files) == 1
    assert brain._pending_files == []


@pytest.mark.asyncio
async def test_extract_uses_provider_directly(tmp_memory, empty_skills):
    """extract() bypasses the ReAct loop and calls provider.complete() with json_mode=True."""
    provider = MockProvider([LLMResponse(content='{"name": "Atlas"}', tool_calls=[])])
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)
    result = await brain.extract("My name is Atlas", schema={"type": "object"})
    assert result == {"name": "Atlas"}


# ── Status callback tests ───────────────────────────────────────────────────


def test_tool_status_message():
    """Tool names map to human-readable status messages."""
    from brain.status import _tool_status_message

    assert _tool_status_message("skill_web_search") == "Using web search..."
    assert _tool_status_message("skill_browser") == "Using browser..."
    assert _tool_status_message("remember") == "Saving to memory..."
    assert _tool_status_message("ask_user") is None  # suppressed


@pytest.mark.asyncio
async def test_think_calls_on_status(tmp_memory, empty_skills):
    """Brain.think() delivers status updates when on_status is provided."""
    received = []

    async def on_status(msg):
        received.append(msg)

    provider = MockProvider(
        [
            LLMResponse(content="Done", tool_calls=[]),
        ]
    )
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)
    _precache_tools(brain)
    response, _ = await brain.think("Hello", [], on_status=on_status)

    assert response == "Done"
    assert "Thinking..." in received


@pytest.mark.asyncio
async def test_think_on_status_exception_does_not_crash(tmp_memory, empty_skills):
    """If on_status raises, think() still returns normally."""

    async def bad_on_status(msg):
        raise RuntimeError("boom")

    provider = MockProvider(
        [
            LLMResponse(content="All good", tool_calls=[]),
        ]
    )
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)
    _precache_tools(brain)
    response, _ = await brain.think("Hello", [], on_status=bad_on_status)

    assert response == "All good"


# ── Skill selection tests ──────────────────────────────────────────────────


def _make_multi_skill_registry(tmp_path, count):
    """Create a SkillRegistry with `count` dummy skills."""
    skills_dir = tmp_path / "multi_skills"
    skills_dir.mkdir(exist_ok=True)
    for i in range(count):
        sd = skills_dir / f"skill_{i}"
        sd.mkdir(exist_ok=True)
        (sd / "SKILL.md").write_text(f"Skill number {i}")
        (sd / "tool.py").write_text(
            f"PARAMETERS = {{'type': 'object', 'properties': {{'x': {{'type': 'string'}}}}}}\n"
            f"def run(x='', **kwargs): return 'skill_{i}: ' + x\n"
        )
    with (
        patch("skills.registry.CORE_SKILLS_DIR", skills_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon_multi"),
    ):
        return SkillRegistry()


@pytest.mark.asyncio
async def test_select_tools_skips_below_threshold(tmp_memory, empty_skills):
    """_select_tools returns all tool names when count < threshold (no LLM call)."""
    provider = MockProvider()
    brain = Brain(provider=provider, memory=tmp_memory, skills=empty_skills)

    result = await brain._select_tools("hello")
    all_names, _ = brain._tool_catalog()
    assert result == all_names
    assert provider._call_count == 0  # no LLM call made


@pytest.mark.asyncio
async def test_select_tools_calls_provider(tmp_memory, tmp_path):
    """_select_tools makes an LLM call and parses the JSON response."""
    skills = _make_multi_skill_registry(tmp_path, TOOL_SELECTION_THRESHOLD + 1)
    response_json = '{"tools": ["remember", "skill_skill_0"]}'
    provider = MockProvider([LLMResponse(content=response_json, tool_calls=[])])
    brain = Brain(provider=provider, memory=tmp_memory, skills=skills)

    result = await brain._select_tools("do something")
    assert "remember" in result
    assert "skill_skill_0" in result
    assert provider._call_count == 1


@pytest.mark.asyncio
async def test_select_tools_fallback_on_bad_json(tmp_memory, tmp_path):
    """_select_tools returns all tool names when provider returns garbage."""
    skills = _make_multi_skill_registry(tmp_path, TOOL_SELECTION_THRESHOLD + 1)
    provider = MockProvider([LLMResponse(content="not json at all", tool_calls=[])])
    brain = Brain(provider=provider, memory=tmp_memory, skills=skills)

    result = await brain._select_tools("query")
    all_names, _ = brain._tool_catalog()
    assert result == all_names


@pytest.mark.asyncio
async def test_select_tools_strips_markdown_fences(tmp_memory, tmp_path):
    """_select_tools handles JSON wrapped in markdown code fences."""
    skills = _make_multi_skill_registry(tmp_path, TOOL_SELECTION_THRESHOLD + 1)
    fenced = '```json\n{"tools": ["skill_skill_1"]}\n```'
    provider = MockProvider([LLMResponse(content=fenced, tool_calls=[])])
    brain = Brain(provider=provider, memory=tmp_memory, skills=skills)

    result = await brain._select_tools("query")
    assert result == ["skill_skill_1"]


@pytest.mark.asyncio
async def test_select_tools_filters_invalid_names(tmp_memory, tmp_path):
    """_select_tools drops tool names that don't exist."""
    skills = _make_multi_skill_registry(tmp_path, TOOL_SELECTION_THRESHOLD + 1)
    response_json = '{"tools": ["remember", "nonexistent_tool"]}'
    provider = MockProvider([LLMResponse(content=response_json, tool_calls=[])])
    brain = Brain(provider=provider, memory=tmp_memory, skills=skills)

    result = await brain._select_tools("query")
    assert result == ["remember"]


def test_build_tools_filters_all(tmp_memory, tmp_path):
    """_build_tools filters both built-in tools and skills."""
    skills = _make_multi_skill_registry(tmp_path, 3)
    brain = Brain(provider=MockProvider(), memory=tmp_memory, skills=skills)
    state = _TurnState()

    # Select only remember + one skill
    tools = brain._build_tools(state, selected_tools=["remember", "skill_skill_1"])
    names = [t.name for t in tools]

    # Should have: remember, skill_skill_1, plus always-included (ask_user, request_skills)
    assert "remember" in names
    assert "skill_skill_1" in names
    assert "ask_user" in names
    assert "request_skills" in names
    # Should NOT have other built-ins or skills
    assert "forget" not in names
    assert "skill_skill_0" not in names
    assert "skill_skill_2" not in names


def test_build_tools_always_includes_essentials(tmp_memory, empty_skills):
    """_build_tools always includes ask_user and request_skills even with empty selection."""
    brain = Brain(provider=MockProvider(), memory=tmp_memory, skills=empty_skills)
    state = _TurnState()
    tools = brain._build_tools(state, selected_tools=[])
    names = [t.name for t in tools]
    assert "request_skills" in names
    assert "ask_user" in names
    assert len(names) == 2  # only the always-included tools


def test_request_skills_tool_validates_names(tmp_path, tmp_memory):
    """request_skills stores valid names and reports not-found ones."""
    skills = _make_multi_skill_registry(tmp_path, 3)
    state = _TurnState()
    tool = _make_request_skills(state, skills)

    result = tool.func(skill_names="skill_0, nonexistent, skill_2")
    assert "skill_0" in state.requested_skills
    assert "skill_2" in state.requested_skills
    assert "nonexistent" not in state.requested_skills
    assert "Not found: nonexistent" in result


@pytest.mark.asyncio
async def test_react_loop_hotloads_requested_skills(tmp_memory, tmp_path):
    """Skills requested mid-loop are available in subsequent iterations."""
    skills = _make_multi_skill_registry(tmp_path, 3)

    # Iteration 1: LLM calls request_skills to load skill_2
    # Iteration 2: LLM calls skill_skill_2
    # Iteration 3: LLM returns final text
    provider = MockProvider([
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="request_skills", arguments={"skill_names": "skill_2"})],
        ),
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc2", name="skill_skill_2", arguments={"x": "test"})],
        ),
        LLMResponse(content="All done", tool_calls=[]),
    ])
    brain = Brain(provider=provider, memory=tmp_memory, skills=skills)

    # Start with no skills selected (only built-ins + request_skills)
    answer, _ = await brain.think("test", [])
    assert "All done" in answer
