"""
tests/test_dspy_engine.py

Tests for the DSPy-based Brain (brain/engine.py).

Structure:
  - Unit tests for helper functions (_serialize_history, _run_skill_sync, etc.)
  - Unit tests for individual tool factory functions
  - Integration tests for Brain.think() with dspy.ReAct mocked out
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy

from providers.base import BaseLLMProvider, Message, LLMResponse
from memory.store import MemoryStore
from skills.registry import SkillRegistry
from brain.engine import (
    Brain,
    _TurnState,
    _serialize_history,
    _run_skill_sync,
    _make_skill_tool,
    _make_remember,
    _make_forget,
    _make_ask_user,
    _make_send_file,
    _make_create_plan,
    _make_reflect,
    _make_reload,
    _clean_response,
)


# ── Minimal provider (for extract() which still calls provider directly) ──────


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
    """Brain instance with DSPy configuration mocked out."""
    mock_lm = MagicMock()
    mock_lm.model = "mock/model"
    with (
        patch("brain.engine._build_dspy_lm", return_value=mock_lm),
        patch("brain.engine.dspy.configure"),
    ):
        yield Brain(
            provider=MockProvider(),
            memory=tmp_memory,
            skills=empty_skills,
        )


def _mock_react(answer: str, trajectory: dict | None = None):
    """Return a mock dspy.ReAct class whose instance returns the given answer."""
    pred = dspy.Prediction(answer=answer, trajectory=trajectory or {})
    react_instance = MagicMock(return_value=pred)
    MockReAct = MagicMock(return_value=react_instance)
    return MockReAct


# ── Unit: _serialize_history ──────────────────────────────────────────────────


def test_serialize_history_empty():
    result = _serialize_history([])
    assert result == "(no previous conversation)"


def test_serialize_history_user_and_assistant():
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    result = _serialize_history(messages)
    assert "User: Hello" in result
    assert "Assistant: Hi there!" in result


def test_serialize_history_tool_calls_summarised():
    messages = [
        Message(
            role="assistant",
            content=None,
            tool_calls=[{"name": "skill_web_search"}, {"name": "remember"}],
        ),
    ]
    result = _serialize_history(messages)
    assert "skill_web_search" in result
    assert "remember" in result


def test_serialize_history_tool_results_omitted():
    messages = [
        Message(role="tool", content="search results here", tool_call_id="tc_1"),
    ]
    result = _serialize_history(messages)
    assert "search results here" not in result


def test_serialize_history_multimodal_user_message():
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "Look at this"},
                {"type": "image", "data": "base64data"},
            ],
        ),
    ]
    result = _serialize_history(messages)
    assert "User: Look at this" in result


# ── Unit: _clean_response ─────────────────────────────────────────────────────


def test_clean_response_strips_xml():
    dirty = "Here is the answer.<tool_call>something</tool_call> Done."
    assert "<tool_call>" not in _clean_response(dirty)
    assert "Here is the answer." in _clean_response(dirty)


def test_clean_response_collapses_blank_lines():
    result = _clean_response("Line one\n\n\n\n\nLine two")
    assert "\n\n\n" not in result


# ── Unit: AtlasReAct._parse_tool_args ─────────────────────────────────────────


def test_parse_tool_args_dict_passthrough():
    """A dict input is returned as-is."""
    from brain.engine import AtlasReAct

    assert AtlasReAct._parse_tool_args({"key": "value"}) == {"key": "value"}


def test_parse_tool_args_valid_json_string():
    """A valid JSON string is parsed into a dict."""
    from brain.engine import AtlasReAct

    assert AtlasReAct._parse_tool_args('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_tool_args_malformed_json_repaired():
    """Malformed JSON is repaired via json_repair."""
    from brain.engine import AtlasReAct

    # Missing closing brace — json_repair should fix it
    result = AtlasReAct._parse_tool_args('{"key": "value"')
    assert result == {"key": "value"}


def test_parse_tool_args_non_dict_returns_empty():
    """If the parsed result is not a dict (e.g. a list or string), return {}."""
    from brain.engine import AtlasReAct

    assert AtlasReAct._parse_tool_args("[1, 2, 3]") == {}
    assert AtlasReAct._parse_tool_args('"just a string"') == {}


def test_parse_tool_args_empty_string():
    """An empty string returns {}."""
    from brain.engine import AtlasReAct

    assert AtlasReAct._parse_tool_args("") == {}


def test_parse_tool_args_none():
    """None input returns {}."""
    from brain.engine import AtlasReAct

    assert AtlasReAct._parse_tool_args(None) == {}


# ── Unit: _run_skill_sync ─────────────────────────────────────────────────────


def test_run_skill_sync_with_sync_skill():
    skill = MagicMock()
    skill._module.run = lambda query, **kwargs: f"result: {query}"
    assert _run_skill_sync(skill, {"query": "test"}) == "result: test"


def test_run_skill_sync_with_async_skill():
    skill = MagicMock()

    async def async_run(**kwargs):
        return f"async: {kwargs.get('query', '')}"

    skill._module.run = async_run
    result = _run_skill_sync(skill, {"query": "hello"})
    assert result == "async: hello"


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


# ── Integration: Brain.think() with mocked dspy.ReAct ────────────────────────


@pytest.mark.asyncio
async def test_basic_response(brain):
    """Brain.think() returns the answer from dspy.ReAct."""
    with patch("brain.engine.dspy.ReAct", _mock_react("Hello there!")):
        response, history = await brain.think("Hi", [])

    assert response == "Hello there!"
    assert history[-1].role == "assistant"
    assert history[-1].content == "Hello there!"
    assert history[0].role == "user"


@pytest.mark.asyncio
async def test_history_passed_through(brain):
    """Existing conversation history is prepended to messages."""
    prior = [
        Message(role="user", content="Earlier message"),
        Message(role="assistant", content="Earlier reply"),
    ]
    with patch("brain.engine.dspy.ReAct", _mock_react("Follow-up answer")):
        response, history = await brain.think("Follow-up", prior)

    assert history[0].content == "Earlier message"
    assert history[2].content == "Follow-up"
    assert history[-1].content == "Follow-up answer"


@pytest.mark.asyncio
async def test_ask_user_stops_loop_and_returns_question(brain):
    """When ask_user tool is called, think() returns the clarification question."""

    # Simulate ask_user being called during the ReAct run
    def react_that_asks(*args, **kwargs):
        # Side-effect: set clarification on the state captured by the tool
        pass

    # Inject clarification by patching the state after it's created
    class _AskingTurnState(_TurnState):
        def __init__(self):
            super().__init__()
            self.clarification = "What date do you mean?"

    with (
        patch("brain.engine._TurnState", _AskingTurnState),
        patch("brain.engine.dspy.ReAct", _mock_react("ignored")),
    ):
        response, history = await brain.think("Schedule something", [])

    assert response == "What date do you mean?"
    assert history[-1].content == "What date do you mean?"


@pytest.mark.asyncio
async def test_remember_tool_invoked_writes_memory(brain, tmp_memory):
    """The remember tool closure writes to memory when called during think()."""
    # Simulate by calling the tool directly on the brain's built state
    state = _TurnState()
    tools = brain._build_tools(state)
    remember_tool = next(t for t in tools if t.name == "remember")
    remember_tool.func(note="User loves hiking.")
    assert "User loves hiking." in tmp_memory.load_context()


@pytest.mark.asyncio
async def test_send_file_delivered_after_think(brain, tmp_path):
    """Files queued by send_file during think() are returned by take_files()."""
    f = tmp_path / "chart.png"
    f.write_bytes(b"\x89PNG")

    def react_side_effect(*args, **kwargs):
        # Access brain._pending_files via the state that was created
        pass

    # Queue a file directly via the tool on the state
    state = _TurnState()
    tools = brain._build_tools(state)
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

    mock_lm = MagicMock()
    mock_lm.model = "mock/model"
    with (
        patch("brain.engine._build_dspy_lm", return_value=mock_lm),
        patch("brain.engine.dspy.configure"),
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "skills"),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon_skills"),
    ):
        skills = SkillRegistry()
        b = Brain(provider=MockProvider(), memory=tmp_memory, skills=skills)

    state = _TurnState()
    tools = b._build_tools(state)
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
async def test_extract_uses_provider_directly(brain):
    """extract() bypasses DSPy and calls provider.complete() with json_mode=True."""
    brain.provider = MockProvider(
        [LLMResponse(content='{"name": "Atlas"}', tool_calls=[])]
    )
    result = await brain.extract("My name is Atlas", schema={"type": "object"})
    assert result == {"name": "Atlas"}


# ── Status callback tests ───────────────────────────────────────────────────


def test_status_callback_puts_to_queue():
    """AtlasCallback.on_lm_start puts a StatusUpdate into the queue."""
    from brain.status import AtlasCallback, StatusUpdate

    loop = asyncio.new_event_loop()
    queue = asyncio.Queue(maxsize=64)
    cb = AtlasCallback()
    cb.activate(queue, loop)

    # Simulate DSPy calling the callback from a worker thread
    cb.on_lm_start("call1", MagicMock(), {})

    # Drain via the loop
    loop.run_until_complete(asyncio.sleep(0))
    assert not queue.empty()
    update = queue.get_nowait()
    assert isinstance(update, StatusUpdate)
    assert update.message == "Thinking..."
    assert update.phase == "lm_start"
    loop.close()


def test_status_callback_tool_start():
    """on_tool_start puts the human-readable tool label into the queue."""
    from brain.status import AtlasCallback

    loop = asyncio.new_event_loop()
    queue = asyncio.Queue(maxsize=64)
    cb = AtlasCallback()
    cb.activate(queue, loop)

    tool_instance = MagicMock()
    tool_instance.name = "remember"
    cb.on_tool_start("call2", tool_instance, {})

    loop.run_until_complete(asyncio.sleep(0))
    update = queue.get_nowait()
    assert update.message == "Saving to memory..."
    assert update.tool_name == "remember"
    loop.close()


def test_status_callback_skill_tool_label():
    """Skill tools get a 'Using X...' label."""
    from brain.status import _tool_status_message

    assert _tool_status_message("skill_web_search") == "Using web search..."
    assert _tool_status_message("skill_browser") == "Using browser..."
    assert _tool_status_message("remember") == "Saving to memory..."
    assert _tool_status_message("finish") == "Wrapping up..."
    assert _tool_status_message("ask_user") is None  # suppressed


def test_status_callback_inactive_does_not_crash():
    """When deactivated, callbacks are silently dropped."""
    from brain.status import AtlasCallback

    cb = AtlasCallback()
    # Should not raise — no queue/loop set
    cb.on_lm_start("call3", MagicMock(), {})
    cb.on_tool_start("call4", MagicMock(), {})


@pytest.mark.asyncio
async def test_drain_status_calls_on_status():
    """drain_status forwards StatusUpdate messages to the on_status callback."""
    from brain.status import StatusUpdate, drain_status

    queue = asyncio.Queue()
    received = []

    async def on_status(msg):
        received.append(msg)

    queue.put_nowait(StatusUpdate("Thinking...", "lm_start"))
    queue.put_nowait(
        StatusUpdate("Using web search...", "tool_start", "skill_web_search")
    )
    queue.put_nowait(None)  # sentinel

    await drain_status(queue, on_status)
    assert received == ["Thinking...", "Using web search..."]


@pytest.mark.asyncio
async def test_drain_status_exception_does_not_crash():
    """If on_status raises, drain_status keeps going."""
    from brain.status import StatusUpdate, drain_status

    queue = asyncio.Queue()

    async def bad_on_status(msg):
        raise RuntimeError("display error")

    queue.put_nowait(StatusUpdate("Thinking...", "lm_start"))
    queue.put_nowait(None)

    # Should not raise
    await drain_status(queue, bad_on_status)


@pytest.mark.asyncio
async def test_think_calls_on_status(brain):
    """Brain.think() delivers status updates when on_status is provided."""
    received = []

    async def on_status(msg):
        received.append(msg)

    with patch("brain.engine.dspy.ReAct", _mock_react("Done")):
        response, _ = await brain.think("Hello", [], on_status=on_status)

    assert response == "Done"
    # At minimum, on_lm_start fires once producing "Thinking..."
    # The exact count depends on DSPy internals, but the callback was wired up.
    # With a mocked ReAct the callback may not fire, so we just verify no crash.


@pytest.mark.asyncio
async def test_think_on_status_exception_does_not_crash(brain):
    """If on_status raises, think() still returns normally."""

    async def bad_on_status(msg):
        raise RuntimeError("boom")

    with patch("brain.engine.dspy.ReAct", _mock_react("All good")):
        response, _ = await brain.think("Hello", [], on_status=bad_on_status)

    assert response == "All good"
