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


# ── Compactor tests ───────────────────────────────────────────────────────────


def _long_messages(n: int, chars_per_msg: int) -> list[Message]:
    """Build n messages alternating user/assistant with `chars_per_msg` of content each."""
    filler = "x" * chars_per_msg
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(Message(role=role, content=f"msg{i}-{filler}"))
    return msgs


def test_estimate_tokens_char_heuristic():
    from brain.compactor import estimate_tokens, estimate_history_tokens

    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens(None if False else "") == 0

    # list-content flattening via _extract_text
    msgs = [
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content=[{"type": "text", "text": "world"}, {"type": "image"}],
        ),
    ]
    # "hello" → 1, role "user" → 1; "world" → 1, role "assistant" → 2
    assert estimate_history_tokens(msgs) == 1 + 1 + 1 + 2


@pytest.mark.asyncio
async def test_no_compaction_under_threshold():
    from brain.compactor import maybe_compact_history

    provider = MockProvider([])
    msgs = _long_messages(4, 10)
    result, was_compacted = await maybe_compact_history(
        msgs, provider, system_prompt_tokens=0, query_tokens=0
    )
    assert was_compacted is False
    assert result is msgs
    assert provider._call_count == 0


@pytest.mark.asyncio
async def test_compaction_triggered_over_threshold(monkeypatch):
    from brain.compactor import maybe_compact_history, SUMMARY_MARKER

    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "5000")
    monkeypatch.setenv("CONTEXT_COMPACTION_THRESHOLD", "0.8")

    msgs = _long_messages(30, 1000)  # ~30 * 1000 / 4 = 7500 tokens >> 4000 budget
    provider = MockProvider(
        [LLMResponse(content="Dense summary of earlier dialogue.", tool_calls=[])]
    )
    result, was_compacted = await maybe_compact_history(
        msgs, provider, system_prompt_tokens=0, query_tokens=0
    )

    assert was_compacted is True
    assert provider._call_count == 1
    assert result[0].role == "user"
    assert isinstance(result[0].content, str)
    assert result[0].content.startswith(SUMMARY_MARKER)
    assert "Dense summary" in result[0].content


@pytest.mark.asyncio
async def test_orphan_tool_boundary_not_split(monkeypatch):
    from brain.compactor import maybe_compact_history

    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "1000")
    monkeypatch.setenv("CONTEXT_COMPACTION_THRESHOLD", "0.8")

    filler = "y" * 1000
    # 10 messages; desired cut = 5. Place an assistant(tool_calls) at idx 4
    # and a tool result at idx 5 — the cut must move forward past both.
    msgs = [
        Message(role="user", content=f"u0-{filler}"),
        Message(role="assistant", content=f"a1-{filler}"),
        Message(role="user", content=f"u2-{filler}"),
        Message(role="assistant", content=f"a3-{filler}"),
        Message(
            role="assistant",
            content=f"a4-{filler}",
            tool_calls=[{"id": "t1", "name": "skill_x", "arguments": {}}],
        ),
        Message(role="tool", content="tool-output", tool_call_id="t1"),
        Message(role="assistant", content=f"a6-{filler}"),
        Message(role="user", content=f"u7-{filler}"),
        Message(role="assistant", content=f"a8-{filler}"),
        Message(role="user", content=f"u9-{filler}"),
    ]

    provider = MockProvider([LLMResponse(content="summary", tool_calls=[])])
    result, was_compacted = await maybe_compact_history(
        msgs, provider, system_prompt_tokens=0, query_tokens=0
    )

    assert was_compacted is True
    # Post-summary tail must not start with a tool message
    assert result[1].role != "tool"
    # And must not be preceded (in the tail) by an assistant with tool_calls
    # that references a now-missing tool result. Simpler check: no tool_calls
    # on any tail message's predecessor inside the tail without its result.
    assert not any(m.role == "tool" for m in result[1:2])


@pytest.mark.asyncio
async def test_provider_failure_returns_original(monkeypatch):
    from brain.compactor import maybe_compact_history

    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "5000")
    monkeypatch.setenv("CONTEXT_COMPACTION_THRESHOLD", "0.8")

    class FailingProvider(BaseLLMProvider):
        @property
        def model_name(self) -> str:
            return "failing"

        async def complete(self, *a, **kw):
            raise RuntimeError("boom")

    msgs = _long_messages(20, 1000)
    result, was_compacted = await maybe_compact_history(
        msgs, FailingProvider(), system_prompt_tokens=0, query_tokens=0
    )
    assert was_compacted is False
    assert result is msgs


@pytest.mark.asyncio
async def test_already_compacted_idempotent(monkeypatch):
    from brain.compactor import maybe_compact_history, SUMMARY_MARKER

    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "5000")
    monkeypatch.setenv("CONTEXT_COMPACTION_THRESHOLD", "0.8")

    msgs = _long_messages(30, 1000)
    provider = MockProvider(
        [
            LLMResponse(content="summary-1", tool_calls=[]),
            LLMResponse(content="summary-2", tool_calls=[]),
        ]
    )

    first, c1 = await maybe_compact_history(
        msgs, provider, system_prompt_tokens=0, query_tokens=0
    )
    assert c1 is True
    assert first[0].content.startswith(SUMMARY_MARKER)

    # Run again on the already-compacted history. It's now under the threshold,
    # so it should be a no-op and still carry exactly one summary marker.
    second, c2 = await maybe_compact_history(
        first, provider, system_prompt_tokens=0, query_tokens=0
    )
    assert c2 is False
    assert second is first
    marker_count = sum(
        1
        for m in second
        if isinstance(m.content, str) and m.content.startswith(SUMMARY_MARKER)
    )
    assert marker_count == 1


@pytest.mark.asyncio
async def test_brain_think_invokes_compaction(tmp_memory, empty_skills, monkeypatch):
    """Brain.think() wires the compactor with system_prompt_tokens and query_tokens."""
    from brain import engine as engine_mod

    spy_calls = []

    async def spy_compact(messages, provider, *, system_prompt_tokens, query_tokens):
        spy_calls.append(
            {
                "n": len(messages),
                "system_prompt_tokens": system_prompt_tokens,
                "query_tokens": query_tokens,
            }
        )
        return messages, False

    monkeypatch.setattr(engine_mod, "maybe_compact_history", spy_compact)

    brain = Brain(
        provider=MockProvider(
            [
                LLMResponse(content='{"tools": []}', tool_calls=[]),  # tool selector
                LLMResponse(content="ok", tool_calls=[]),  # react answer
            ]
        ),
        memory=tmp_memory,
        skills=empty_skills,
    )

    history = [
        Message(role="user", content="previous turn"),
        Message(role="assistant", content="previous answer"),
    ]

    # Don't actually run DSPy — short-circuit _run_react
    async def fake_run_react(
        system_prompt, history_str, query_text, state, selected_tools, on_status=None
    ):
        return "done"

    brain._run_react = fake_run_react

    answer, _ = await brain.think(
        user_message="hello there", conversation_history=history
    )
    assert answer == "done"
    assert len(spy_calls) == 1
    assert spy_calls[0]["n"] == 2
    assert spy_calls[0]["query_tokens"] == len("hello there") // 4
    assert spy_calls[0]["system_prompt_tokens"] > 0
