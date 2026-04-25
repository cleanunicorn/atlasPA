import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from providers.base import Message, LLMResponse
from memory.summariser import maybe_summarise_history, maybe_summarise, HISTORY_KEEP_RECENT
from brain.engine import Brain
from memory.store import MemoryStore
from skills.registry import SkillRegistry

class MockProvider:
    def __init__(self, response_content="Summary text"):
        self.model_name = "mock-model"
        self.max_context_window = 1000
        self.response_content = response_content

    def count_tokens(self, text: str) -> int:
        # Simple word count as token estimate for tests
        return len(text.split())

    async def complete(self, messages, **kwargs):
        return LLMResponse(content=self.response_content)

@pytest.mark.asyncio
async def test_maybe_summarise_history_triggers_with_extra_tokens():
    provider = MockProvider()
    # 100 words per message * 4 messages = 400 "tokens"
    # extra_tokens = 500
    # Total = 900 (>= 80% of 1000)
    messages = [
        Message(role="user", content="word " * 100) for _ in range(10) # 10 messages
    ]

    # extra_tokens is 500
    updated_messages = await maybe_summarise_history(messages, provider, 1000, extra_tokens=500)

    # Should keep HISTORY_KEEP_RECENT (6) + 1 summary message = 7
    assert len(updated_messages) == HISTORY_KEEP_RECENT + 1
    assert updated_messages[0].content.startswith("[Conversation Summary]:")
    assert updated_messages[0].role == "assistant"

@pytest.mark.asyncio
async def test_maybe_summarise_history_not_triggered():
    provider = MockProvider()
    # 10 words per message * 5 messages = 50 "tokens" (< 800)
    messages = [
        Message(role="user", content="word " * 10) for _ in range(5)
    ]

    updated_messages = await maybe_summarise_history(messages, provider, provider.max_context_window)

    assert len(updated_messages) == len(messages)
    assert messages == updated_messages

@pytest.mark.asyncio
async def test_remember_triggers_summarise(tmp_path):
    from brain.tools import _make_remember

    # Mock MemoryStore
    with patch("memory.store.MEMORY_DIR", tmp_path):
        store = MemoryStore()

    provider = MockProvider()

    # Pre-fill store with many entries to trigger summary
    # SUMMARY_THRESHOLD is 20 by default
    for i in range(25):
        store.append_context(f"Fact {i}")

    # Initially many entries
    entries = store.parse_context_entries()
    assert len(entries) > 20

    result = await maybe_summarise(store, provider)
    assert result is True

    # After summarization, entries should be around half + 1 (summary)
    entries_after = store.parse_context_entries()
    assert len(entries_after) < len(entries)
    assert any("Summary text" in e.content for e in entries_after)

@pytest.mark.asyncio
async def test_brain_think_triggers_compaction_full_context(monkeypatch):
    from brain.engine import Brain
    from brain.compactor import SUMMARY_MARKER
    from providers.base import Message, LLMResponse

    class MockProviderForBrain(MockProvider):
        async def complete(self, messages, **kwargs):
            return LLMResponse(content="compact summary")

    provider = MockProviderForBrain()

    # Force compactor to trigger by shrinking budget
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_COMPACTION_THRESHOLD", "0.5")

    memory = AsyncMock()
    memory.build_system_prompt.return_value = "word " * 200

    skills = MagicMock()
    skills.get_skills_summary.return_value = "Skills summary"

    brain = Brain(provider=provider, memory=memory, skills=skills)
    brain._select_tools = AsyncMock(return_value=[])
    brain._run_react = AsyncMock(return_value="Final answer")

    history = [Message(role="user", content="word " * 100) for _ in range(8)]

    await brain.think("Current message", history)

    args, kwargs = brain._run_react.call_args
    history_str_arg = args[1]
    assert SUMMARY_MARKER in history_str_arg
    # 8 msgs halved → cut=4, so 1 summary + 4 recent = 5 non-empty lines
    lines = [l for l in history_str_arg.split("\n") if l.strip()]
    assert len(lines) == 5

@pytest.mark.asyncio
async def test_remember_tool_is_async_and_summarises(tmp_path):
    from brain.tools import _make_remember
    from memory.store import MemoryStore

    with patch("memory.store.MEMORY_DIR", tmp_path):
        store = MemoryStore()

    # Pre-fill to trigger summary
    for i in range(25):
        store.append_context(f"Fact {i}")

    provider = MockProvider()
    tool = _make_remember(store, provider)

    # Tool should be async
    result = await tool(note="New fact")
    assert "✅ Remembered:" in result

    # Check if summarization happened (context.md should now have the summary)
    entries = store.parse_context_entries()
    assert any("Summary text" in e.content for e in entries)
