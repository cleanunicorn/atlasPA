"""
tests/test_streaming.py

Tests for streaming responses (Phase 6 continuation):
  - BaseLLMProvider.stream() default fallback
  - OpenAI-style stream() with on_token callback
  - brain.think() on_token wiring
  - Telegram _stream_think() placeholder editing
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from providers.base import Message, LLMResponse, ToolCall, ToolDefinition


# ── Provider base: default stream() fallback ──────────────────────────────────

@pytest.mark.asyncio
async def test_base_stream_fallback_calls_on_token():
    """Default stream() fires on_token once with the full content."""
    from providers.base import BaseLLMProvider

    class MinimalProvider(BaseLLMProvider):
        async def complete(self, messages, tools=None, system=None, max_tokens=4096, json_mode=False):
            return LLMResponse(content="hello world", stop_reason="end_turn")
        @property
        def model_name(self): return "mock"

    received = []
    provider = MinimalProvider()
    response = await provider.stream(
        messages=[Message(role="user", content="hi")],
        on_token=lambda chunk: received.append(chunk) or asyncio.sleep(0),
    )
    assert response.content == "hello world"
    assert received == ["hello world"]


@pytest.mark.asyncio
async def test_base_stream_fallback_skips_on_token_for_tool_calls():
    """Default stream() does NOT fire on_token when response is a tool call."""
    from providers.base import BaseLLMProvider

    class ToolProvider(BaseLLMProvider):
        async def complete(self, messages, tools=None, system=None, max_tokens=4096, json_mode=False):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="x", name="remember", arguments={"note": "hi"})],
                stop_reason="tool_use",
            )
        @property
        def model_name(self): return "mock"

    received = []
    provider = ToolProvider()
    response = await provider.stream(
        messages=[Message(role="user", content="hi")],
        on_token=lambda chunk: received.append(chunk) or asyncio.sleep(0),
    )
    assert response.tool_calls
    assert received == []  # no tokens for tool-call responses


# ── brain.think() on_token wiring ────────────────────────────────────────────

@pytest.fixture
def empty_skills(tmp_path):
    from skills.registry import SkillRegistry
    with patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"), \
         patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon"):
        (tmp_path / "core").mkdir()
        (tmp_path / "addon").mkdir()
        yield SkillRegistry()


@pytest.fixture
def brain(empty_skills):
    from memory.store import MemoryStore
    from brain.engine import Brain
    provider = MagicMock()
    provider.model_name = "mock"
    memory = MagicMock(spec=MemoryStore)
    memory.build_system_prompt = AsyncMock(return_value="sys")
    return Brain(provider=provider, memory=memory, skills=empty_skills)


@pytest.mark.asyncio
async def test_think_passes_on_token_to_stream(brain):
    """brain.think() calls provider.stream() with the on_token callback."""
    tokens_seen = []

    async def fake_stream(messages, tools=None, system=None, max_tokens=4096, on_token=None):
        if on_token:
            await on_token("Hello")
            await on_token(" world")
        return LLMResponse(content="Hello world", stop_reason="end_turn")

    brain.provider.stream = fake_stream

    received = []
    text, _ = await brain.think("hi", [], on_token=lambda c: received.append(c) or asyncio.sleep(0))
    assert text == "Hello world"
    assert received == ["Hello", " world"]


@pytest.mark.asyncio
async def test_think_on_token_not_called_during_tool_iterations(brain):
    """on_token is suppressed during tool-call iterations; fires only on final text."""
    call_count = 0

    tool_response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="t1", name="remember", arguments={"note": "x"})],
        stop_reason="tool_use",
    )
    final_response = LLMResponse(content="Done.", stop_reason="end_turn")

    responses = [tool_response, final_response]
    call_index = 0

    async def fake_stream(messages, tools=None, system=None, max_tokens=4096, on_token=None):
        nonlocal call_index
        r = responses[call_index]
        call_index += 1
        # Only fire on_token when there are no tool calls (final response)
        if on_token and not r.tool_calls and r.content:
            await on_token(r.content)
        return r

    brain.provider.stream = fake_stream

    received = []
    text, _ = await brain.think("do it", [], on_token=lambda c: received.append(c) or asyncio.sleep(0))
    assert text == "Done."
    assert received == ["Done."]


# ── Telegram _stream_think() ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_think_edits_placeholder(brain):
    """_stream_think sends a placeholder then edits it with the final HTML."""
    from channels.telegram.bot import TelegramBot

    # Build a minimal TelegramBot without connecting to Telegram
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake:token"}), \
         patch("channels.telegram.bot.Application"):
        bot = TelegramBot.__new__(TelegramBot)
        bot.brain = brain
        bot._history = MagicMock()
        bot._history.load.return_value = []

    # Mock update
    placeholder_msg = AsyncMock()
    update = MagicMock()
    update.message.reply_text = AsyncMock(return_value=placeholder_msg)
    update.effective_user.id = 123

    async def fake_stream_think_inner(messages, tools=None, system=None, max_tokens=4096, on_token=None):
        if on_token:
            await on_token("Hi")
            await on_token(" there")
        return LLMResponse(content="Hi there", stop_reason="end_turn")

    brain.provider.stream = fake_stream_think_inner

    response_text, _ = await bot._stream_think(update, "hello", [])

    assert response_text == "Hi there"
    # Placeholder was sent
    update.message.reply_text.assert_called_once_with("…")
    # Final edit was called with HTML content
    final_edit_args = placeholder_msg.edit_text.call_args_list[-1]
    assert "Hi there" in final_edit_args[0][0] or "Hi there" in str(final_edit_args)
