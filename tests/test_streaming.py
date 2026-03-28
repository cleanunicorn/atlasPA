"""
tests/test_streaming.py

Tests for streaming at the provider level.

Note: Brain.think() uses on_status (not on_token) for progress updates.
Provider-level streaming tests for the base stream() fallback remain here.
"""

import asyncio
import pytest

from providers.base import Message, LLMResponse, ToolCall


# ── Provider base: default stream() fallback ──────────────────────────────────


@pytest.mark.asyncio
async def test_base_stream_fallback_calls_on_token():
    """Default stream() fires on_token once with the full content."""
    from providers.base import BaseLLMProvider

    class MinimalProvider(BaseLLMProvider):
        async def complete(
            self, messages, tools=None, system=None, max_tokens=4096, json_mode=False
        ):
            return LLMResponse(content="hello world", stop_reason="end_turn")

        @property
        def model_name(self):
            return "mock"

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
        async def complete(
            self, messages, tools=None, system=None, max_tokens=4096, json_mode=False
        ):
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="x", name="remember", arguments={"note": "hi"})
                ],
                stop_reason="tool_use",
            )

        @property
        def model_name(self):
            return "mock"

    received = []
    provider = ToolProvider()
    response = await provider.stream(
        messages=[Message(role="user", content="hi")],
        on_token=lambda chunk: received.append(chunk) or asyncio.sleep(0),
    )
    assert response.tool_calls
    assert received == []
