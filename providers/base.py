"""
providers/base.py

Abstract base class for all LLM providers.
Every provider must implement `complete()` — that's the only contract.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Awaitable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single message in a conversation."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None   # for tool result messages
    tool_calls: list[dict] | None = None  # for assistant tool call messages


@dataclass
class ToolDefinition:
    """Describes a tool the LLM can call."""
    name: str
    description: str
    parameters: dict   # JSON Schema object


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Unified response object returned by all providers."""
    content: str | None           # Text response (None if tool call only)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "max_tokens"
    usage: dict[str, int] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    All LLM providers implement this interface.

    Usage:
        provider = AnthropicProvider()
        response = await provider.complete(messages, tools)
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Send messages to the LLM and return a unified response.

        Args:
            messages:   Conversation history.
            tools:      Available tools the model may call.
            system:     System prompt (injected separately from messages).
            max_tokens: Maximum tokens to generate.
            json_mode:  If True, instruct the model to return valid JSON only.
                        Incompatible with tool use — do not pass tools when set.

        Returns:
            LLMResponse with content and/or tool_calls.
        """
        ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """
        Like complete(), but calls *on_token* for each text token as it arrives.

        Default implementation: calls complete() then fires on_token once with
        the full content.  Override for true incremental streaming.

        Args:
            on_token: Async callback receiving each text chunk.
                      Not called when the response is a tool call.
        """
        response = await self.complete(messages, tools, system, max_tokens)
        if on_token and response.content and not response.tool_calls:
            await on_token(response.content)
        return response

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging."""
        ...
