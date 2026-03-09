"""
providers/anthropic.py

LLM provider for Anthropic's Claude models.
"""

import os
import anthropic
from .base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse


class AnthropicProvider(BaseLLMProvider):

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:

        # Convert our unified Message format → Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg.role == "tool":
                # Tool result
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }]
                })
            elif msg.tool_calls:
                # Assistant message that called tools
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        # Convert tool definitions
        anthropic_tools = []
        if tools:
            for t in tools:
                anthropic_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })

        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
        )
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await self.client.messages.create(**kwargs)

        # Convert back to unified format
        content_text = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
