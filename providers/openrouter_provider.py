"""
providers/openrouter_provider.py

LLM provider for OpenRouter — OpenAI-compatible API with access to many models.
"""

import json
import os
import openai
from .base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider — routes to any supported model via OpenAI-compatible API."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
        )
        self._model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")

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

        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                openai_messages.append({"role": msg.role, "content": msg.content})

        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=openai_messages,
        )

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        content_text = msg.content
        tool_calls: list[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        stop_reason = stop_reason_map.get(choice.finish_reason or "stop", "end_turn")

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        )
