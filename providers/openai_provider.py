"""
providers/openai_provider.py

LLM provider for OpenAI's GPT models (and compatible APIs).
"""

import json
import os
import openai
from .base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider (also compatible with Azure OpenAI and other OpenAI-compatible APIs)."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", None)  # Optional: override for compatible APIs
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o")

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send messages to OpenAI and return a unified LLMResponse."""

        openai_messages = []

        # OpenAI injects system as a message with role "system"
        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "tool":
                # Tool result message
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                # Assistant message that called tools
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

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

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
