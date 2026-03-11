"""
providers/openai_provider.py

LLM provider for OpenAI's GPT models (and compatible APIs).
"""

import json
import os
from collections.abc import Callable, Awaitable
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
        openai_messages = self._build_messages(messages, system)
        kwargs = dict(model=self._model, max_tokens=max_tokens, messages=openai_messages)

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

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Stream tokens via on_token callback; return full LLMResponse when done."""
        openai_messages = self._build_messages(messages, system)

        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=openai_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        content_parts: list[str] = []
        # tool-call accumulator: index → {id, name, args_str}
        tc_acc: dict[int, dict] = {}
        finish_reason: str = "stop"
        usage: dict[str, int] = {}

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage:
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            fr = chunk.choices[0].finish_reason
            if fr:
                finish_reason = fr

            if delta.content:
                content_parts.append(delta.content)
                if on_token and not delta.tool_calls:
                    await on_token(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_acc[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_acc[idx]["args"] += tc_delta.function.arguments

        tool_calls: list[ToolCall] = []
        for tc in sorted(tc_acc.values(), key=lambda x: list(tc_acc.values()).index(x)):
            try:
                arguments = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=arguments))

        stop_reason_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason_map.get(finish_reason, "end_turn"),
            usage=usage,
        )

    def _build_messages(self, messages: list[Message], system: str | None) -> list[dict]:
        """Convert unified Messages to OpenAI format."""
        result = []
        if system:
            result.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "tool":
                result.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
            elif msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                        for tc in msg.tool_calls
                    ],
                })
            elif isinstance(msg.content, list):
                # Multimodal content (text + images)
                openai_content = []
                for block in msg.content:
                    if block.get("type") == "image":
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block['media_type']};base64,{block['data']}",
                            },
                        })
                    else:
                        openai_content.append({"type": "text", "text": block.get("text", "")})
                result.append({"role": msg.role, "content": openai_content})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result
