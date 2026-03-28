"""
providers/logging_provider.py

Transparent wrapper around any BaseLLMProvider that logs every request
and response to a file in JSON-lines format.

Log location: logs/llm.jsonl  (one JSON object per line)
Override with LLM_LOG_FILE env var.
"""

import json
import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict

from collections.abc import Callable, Awaitable
from .base import BaseLLMProvider, Message, ToolDefinition, LLMResponse


def _message_to_dict(msg: Message) -> dict:
    return {
        "role": msg.role,
        "content": msg.content,
        **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
        **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
    }


def _tool_def_to_dict(t: ToolDefinition) -> dict:
    return {"name": t.name, "description": t.description, "parameters": t.parameters}


class LoggingProvider(BaseLLMProvider):
    """Wraps another provider and logs all traffic to a JSONL file."""

    def __init__(self, inner: BaseLLMProvider):
        self._inner = inner
        log_path = os.getenv("LLM_LOG_FILE", "logs/llm.jsonl")
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        ts = datetime.now(timezone.utc).isoformat()

        response = await self._inner.complete(
            messages, tools, system, max_tokens, json_mode
        )

        entry = {
            "ts": ts,
            "provider": type(self._inner).__name__,
            "model": self._inner.model_name,
            "request": {
                "system": system,
                "max_tokens": max_tokens,
                "messages": [_message_to_dict(m) for m in messages],
                "tools": [_tool_def_to_dict(t) for t in tools] if tools else [],
            },
            "response": {
                "content": response.content,
                "stop_reason": response.stop_reason,
                "usage": response.usage,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            },
        }

        async with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return response

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Stream via inner provider; log the completed response afterward."""
        ts = datetime.now(timezone.utc).isoformat()
        response = await self._inner.stream(
            messages, tools, system, max_tokens, on_token
        )
        entry = {
            "ts": ts,
            "provider": type(self._inner).__name__,
            "model": self._inner.model_name,
            "request": {
                "system": system,
                "max_tokens": max_tokens,
                "messages": [_message_to_dict(m) for m in messages],
                "tools": [_tool_def_to_dict(t) for t in tools] if tools else [],
                "streamed": True,
            },
            "response": {
                "content": response.content,
                "stop_reason": response.stop_reason,
                "usage": response.usage,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            },
        }
        async with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return response
