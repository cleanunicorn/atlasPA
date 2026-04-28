import dspy
import asyncio
import json
import logging
import os
from typing import Any, Optional
from providers.base import BaseLLMProvider, Message, ToolDefinition
from brain.tools import BrainTool
from dspy.dsp.utils import dotdict

logger = logging.getLogger(__name__)

class AtlasLM(dspy.LM):
    """
    DSPy LM adapter for Atlas BaseLLMProvider.
    """
    def __init__(self, atlas_provider: BaseLLMProvider, **kwargs):
        self.atlas_provider = atlas_provider
        kwargs.setdefault("max_tokens", int(os.getenv("LLM_MAX_TOKENS", "8192")))
        super().__init__(model=atlas_provider.model_name, **kwargs)

    @property
    def supports_function_calling(self) -> bool:
        return True

    async def aforward(self, prompt: Optional[str] = None, messages: Optional[list[dict[str, Any]]] = None, **kwargs) -> dotdict:
        atlas_messages = []
        system_prompt = None

        if messages:
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    system_prompt = content
                elif role == "tool":
                    atlas_messages.append(Message(
                        role=role,
                        content=content,
                        tool_call_id=msg.get("tool_call_id") or msg.get("id")
                    ))
                elif role == "assistant" and "tool_calls" in msg:
                    tool_calls = []
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        tool_calls.append({
                            "id": tc.get("id"),
                            "name": func.get("name"),
                            "arguments": json.loads(func.get("arguments", "{}")) if isinstance(func.get("arguments"), str) else func.get("arguments")
                        })
                    atlas_messages.append(Message(role=role, content=content, tool_calls=tool_calls))
                else:
                    atlas_messages.append(Message(role=role, content=content))
        elif prompt:
             atlas_messages.append(Message(role="user", content=prompt))

        # Handle tools
        tools = kwargs.get("tools")
        atlas_tools = None
        if tools:
            atlas_tools = []
            for t in tools:
                if isinstance(t, dict) and "function" in t:
                    f = t["function"]
                    atlas_tools.append(ToolDefinition(
                        name=f["name"],
                        description=f.get("description", ""),
                        parameters=f.get("parameters", {"type": "object", "properties": {}})
                    ))
                elif hasattr(t, "name") and hasattr(t, "description"):
                     atlas_tools.append(ToolDefinition(
                        name=t.name,
                        description=t.description,
                        parameters=getattr(t, "parameters", {"type": "object", "properties": {}})
                    ))

        # Filter out dspy-specific kwargs that provider might not like
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("tools", "messages", "prompt", "cache")}

        response = await self.atlas_provider.complete(
            messages=atlas_messages,
            system=system_prompt,
            tools=atlas_tools,
            **clean_kwargs
        )

        # Convert LLMResponse to OpenAI format
        message_dict = {"role": "assistant", "content": response.content or ""}

        if response.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if not isinstance(tc.arguments, str) else tc.arguments
                    }
                }
                for tc in response.tool_calls
            ]

        choices = [dotdict({
            "index": 0,
            "message": dotdict(message_dict),
            "finish_reason": "tool_calls" if response.tool_calls else "stop"
        })]

        return dotdict({
            "id": "atlas-" + str(id(response)),
            "object": "chat.completion",
            "model": self.model,
            "choices": choices,
            "usage": response.usage or {}
        })

    def forward(self, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.aforward(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(self.aforward(*args, **kwargs))

def brain_tool_to_dspy(tool: BrainTool, on_status: Any = None) -> dspy.Tool:
    """
    Wrap a BrainTool into a dspy.Tool with status updates.
    """
    from brain.status import _tool_status_message

    async def wrapped_func(**kwargs):
        if on_status:
            status_msg = _tool_status_message(tool.name)
            if status_msg:
                try:
                    await on_status(status_msg)
                except Exception:
                    pass

        result = tool.func(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)

    wrapped_func.__name__ = tool.name
    wrapped_func.__doc__ = tool.description

    return dspy.Tool(
        func=wrapped_func,
        name=tool.name,
        desc=tool.description
    )

class AtlasSignature(dspy.Signature):
    """
    Primary signature for Atlas agent.
    """
    context = dspy.InputField(desc="Relevant facts and long-term memory about the user.")
    history = dspy.InputField(desc="Recent conversation history.")
    question = dspy.InputField(desc="The user's current request or message.")
    answer = dspy.OutputField(desc="A helpful, concise response to the user.")
