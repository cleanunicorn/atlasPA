"""
brain/engine.py

The Brain — implements the ReAct (Reason + Act) loop.

Loop:
    1. Build context: system prompt (soul + relevant memory + skills index)
    2. Call LLM with conversation history + available tools
    3. If LLM returns tool calls → execute them → append results → go to 2
    4. If LLM returns final text → return it
    5. Loop max MAX_ITERATIONS times to prevent runaway agents

Phase 2 additions:
    - `forget` built-in tool: remove a memory entry by content match
    - Relevance-aware system prompt: passes current query to MemoryStore
    - Auto-summarisation: compresses context.md after `remember` if needed
"""

import logging
from providers.base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse
from memory.store import MemoryStore
from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10  # Safety cap on the ReAct loop


class Brain:
    """
    ReAct reasoning engine.

    Receives a user message + conversation history, runs a tool-use loop
    until the LLM produces a final text response, and returns it.
    """

    def __init__(self, provider: BaseLLMProvider, memory: MemoryStore, skills: SkillRegistry):
        self.provider = provider
        self.memory = memory
        self.skills = skills

        # Built-in tools — always available, regardless of installed skills
        self._builtin_tools = [
            ToolDefinition(
                name="remember",
                description=(
                    "Save an important fact or note about the user to long-term memory. "
                    "Use this when you learn something about the user worth remembering "
                    "across sessions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "The fact or note to remember, written as a concise statement",
                        }
                    },
                    "required": ["note"],
                },
            ),
            ToolDefinition(
                name="forget",
                description=(
                    "Remove an outdated or incorrect fact from long-term memory. "
                    "Use when a previously remembered fact is no longer true or was wrong."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Description of the fact to forget (partial match is fine)",
                        }
                    },
                    "required": ["note"],
                },
            ),
        ]

    def _get_all_tools(self) -> list[ToolDefinition]:
        return self._builtin_tools + self.skills.get_tool_definitions()

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a string."""

        if tool_call.name == "remember":
            note = tool_call.arguments.get("note", "")
            self.memory.append_context(note)
            logger.info(f"Remembered: {note[:80]}")
            # Auto-summarise if context has grown large (fire-and-forget)
            try:
                from memory.summariser import maybe_summarise
                await maybe_summarise(self.memory, self.provider)
            except Exception as e:
                logger.warning(f"Summarisation skipped: {e}")
            return f"✅ Remembered: {note}"

        if tool_call.name == "forget":
            note = tool_call.arguments.get("note", "")
            result = self.memory.forget_entry(note)
            logger.info(f"Forget request: '{note[:80]}' → {result[:80]}")
            return result

        if tool_call.name.startswith("skill_"):
            skill_name = tool_call.name[len("skill_"):]
            skill = self.skills.get_skill(skill_name)
            if skill:
                logger.info(f"Running skill: {skill_name} with {tool_call.arguments}")
                return await skill.run(**tool_call.arguments)
            return f"Unknown skill: {skill_name}"

        return f"Unknown tool: {tool_call.name}"

    async def think(
        self,
        user_message: str,
        conversation_history: list[Message],
    ) -> tuple[str, list[Message]]:
        """
        Run the ReAct loop for a single user message.

        Args:
            user_message:           The user's new message.
            conversation_history:   Previous messages in this conversation.

        Returns:
            (final_response_text, updated_history)
        """
        # Build system prompt — pass query for relevance-filtered context injection
        skills_summary = self.skills.get_skills_summary()
        system = self.memory.build_system_prompt(skills_summary, query=user_message)

        messages = list(conversation_history) + [
            Message(role="user", content=user_message)
        ]

        all_tools = self._get_all_tools()

        for iteration in range(MAX_ITERATIONS):
            logger.debug(f"ReAct iteration {iteration + 1}/{MAX_ITERATIONS}")

            response: LLMResponse = await self.provider.complete(
                messages=messages,
                tools=all_tools,
                system=system,
            )

            if response.tool_calls:
                logger.info(f"Tool calls: {[tc.name for tc in response.tool_calls]}")

                messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                ))

                for tool_call in response.tool_calls:
                    result = await self._execute_tool(tool_call)
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                    ))

                continue

            else:
                final_text = response.content or "(no response)"
                messages.append(Message(role="assistant", content=final_text))
                logger.info(
                    f"Brain finished in {iteration + 1} iteration(s). "
                    f"Provider: {self.provider.model_name}"
                )
                return final_text, messages

        fallback = "I got stuck in a reasoning loop. Please try rephrasing your request."
        messages.append(Message(role="assistant", content=fallback))
        return fallback, messages
