"""
brain/engine.py

The Brain — implements the ReAct (Reason + Act) loop.

Loop:
    1. Build context: system prompt (soul + memory + skills index)
    2. Call LLM with conversation history + available tools
    3. If LLM returns tool calls → execute them → append results → go to 2
    4. If LLM returns final text → return it
    5. Loop max MAX_ITERATIONS times to prevent runaway agents
"""

import logging
from providers.base import BaseLLMProvider, Message, ToolDefinition, ToolCall, LLMResponse
from memory.store import MemoryStore
from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10  # Safety cap on the ReAct loop


class Brain:
    def __init__(self, provider: BaseLLMProvider, memory: MemoryStore, skills: SkillRegistry):
        self.provider = provider
        self.memory = memory
        self.skills = skills

        # Built-in tools (not skills, but always available)
        self._builtin_tools = [
            ToolDefinition(
                name="remember",
                description=(
                    "Save an important fact or note about the user to long-term memory. "
                    "Use this when you learn something about the user worth remembering."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "The fact or note to remember"
                        }
                    },
                    "required": ["note"]
                }
            )
        ]

    def _get_all_tools(self) -> list[ToolDefinition]:
        return self._builtin_tools + self.skills.get_tool_definitions()

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a string."""

        # Built-in: remember
        if tool_call.name == "remember":
            note = tool_call.arguments.get("note", "")
            self.memory.append_context(note)
            logger.info(f"Remembered: {note[:80]}")
            return f"✅ Remembered: {note}"

        # Skill tools (prefixed with "skill_")
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
        # Build system prompt
        skills_summary = self.skills.get_skills_summary()
        system = self.memory.build_system_prompt(skills_summary)

        # Add the new user message to history
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
                # LLM wants to use tools — execute them and loop
                logger.info(f"Tool calls: {[tc.name for tc in response.tool_calls]}")

                # Append assistant message (with tool calls)
                messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ]
                ))

                # Execute each tool and append results
                for tool_call in response.tool_calls:
                    result = await self._execute_tool(tool_call)
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                    ))

                # Continue loop — LLM will now synthesize the tool results
                continue

            else:
                # LLM returned a final text response — we're done
                final_text = response.content or "(no response)"

                # Append final assistant message to history
                messages.append(Message(role="assistant", content=final_text))

                logger.info(
                    f"Brain finished in {iteration + 1} iteration(s). "
                    f"Provider: {self.provider.model_name}"
                )
                return final_text, messages

        # Safety: hit max iterations
        fallback = "I got stuck in a reasoning loop. Please try rephrasing your request."
        messages.append(Message(role="assistant", content=fallback))
        return fallback, messages
