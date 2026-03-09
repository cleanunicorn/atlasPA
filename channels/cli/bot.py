"""
channels/cli/bot.py

Local terminal interface for testing the agent without Telegram.

Usage:
    python gateway.py --cli

Commands:
    /clear   — Reset conversation history
    /status  — Show agent status
    /quit    — Exit the agent
"""

import asyncio
import logging
import os
import sys
from providers.base import Message

logger = logging.getLogger(__name__)


class CLIBot:
    """
    Simple readline-based CLI channel for local testing.

    Prints a prompt, reads input, calls brain.think(), prints the response.
    Supports /clear, /status, and /quit commands.
    """

    def __init__(self, brain):
        """
        Args:
            brain: The Brain instance to forward messages to.
        """
        self.brain = brain
        self._history: list[Message] = []
        self._running = False

    async def start(self) -> None:
        """Start the interactive CLI loop."""
        self._running = True
        agent_name = os.getenv("AGENT_NAME", "Atlas")

        print(f"\n{'=' * 50}")
        print(f"  {agent_name} — Personal AI Agent (CLI Mode)")
        print(f"{'=' * 50}")
        print(f"  Commands: /clear, /status, /quit")
        print(f"{'=' * 50}\n")

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Use run_in_executor so readline doesn't block the event loop
                user_input = await loop.run_in_executor(
                    None, self._read_input, "You: "
                )
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Built-in commands
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                self._history = []
                print("🧹 Conversation cleared.\n")
                continue
            elif user_input.lower() == "/status":
                self._print_status()
                continue

            # Forward to the brain
            print()
            try:
                response, self._history = await self.brain.think(
                    user_message=user_input,
                    conversation_history=self._history,
                )
                print(f"{agent_name}: {response}\n")
            except Exception as e:
                logger.exception("Error in brain.think()")
                print(f"⚠️  Error: {e}\n")

        self._running = False

    def _read_input(self, prompt: str) -> str:
        """Blocking readline call — run inside an executor."""
        try:
            return input(prompt)
        except EOFError:
            raise

    def _print_status(self) -> None:
        """Print current agent status to stdout."""
        agent_name = os.getenv("AGENT_NAME", "Atlas")
        provider_name = self.brain.provider.model_name
        skills = self.brain.skills.all_skill_names()
        print(
            f"\n🤖 Agent Status\n"
            f"   Name:                {agent_name}\n"
            f"   Model:               {provider_name}\n"
            f"   Conversation length: {len(self._history)} messages\n"
            f"   Skills:              {', '.join(skills) or 'none'}\n"
        )

    async def stop(self) -> None:
        """Signal the CLI loop to stop."""
        self._running = False
