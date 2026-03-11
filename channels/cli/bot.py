"""
channels/cli/bot.py

Local terminal interface for testing the agent without Telegram.

Usage:
    python gateway.py --cli

Commands:
    /clear          — Reset conversation history
    /status         — Show agent status
    /image <path>   — Send an image file to the agent
    /quit           — Exit the agent
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from memory.history import ConversationHistory

logger = logging.getLogger(__name__)

CLI_USER_ID = "cli"


class CLIBot:
    """
    Simple readline-based CLI channel for local testing.

    Prints a prompt, reads input, calls brain.think(), prints the response.
    Supports /clear, /status, and /quit commands.
    Conversation history is persisted so sessions resume across restarts.
    """

    def __init__(self, brain):
        """
        Args:
            brain: The Brain instance to forward messages to.
        """
        self.brain = brain
        self._history_store = ConversationHistory()
        self._running = False

    async def start(self) -> None:
        """Start the interactive CLI loop."""
        self._running = True
        agent_name = os.getenv("AGENT_NAME", "Atlas")

        history = self._history_store.load(CLI_USER_ID)
        resume_note = f" (resuming — {len(history)} messages)" if history else ""

        print(f"\n{'=' * 50}")
        print(f"  {agent_name} — Personal AI Agent (CLI Mode){resume_note}")
        print(f"{'=' * 50}")
        print(f"  Commands: /clear, /status, /image <path>, /quit")
        print(f"{'=' * 50}\n")

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                user_input = await loop.run_in_executor(
                    None, self._read_input, "You: "
                )
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                self._history_store.clear(CLI_USER_ID)
                print("🧹 Conversation cleared.\n")
                history = []
                continue
            elif user_input.lower() == "/status":
                history = self._history_store.load(CLI_USER_ID)
                self._print_status(len(history))
                continue
            elif user_input.lower().startswith("/image "):
                image_path = Path(user_input[7:].strip()).expanduser()
                if not image_path.exists():
                    print(f"⚠️  File not found: {image_path}\n")
                    continue
                suffix = image_path.suffix.lower()
                media_type_map = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                media_type = media_type_map.get(suffix, "image/jpeg")
                img_b64 = base64.b64encode(image_path.read_bytes()).decode()
                user_input = [{"type": "image", "media_type": media_type, "data": img_b64}]
                print(f"📸 Image loaded: {image_path.name}\n")

            print()
            history = self._history_store.load(CLI_USER_ID)
            try:
                response, updated_history = await self.brain.think(
                    user_message=user_input,
                    conversation_history=history,
                )
                self._history_store.save(CLI_USER_ID, updated_history)
                print(f"{agent_name}: {response}\n")
                for path, caption in self.brain.take_files():
                    note = f" — {caption}" if caption else ""
                    print(f"📎 File: {path}{note}\n")
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

    def _print_status(self, history_len: int) -> None:
        """Print current agent status to stdout."""
        agent_name = os.getenv("AGENT_NAME", "Atlas")
        provider_name = self.brain.provider.model_name
        skills = self.brain.skills.all_skill_names()
        context_entries = len(self.brain.memory.parse_context_entries())
        print(
            f"\n🤖 Agent Status\n"
            f"   Name:              {agent_name}\n"
            f"   Model:             {provider_name}\n"
            f"   Conversation:      {history_len} messages\n"
            f"   Long-term memories:{context_entries}\n"
            f"   Skills:            {', '.join(skills) or 'none'}\n"
        )

    async def stop(self) -> None:
        """Signal the CLI loop to stop."""
        self._running = False
