"""
gateway.py

The Gateway — main entry point for the personal agent.

Responsibilities:
    1. Load configuration (.env)
    2. Instantiate all components: Provider, Memory, Skills, Brain, Channels, Heartbeat
    3. Start all components concurrently
    4. Handle graceful shutdown on SIGINT/SIGTERM

Run modes:
    python gateway.py          — Telegram bot (default)
    python gateway.py --cli    — Interactive terminal session
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Load .env before importing anything else
from dotenv import load_dotenv

env_path = Path(__file__).parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

# ── Logging setup ─────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gateway")


async def run_cli(brain) -> None:
    """Run the agent in interactive CLI mode."""
    from channels.cli.bot import CLIBot

    cli = CLIBot(brain=brain)
    await cli.start()


async def run_telegram(brain) -> None:
    """Run the agent as a Telegram bot."""
    from channels.telegram.bot import TelegramBot
    from heartbeat.scheduler import Heartbeat

    heartbeat = Heartbeat(brain=brain)
    telegram = TelegramBot(brain=brain)

    stop_event = asyncio.Event()

    def handle_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        await heartbeat.start()
        await telegram.start()

        agent_name = os.getenv("AGENT_NAME", "Atlas")
        logger.info(f"✅ {agent_name} is running! Open Telegram and say hello.")
        logger.info("   Press Ctrl+C to stop.\n")

        await stop_event.wait()

    finally:
        logger.info("Shutting down components...")
        await telegram.stop()
        await heartbeat.stop()
        logger.info("Goodbye.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas — Personal AI Agent")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in interactive CLI mode instead of Telegram",
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("  Personal Agent — Starting up")
    logger.info("=" * 50)

    # ── 1. LLM Provider ───────────────────────────────────────────────────────
    from providers import get_provider

    provider = get_provider()
    logger.info(f"LLM Provider: {provider.model_name}")

    # ── 2. Memory ─────────────────────────────────────────────────────────────
    from memory import MemoryStore

    memory = MemoryStore()
    logger.info("Memory loaded")

    # ── 3. Skills ─────────────────────────────────────────────────────────────
    from skills.registry import SkillRegistry

    skills = SkillRegistry()
    logger.info(f"Skills loaded: {skills.all_skill_names()}")

    # ── 4. Brain ──────────────────────────────────────────────────────────────
    from brain import Brain

    brain = Brain(provider=provider, memory=memory, skills=skills)
    logger.info("Brain initialized")

    # ── 5. Run ────────────────────────────────────────────────────────────────
    if args.cli:
        await run_cli(brain)
    else:
        await run_telegram(brain)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
