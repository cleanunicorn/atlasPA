"""
gateway.py

The Gateway — main entry point for the personal agent.

Responsibilities:
    1. Load configuration (.env)
    2. Instantiate all components: Provider, Memory, Skills, Brain, Channels, Heartbeat
    3. Start all components concurrently
    4. Handle graceful shutdown on SIGINT/SIGTERM

Run modes:
    python gateway.py             — Telegram bot (default)
    python gateway.py --cli       — Interactive terminal session
    python gateway.py --discord   — Discord bot
    python gateway.py --web       — Browser-based web UI (http://localhost:7860)
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


async def _run_with_heartbeat(brain, channel) -> None:
    """
    Generic runner: start a channel + heartbeat, wait for shutdown signal.

    channel must expose:
        .push_message(text, files)   — used as notify_callback
        .start()
        .stop()
    """
    from heartbeat import Heartbeat

    heartbeat = Heartbeat(brain=brain, notify_callback=channel.push_message)
    brain.heartbeat = heartbeat  # Back-reference so brain tools can reload jobs

    stop_event = asyncio.Event()

    def handle_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        await heartbeat.start()
        await channel.start()

        agent_name = os.getenv("AGENT_NAME", "Atlas")
        logger.info(f"✅ {agent_name} is running! Press Ctrl+C to stop.\n")

        await stop_event.wait()

    finally:
        logger.info("Shutting down components...")
        await channel.stop()
        await heartbeat.stop()
        logger.info("Goodbye.")


async def run_telegram(brain) -> None:
    """Run the agent as a Telegram bot."""
    from channels.telegram.bot import TelegramBot

    telegram = TelegramBot(brain=brain)
    agent_name = os.getenv("AGENT_NAME", "Atlas")
    logger.info(f"✅ {agent_name} — open Telegram and say hello.")
    await _run_with_heartbeat(brain, telegram)


async def run_discord(brain) -> None:
    """Run the agent as a Discord bot."""
    from channels.discord.bot import DiscordBot

    discord_bot = DiscordBot(brain=brain)
    agent_name = os.getenv("AGENT_NAME", "Atlas")
    logger.info(f"✅ {agent_name} — invite the bot to your server or DM it.")
    await _run_with_heartbeat(brain, discord_bot)


async def run_web(brain) -> None:
    """Run the agent as a web UI."""
    from channels.web.bot import WebBot

    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "7860"))
    web = WebBot(brain=brain, host=host, port=port)
    logger.info(f"Web UI → http://localhost:{port}")
    await _run_with_heartbeat(brain, web)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas — Personal AI Agent")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cli", action="store_true", help="Interactive terminal session")
    mode.add_argument("--discord", action="store_true", help="Discord bot")
    mode.add_argument("--web", action="store_true", help="Browser-based web UI")
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
    elif args.discord:
        await run_discord(brain)
    elif args.web:
        await run_web(brain)
    else:
        await run_telegram(brain)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

    # Brain reload tool sets this env var before sending SIGTERM
    if os.environ.pop("_ATLAS_RESTART", "") == "1":
        import time

        logger.info("Restarting…")
        time.sleep(0.3)
        os.execv(sys.executable, [sys.executable] + sys.argv)
