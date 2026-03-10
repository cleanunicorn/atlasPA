#!/usr/bin/env python3
"""
main.py

Unified entry point for Atlas.

    uv run python main.py run            Start the Telegram bot (default)
    uv run python main.py run --cli      Interactive terminal session
    uv run python main.py run --discord  Discord bot
    uv run python main.py run --web      Browser-based web UI
    uv run python main.py logs           Browse LLM logs in the browser
    uv run python main.py setup          Interactive setup wizard
"""

import asyncio
import os
import shutil
import signal
import sys
from pathlib import Path
from urllib.parse import urlparse

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

ROOT = Path(__file__).parent
ENV_FILE = ROOT / "config" / ".env"
ENV_EXAMPLE = ROOT / "config" / ".env.example"

app = typer.Typer(
    name="atlas",
    help="Atlas — Personal AI Agent",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    """Read .env without mutating os.environ."""
    from dotenv import dotenv_values
    return dict(dotenv_values(ENV_FILE)) if ENV_FILE.exists() else {}


# ── Pre-flight checks ─────────────────────────────────────────────────────────

class _Check:
    def __init__(self, label: str, ok: bool, note: str = "", required: bool = True):
        self.label = label
        self.ok = ok
        self.note = note
        self.required = required


def _preflight(env: dict, mode: str) -> tuple[list[_Check], bool]:
    checks: list[_Check] = []

    # config/.env
    checks.append(_Check("config/.env", ENV_FILE.exists(), required=True,
                         note="Run: atlas setup" if not ENV_FILE.exists() else ""))
    if not ENV_FILE.exists():
        return checks, False

    # LLM provider
    provider = (env.get("LLM_PROVIDER") or "ollama").lower()
    checks.append(_Check(f"LLM_PROVIDER = {provider}", True, required=False))

    if provider == "anthropic":
        ok = bool(env.get("ANTHROPIC_API_KEY", "").strip())
        checks.append(_Check("ANTHROPIC_API_KEY", ok, required=True,
                             note="Set in config/.env" if not ok else ""))
    elif provider == "openai":
        ok = bool(env.get("OPENAI_API_KEY", "").strip())
        checks.append(_Check("OPENAI_API_KEY", ok, required=True,
                             note="Set in config/.env" if not ok else ""))
    elif provider == "openrouter":
        ok = bool(env.get("OPENROUTER_API_KEY", "").strip())
        checks.append(_Check("OPENROUTER_API_KEY", ok, required=True,
                             note="Set in config/.env" if not ok else ""))
    elif provider == "ollama":
        base_url = env.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        parsed = urlparse(base_url)
        root_url = f"{parsed.scheme}://{parsed.netloc}"
        try:
            import httpx
            r = httpx.get(root_url, timeout=2)
            reachable = r.status_code < 500
        except Exception:
            reachable = False
        checks.append(_Check(f"Ollama reachable at {root_url}", reachable, required=True,
                             note="Run: ollama serve" if not reachable else ""))

    # Channel-specific
    if mode == "telegram":
        ok = bool(env.get("TELEGRAM_BOT_TOKEN", "").strip())
        checks.append(_Check("TELEGRAM_BOT_TOKEN", ok, required=True,
                             note="Get one from @BotFather" if not ok else ""))
        ok = bool(env.get("TELEGRAM_ALLOWED_USERS", "").strip())
        checks.append(_Check("TELEGRAM_ALLOWED_USERS", ok, required=False,
                             note="⚠ Anyone can message your bot!" if not ok else ""))
    elif mode == "discord":
        ok = bool(env.get("DISCORD_BOT_TOKEN", "").strip())
        checks.append(_Check("DISCORD_BOT_TOKEN", ok, required=True,
                             note="Set in config/.env" if not ok else ""))

    # Optional features
    docker_ok = shutil.which("docker") is not None
    checks.append(_Check("Docker (voice transcription)", docker_ok, required=False,
                         note="Install Docker to enable voice messages" if not docker_ok else ""))
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    checks.append(_Check("ffmpeg (audio conversion)", ffmpeg_ok, required=False,
                         note="sudo apt install ffmpeg" if not ffmpeg_ok else ""))

    all_required_ok = all(c.ok for c in checks if c.required)
    return checks, all_required_ok


def _show_preflight(checks: list[_Check], mode: str) -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("", width=2)
    table.add_column("Check")
    table.add_column("Note", style="dim")

    for c in checks:
        if c.ok:
            icon = "[green]✓[/green]"
        elif c.required:
            icon = "[red]✗[/red]"
        else:
            icon = "[yellow]⚠[/yellow]"
        table.add_row(icon, c.label, c.note)

    console.print(Panel(table, title=f"[bold]Atlas — {mode} mode[/bold]",
                        border_style="blue", expand=False))


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    cli: bool = typer.Option(False, "--cli", help="Interactive terminal session"),
    discord: bool = typer.Option(False, "--discord", help="Discord bot"),
    web: bool = typer.Option(False, "--web", help="Browser-based web UI"),
    skip_checks: bool = typer.Option(False, "--skip-checks", help="Skip pre-flight checks"),
) -> None:
    """Start the Atlas agent."""
    mode = "cli" if cli else "discord" if discord else "web" if web else "telegram"

    env = _load_env()

    if not skip_checks:
        checks, ok = _preflight(env, mode)
        _show_preflight(checks, mode)

        if not ok:
            failed = [c.label for c in checks if c.required and not c.ok]
            console.print(f"\n[red]Cannot start — missing required config:[/red] {', '.join(failed)}")
            if Confirm.ask("\nRun setup wizard now?", default=True):
                _run_setup()
                env = _load_env()
                checks, ok = _preflight(env, mode)
                if not ok:
                    console.print("[red]Setup incomplete. Fix the issues above and try again.[/red]")
                    raise typer.Exit(1)
            else:
                raise typer.Exit(1)
        else:
            warnings = [c for c in checks if not c.ok and not c.required]
            if warnings:
                console.print("[dim]Some optional features are unavailable (see ⚠ above)[/dim]")

    # Load env into os.environ before starting the agent
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)

    console.print(f"\n[green]Starting Atlas ({mode} mode)…[/green]\n")
    asyncio.run(_run_agent(mode))


@app.command()
def logs(
    port: int = typer.Option(7331, "--port", "-p", help="Port to serve on"),
) -> None:
    """Browse LLM observability logs in the browser."""
    import uvicorn

    log_dir = ROOT / "logs"
    jsonl_files = list(log_dir.glob("*.jsonl")) if log_dir.exists() else []

    if not jsonl_files:
        console.print("[yellow]⚠ No log files found in logs/[/yellow]")
        console.print("[dim]Logs appear after Atlas processes its first message.[/dim]\n")
    else:
        console.print(f"Found [green]{len(jsonl_files)}[/green] log file(s) in [dim]{log_dir}[/dim]")

    console.print(f"Log viewer → [bold blue]http://localhost:{port}[/bold blue]")
    console.print("Press [bold]Ctrl+C[/bold] to stop.\n")

    os.environ["LOG_VIEWER_PORT"] = str(port)
    sys.path.insert(0, str(ROOT))
    uvicorn.run("logviewer.server:app", host="0.0.0.0", port=port, reload=False)


@app.command()
def setup() -> None:
    """Interactive setup wizard — create or update config/.env."""
    _run_setup()
    console.print("\n[bold green]Setup complete![/bold green] Run [bold]atlas run[/bold] to start.")


# ── Setup wizard ──────────────────────────────────────────────────────────────

def _run_setup() -> None:
    existing = _load_env()

    console.print(Panel(
        "[bold]Atlas Setup Wizard[/bold]\n"
        "[dim]Press Enter to keep the current value. Ctrl+C to cancel.[/dim]",
        border_style="blue", expand=False,
    ))

    config = dict(existing)  # preserve any keys the wizard doesn't ask about

    # ── Agent identity ───────────────────────────────────────────────────────
    console.print("\n[bold cyan]Agent Identity[/bold cyan]")
    config["AGENT_NAME"] = Prompt.ask(
        "Agent name", default=existing.get("AGENT_NAME", "Atlas"))
    config["AGENT_TIMEZONE"] = Prompt.ask(
        "Timezone", default=existing.get("AGENT_TIMEZONE", "UTC"))

    # ── LLM provider ─────────────────────────────────────────────────────────
    console.print("\n[bold cyan]LLM Provider[/bold cyan]")
    console.print("[dim]Options: anthropic · openai · ollama · openrouter[/dim]")
    config["LLM_PROVIDER"] = Prompt.ask(
        "Provider", default=existing.get("LLM_PROVIDER", "ollama"),
        choices=["anthropic", "openai", "ollama", "openrouter"])

    provider = config["LLM_PROVIDER"]

    if provider == "anthropic":
        config["ANTHROPIC_API_KEY"] = Prompt.ask(
            "Anthropic API key", default=existing.get("ANTHROPIC_API_KEY", ""), password=True)
        config["ANTHROPIC_MODEL"] = Prompt.ask(
            "Model", default=existing.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    elif provider == "openai":
        config["OPENAI_API_KEY"] = Prompt.ask(
            "OpenAI API key", default=existing.get("OPENAI_API_KEY", ""), password=True)
        config["OPENAI_MODEL"] = Prompt.ask(
            "Model", default=existing.get("OPENAI_MODEL", "gpt-4o"))
    elif provider == "openrouter":
        config["OPENROUTER_API_KEY"] = Prompt.ask(
            "OpenRouter API key", default=existing.get("OPENROUTER_API_KEY", ""), password=True)
        config["OPENROUTER_MODEL"] = Prompt.ask(
            "Model", default=existing.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"))
    elif provider == "ollama":
        config["OLLAMA_BASE_URL"] = Prompt.ask(
            "Ollama base URL", default=existing.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        config["OLLAMA_MODEL"] = Prompt.ask(
            "Model", default=existing.get("OLLAMA_MODEL", "qwen3.5:9b-q8_0"))

    # ── Telegram ─────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Telegram[/bold cyan]")
    config["TELEGRAM_BOT_TOKEN"] = Prompt.ask(
        "Bot token (from @BotFather)", default=existing.get("TELEGRAM_BOT_TOKEN", ""), password=True)
    config["TELEGRAM_ALLOWED_USERS"] = Prompt.ask(
        "Allowed user IDs (comma-separated, empty = allow anyone)",
        default=existing.get("TELEGRAM_ALLOWED_USERS", ""))

    # ── Voice (optional) ─────────────────────────────────────────────────────
    console.print("\n[bold cyan]Voice Transcription[/bold cyan] [dim](optional — requires Docker)[/dim]")
    if Confirm.ask("Configure Parakeet voice transcription?", default=bool(existing.get("PARAKEET_URL"))):
        config["PARAKEET_URL"] = Prompt.ask(
            "Parakeet server URL", default=existing.get("PARAKEET_URL", "http://localhost:8765"))

    # ── Logging ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Logging[/bold cyan]")
    config["LOG_LEVEL"] = Prompt.ask(
        "Log level", default=existing.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    _write_env(config)
    console.print(f"\n[green]✓[/green] Saved to [bold]{ENV_FILE}[/bold]")


def _write_env(config: dict[str, str]) -> None:
    """Write config to .env, grouped into sections. Empty values are skipped."""
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        ("# ── Agent identity ────────────────────────────────────────────────────────────",
         ["AGENT_NAME", "AGENT_TIMEZONE"]),
        ("# ── LLM Provider ─────────────────────────────────────────────────────────────",
         ["LLM_PROVIDER"]),
        ("# ── Anthropic (Claude) ────────────────────────────────────────────────────────",
         ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"]),
        ("# ── OpenAI (GPT) ─────────────────────────────────────────────────────────────",
         ["OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"]),
        ("# ── OpenRouter ────────────────────────────────────────────────────────────────",
         ["OPENROUTER_API_KEY", "OPENROUTER_MODEL"]),
        ("# ── Ollama (local models) ─────────────────────────────────────────────────────",
         ["OLLAMA_MODEL", "OLLAMA_BASE_URL"]),
        ("# ── Telegram ─────────────────────────────────────────────────────────────────",
         ["TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS"]),
        ("# ── Voice transcription (Parakeet) ────────────────────────────────────────────",
         ["PARAKEET_URL", "PARAKEET_IMAGE"]),
        ("# ── Logging ───────────────────────────────────────────────────────────────────",
         ["LOG_LEVEL"]),
    ]

    known_keys: set[str] = {k for _, keys in sections for k in keys}
    extra = {k: v for k, v in config.items() if k not in known_keys and v}

    lines: list[str] = []
    for header, keys in sections:
        section_lines = [f"{k}={config[k]}\n" for k in keys if config.get(k)]
        if section_lines:
            lines.append(f"{header}\n")
            lines.extend(section_lines)
            lines.append("\n")

    if extra:
        lines.append("# ── Other ─────────────────────────────────────────────────────────────────────\n")
        for k, v in extra.items():
            lines.append(f"{k}={v}\n")

    ENV_FILE.write_text("".join(lines).rstrip("\n") + "\n")


# ── Agent runner ──────────────────────────────────────────────────────────────

async def _run_agent(mode: str) -> None:
    import logging

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from providers import get_provider
    from memory import MemoryStore
    from skills.registry import SkillRegistry
    from brain import Brain

    provider = get_provider()
    memory = MemoryStore()
    skills = SkillRegistry()
    brain = Brain(provider=provider, memory=memory, skills=skills)

    if mode == "cli":
        from channels.cli.bot import CLIBot
        await CLIBot(brain=brain).start()
        return

    if mode == "discord":
        from channels.discord.bot import DiscordBot
        channel = DiscordBot(brain=brain)
    elif mode == "web":
        from channels.web.bot import WebBot
        channel = WebBot(
            brain=brain,
            host=os.getenv("WEB_HOST", "0.0.0.0"),
            port=int(os.getenv("WEB_PORT", "7860")),
        )
    else:  # telegram
        from channels.telegram.bot import TelegramBot
        channel = TelegramBot(brain=brain)

    await _run_with_heartbeat(brain, channel)


async def _run_with_heartbeat(brain, channel) -> None:
    from heartbeat import Heartbeat

    heartbeat = Heartbeat(brain=brain, notify_callback=channel.push_message)
    brain.heartbeat = heartbeat

    stop_event = asyncio.Event()

    def _shutdown(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        await heartbeat.start()
        await channel.start()
        agent_name = os.getenv("AGENT_NAME", "Atlas")
        console.print(f"[bold green]✓ {agent_name} is running![/bold green] Press Ctrl+C to stop.")
        await stop_event.wait()
    finally:
        console.print("\n[dim]Shutting down…[/dim]")
        await channel.stop()
        await heartbeat.stop()
        console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    app()
