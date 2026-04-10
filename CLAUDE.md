# Atlas — Development Guide for Claude

## Project Overview

Atlas is a personal AI agent built in Python. It is model-agnostic (supports Anthropic, OpenAI, Ollama, OpenRouter), extensible via a skills system, and can run across multiple channels (Telegram, CLI, Discord, Web).

## Setup

```bash
# Install dependencies (uses uv)
uv sync

# Interactive setup (creates config/.env)
uv run python main.py setup

# Install Playwright for browser skill
uv run playwright install chromium
```

## Running

```bash
uv run python main.py run          # Telegram bot (default)
uv run python main.py run --cli    # Interactive terminal
uv run python main.py run --discord
uv run python main.py run --web    # Browser UI on port 7860
uv run python main.py logs         # LLM log viewer on port 7331
```

Or via `make`:

```bash
make install    # uv sync
make run        # Telegram bot
make cli        # Terminal session
make test       # Run test suite
```

## Testing

```bash
uv run python -m pytest tests/ -v
# or
make test
```

All tests live in `tests/`. Use `pytest-asyncio` for async tests (already configured in `pyproject.toml`).

## Architecture

```
main.py             — Unified CLI entry point (run · logs · setup)
gateway.py          — Legacy entry point
channels/           — I/O adapters: Telegram, CLI, Discord, Web
brain/              — ReAct loop: think → act → observe → repeat
  engine.py         — Core reasoning loop
  tools.py          — Built-in tools (remember, forget, schedule, etc.)
providers/          — LLM abstraction layer
  base.py           — BaseLLMProvider interface
  anthropic_provider.py
  openai_provider.py
  ollama_provider.py
  openrouter_provider.py
memory/             — Persistent memory: markdown files + conversation history
skills/             — Core and addon skills (auto-discovered at startup)
heartbeat/          — Cron scheduler for background/proactive tasks
logviewer/          — Web UI for LLM observability logs
config/             — Secrets and configuration (.env)
```

## Configuration

Copy `config/.env.example` to `config/.env` and fill in the required values. Key env vars:

- `LLM_PROVIDER` — `anthropic`, `openai`, `ollama`, or `openrouter`
- `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL`
- `OPENAI_API_KEY` / `OPENAI_MODEL`
- `OLLAMA_MODEL` / `OLLAMA_BASE_URL`
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_ALLOWED_USERS`

## Adding a Skill

1. Create `skills/<name>/SKILL.md` — describe what the skill does and when to use it
2. Create `skills/<name>/tool.py` — implement `run(**kwargs) -> str` and a `PARAMETERS` dict (JSON Schema)

Skills are auto-discovered at startup. No changes to existing files are needed.

## Adding an LLM Provider

1. Create `providers/<name>.py`
2. Subclass `BaseLLMProvider` from `providers/base.py`
3. Implement `complete(messages, tools, system) -> LLMResponse`
4. Register it in `providers/__init__.py`
5. Set `LLM_PROVIDER=<name>` in `config/.env`

## Code Style

- Linting: `ruff` (included as a dependency)
- Python 3.11+
- Async-first: the brain loop is natively async; use `asyncio.to_thread` to bridge sync skill `run()` functions
- No framework magic — skills just need a `run(**kwargs) -> str` function and a `PARAMETERS` dict
