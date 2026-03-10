# Atlas — Personal AI Agent

A personal AI agent inspired by OpenClaw, built in Python. Model-agnostic, extensible, runs locally.

## Architecture

```
gateway.py          — Main process. Run with --cli for terminal mode.
channels/           — Input/output adapters (Telegram, CLI)
brain/              — ReAct reasoning loop (think → act → observe → repeat)
providers/          — LLM provider abstraction (Claude, OpenAI, Ollama)
memory/             — Persistent memory: markdown files + conversation history
skills/             — Plug-in capabilities (each skill = a folder with SKILL.md)
heartbeat/          — Scheduler for proactive/background tasks
config/             — Configuration and secrets
```

## Phases

- [x] Phase 1 — Core loop: Telegram + CLI + Brain + Model-agnostic providers + Memory + Skills
- [x] Phase 2 — Richer memory: persistent history, relevance filtering, auto-summarisation, `forget` tool
- [x] Phase 3 — More skills: `shell_exec`, `http_request`, `code_runner`, `browser` (Playwright)
- [x] Phase 4 — Heartbeat: cron scheduler, `schedule_job` / `list_jobs` / `delete_job` tools, proactive Telegram push
- [ ] Phase 5 — Calendar manager
- [ ] Phase 6 — Advanced reasoning

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in your config
cp config/.env.example config/.env
# Edit config/.env — set LLM_PROVIDER, API keys, TELEGRAM_BOT_TOKEN

# 3. Run (CLI mode, no Telegram needed)
uv run python gateway.py --cli

# 3b. Run as Telegram bot
uv run python gateway.py
```

## Adding a Skill

1. Create `skills/<your-skill>/SKILL.md` — describe what the skill does and when to use it
2. Create `skills/<your-skill>/tool.py` — implement `run(**kwargs) -> str` and a `PARAMETERS` dict (JSON Schema)

Skills are auto-discovered at startup. No changes to any existing file needed.

## Adding an LLM Provider

1. Create `providers/<name>.py`
2. Subclass `BaseLLMProvider` from `providers/base.py`
3. Implement `complete(messages, tools, system) -> LLMResponse`
4. Register it in `providers/__init__.py`
5. Set `LLM_PROVIDER=<name>` in `config/.env`

## Browser skill (Playwright)

The `browser` skill requires downloading Chromium once:

```bash
uv run playwright install chromium
```

## Running tests

```bash
uv run python -m pytest tests/
```
