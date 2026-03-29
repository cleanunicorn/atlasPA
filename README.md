# Atlas — Personal Assistant

A personal AI agent inspired by OpenClaw, built in Python. Model-agnostic, extensible, runs locally.

## Architecture

```
main.py             — Unified CLI entry point (run · logs · setup)
gateway.py          — Legacy entry point (still works)
channels/           — Input/output adapters (Telegram, CLI, Discord, Web)
  telegram/
    bot.py          — Telegram bot with voice support
    transcribe.py   — NVIDIA Parakeet voice transcription (Docker auto-start)
    formatting.py   — Markdown → Telegram HTML converter
brain/              — ReAct reasoning loop (think → act → observe → repeat)
providers/          — LLM abstraction: Anthropic · OpenAI · Ollama · OpenRouter
memory/             — Persistent memory: markdown files + conversation history
skills/             — Core skills (bundled) + addon skills (user-installed)
heartbeat/          — Scheduler for proactive/background tasks
logviewer/          — Web UI for browsing LLM observability logs
transcribe_server/  — Parakeet HTTP microservice (runs inside Docker)
config/             — Configuration and secrets
```

## Phases

- [x] Phase 1 — Core loop: Telegram + CLI + Brain + Model-agnostic providers + Memory + Skills
- [x] Phase 2 — Richer memory: persistent history, relevance filtering, auto-summarisation, `forget` tool
- [x] Phase 3 — More skills: `shell_exec`, `http_request`, `code_runner`, `browser` (Playwright)
- [x] Phase 4 — Heartbeat: cron scheduler, `schedule_job` / `list_jobs` / `delete_job` tools, proactive Telegram push
- [x] Phase 5 — Calendar + channels: Google Calendar skill (list/create/RSVP), voice transcription (Parakeet/Docker), addon skill management, Telegram HTML formatting, unified CLI (`main.py`)
- [ ] Phase 6 — Advanced reasoning

### Phase 6 — Advanced reasoning (planned)

Building blocks to make the agent smarter and more reliable:

| Feature | Description |
|---|---|
| **Planning step** | For complex requests, generate a plan first, then execute step by step |
| **Self-critique** | After producing an answer, evaluate it against the original goal; retry if needed |
| **Clarification** | Ask the user a single focused question when intent is ambiguous, instead of guessing |
| **Parallel tool calls** | Execute independent tool calls concurrently to reduce latency |
| **Confidence gating** | Flag low-confidence responses and offer to verify via a second tool call |
| **Structured output** | Request JSON-mode responses from the LLM for tool-heavy flows |

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Interactive setup wizard (creates config/.env)
uv run python main.py setup

# 3. Start
uv run python main.py run          # Telegram bot (default)
uv run python main.py run --cli    # Interactive terminal
```

### All commands

```bash
uv run python main.py run            # Telegram bot
uv run python main.py run --cli      # Terminal session
uv run python main.py run --discord  # Discord bot
uv run python main.py run --web      # Browser UI (port 7860)
uv run python main.py logs           # LLM log viewer (port 7331)
uv run python main.py setup          # Re-run setup wizard
```

## Adding a Skill

1. Create `skills/<your-skill>/SKILL.md` — describe what the skill does and when to use it
2. Create `skills/<your-skill>/tool.py` — implement `run(**kwargs) -> str` and a `PARAMETERS` dict (JSON Schema)

Skills are auto-discovered at startup. No changes to any existing file needed.

Atlas can also install addon skills at runtime via the `manage_skills` built-in tool:
- Addon skills live in `~/agent-files/skills/` and persist across restarts
- Say *"install a skill that does X"* and Atlas will write + load it immediately

## Adding an LLM Provider

1. Create `providers/<name>.py`
2. Subclass `BaseLLMProvider` from `providers/base.py`
3. Implement `complete(messages, tools, system) -> LLMResponse`
4. Register it in `providers/__init__.py`
5. Set `LLM_PROVIDER=<name>` in `config/.env`

## Voice Messages (Telegram)

Atlas transcribes voice notes using NVIDIA Parakeet (TDT 1.1B) via a Docker sidecar. The container starts automatically on the first voice message.

Requirements:
- Docker installed
- `ffmpeg` for audio conversion: `sudo apt install ffmpeg`
- NVIDIA GPU recommended (falls back to CPU automatically if GPU runtime is unavailable)

No manual container management needed — Atlas handles everything.

## Browser Skill (Playwright)

The `browser` skill requires downloading Chromium once:

```bash
uv run playwright install chromium
```

## LLM Observability

Every request to and response from the LLM is logged to `logs/llm.jsonl`.

Each entry contains:
- `ts` — UTC timestamp
- `provider` / `model` — which provider and model was used
- `request` — system prompt, messages, tools sent, max_tokens
- `response` — content, tool calls, stop reason, token usage

**Disable logging:** set `LLM_LOG_FILE=off` in `config/.env`

**Browse logs:**
```bash
uv run python main.py logs        # → http://localhost:7331
uv run python main.py logs -p 8080
```

## Running Tests

```bash
uv run python -m pytest tests/
```
