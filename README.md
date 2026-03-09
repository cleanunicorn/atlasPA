# Personal Agent

A personal AI agent inspired by OpenClaw, built in Python. Model-agnostic, extensible, and runs locally.

## Architecture

```
gateway.py          — Main process. Routes messages between all components.
channels/           — Input/output adapters (Telegram, CLI, Discord, etc.)
brain/              — ReAct reasoning loop (think → act → observe → repeat)
providers/          — LLM provider abstraction (Claude, OpenAI, Ollama, etc.)
memory/             — Persistent memory via markdown files
skills/             — Plug-in capabilities (each skill = a folder with SKILL.md)
heartbeat/          — Scheduler for proactive/background tasks
config/             — Configuration and secrets
```

## Phases

- [x] Phase 1 — Core loop: Telegram bot + Brain + Model-agnostic providers + Memory skeleton + Skills skeleton
- [ ] Phase 2 — Richer memory (semantic search, long-term context)
- [ ] Phase 3 — More skills (web search, file ops, shell exec)
- [ ] Phase 4 — Heartbeat / proactive scheduler
- [ ] Phase 5 — More channels (Discord, web UI)
- [ ] Phase 6 — Security hardening (approval workflows, sandboxing)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your config
cp config/.env.example config/.env

# 3. Run
python gateway.py
```

## Adding a Skill

1. Create `skills/<your-skill>/SKILL.md` — describe what the skill does and when to use it
2. Create `skills/<your-skill>/tool.py` — implement the `run(input: str) -> str` function
3. Register it in `skills/registry.py`

That's it. No restart needed at runtime.

## Adding an LLM Provider

1. Create `providers/<name>.py`
2. Subclass `BaseLLMProvider` from `providers/base.py`
3. Implement `complete(messages, tools) -> LLMResponse`
4. Set `LLM_PROVIDER=<name>` in your `.env`
