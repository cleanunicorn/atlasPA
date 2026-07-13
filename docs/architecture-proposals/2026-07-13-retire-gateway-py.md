# 2026-07-13 — Retire `gateway.py` as a full orchestration implementation

**Status:** Proposed
**Author:** Architect (automated structural-debt scan)

## Proposal

`gateway.py` (180 lines) and `main.py`'s `_run_agent()` / `_run_with_heartbeat()`
(~90 lines, `main.py:588-660`) are two **independent, hand-synchronized
implementations** of the same startup sequence: wire up `provider → memory →
skills → brain`, pick a channel (`cli`/`discord`/`web`/telegram-default), start
the heartbeat scheduler, install SIGINT/SIGTERM handlers, and shut everything
down gracefully.

Collapse `gateway.py` into a thin compatibility shim that forwards to
`main.py run`, so there is exactly **one** place that wires the agent
together.

## Why now?

This is already labeled debt, not a hypothetical: README.md:9 and
CLAUDE.md:53 both call it a "Legacy entry point (still works)", and
`skills/self_inspect/tool.py:92` surfaces that same label to the LLM itself
when a user asks Atlas to describe its own codebase. Six files reference
`gateway.py` today (`README.md`, `CLAUDE.md`, `skills/registry.py`,
`skills/self_inspect/tool.py`, `channels/cli/bot.py` docstring,
`heartbeat/__init__.py` docstring) — evidence this is a known, load-bearing
fork, not dead code someone forgot to delete.

`main.py` is the actively developed path (`make run`, `make cli` both use
it) and is now a strict superset: it added the `setup` wizard, `logs`
viewer, file-watch auto-restart, and richer console output — none of which
`gateway.py` has. Recent orchestration-adjacent fixes (e.g. `7c07774` "await
coroutine results returned by sync skill wrappers", the `BaseChannel` ACL
extraction in `e7b302f`) only had to touch one call site because they lived
in shared modules — but the *next* fix to signal handling, channel
selection, or heartbeat wiring would have to be applied twice, and nothing
would catch it if someone forgot the second copy. That's the concrete
failure mode this proposal removes.

## Before / after

```
BEFORE
──────
main.py run [--cli|--discord|--web]        gateway.py [--cli|--discord|--web]
        │                                            │
        ▼                                            ▼
  _run_agent()                                    main()
        │                                            │
        ├─ get_provider()                            ├─ get_provider()
        ├─ MemoryStore()                              ├─ MemoryStore()
        ├─ SkillRegistry()                            ├─ SkillRegistry()
        ├─ Brain(...)                                 ├─ Brain(...)
        ├─ pick channel by flag                       ├─ pick channel by flag
        └─ _run_with_heartbeat()                      └─ _run_with_heartbeat()
              (signal handlers, heartbeat,                   (signal handlers, heartbeat,
               start/stop, near-identical)                    start/stop, near-identical)

Two independent copies of the same ~90 lines. A fix applied to one
silently does not apply to the other.

AFTER
─────
gateway.py                    main.py run [--cli|--discord|--web]
   (thin shim: parses               │
    the same flags, then            ▼
    calls main.app(...))      _run_agent()   ← single source of truth
        │                           │
        └──────────────────────────►
```

## Migration plan (backward-compatible)

This proposal covers **Stage 1 only** — it is a single, self-contained,
<1-day change:

1. **Stage 1 (this proposal):** Replace the body of `gateway.py` with a
   ~15-line shim: parse the same `--cli` / `--discord` / `--web` flags it
   parses today, then invoke `main.py`'s `run` command with the equivalent
   arguments (either via `typer`'s `CliRunner`-style programmatic
   invocation, or `os.execv(sys.executable, [sys.executable, "main.py",
   "run", ...])`). `python gateway.py --cli` continues to work exactly as
   before — same flags, same behavior — because it now *runs* the same
   code, not a copy of it. Update `README.md` / `CLAUDE.md` to describe
   `gateway.py` as a "compatibility shim for `main.py run`" instead of
   "legacy entry point," and refresh the stale `channels/cli/bot.py`
   docstring example (`python gateway.py --cli` → `python main.py run
   --cli`).
2. **Stage 2 (separate future proposal, not in scope here):** once the
   shim has been live for a release or two with no reports of a divergent
   invocation someone relied on, delete `gateway.py` outright and drop the
   remaining references in `skills/registry.py`,
   `skills/self_inspect/tool.py`, and `heartbeat/__init__.py`.

Each stage is independently revertible with `git revert` — Stage 1 touches
one file's implementation plus doc comments; nothing downstream depends on
`gateway.py`'s internals (no imports of it exist anywhere in the tree).

## Performance impact

None. This is a process-startup script, not a hot path — the shim adds at
most a few milliseconds of argument-forwarding overhead once per process
launch.

## Risk level

**Low.**
- No `/api`-equivalent contract changes (n/a for this project — no HTTP API
  surface changes; `WebBot`'s routes are untouched).
- No data model or migration involved.
- `main.py run` is already the primary, most-exercised path (`make run`,
  `make cli`), so routing `gateway.py` through it doesn't introduce new
  untested code paths — it removes an under-exercised duplicate.
- The only user-visible risk is an exact-invocation mismatch (e.g. some
  automation script calling `python gateway.py` with a flag combination
  `main.py run` doesn't support), mitigated by keeping 1:1 flag parity in
  the shim.

## Test strategy

- New `tests/test_gateway_shim.py`: for each of `--cli`, `--discord`,
  `--web`, and no-flag (telegram default), assert the shim resolves to the
  same mode `main.py run` would use — mocking the actual channel/bot
  construction so no real bot starts.
- Existing `tests/test_channels.py` and `tests/test_brain.py` are
  untouched and continue to pass unmodified (`Brain`/channel construction
  logic is not being changed, only which file drives it).
- Manual smoke test before merging Stage 1: run `python gateway.py --cli`
  and `python main.py run --cli` side-by-side and confirm identical
  startup banner and behavior.

## Who should review

This is a single-maintainer project (no separate infra/backend-lead/security
roles are defined in `CLAUDE.md`) — recommend the repo owner
(`cleanunicorn`) review directly rather than routing to a team.

## Timeline estimate

Half a day: ~2 hours to write the shim + update the six doc references,
~1 hour for the new test file, ~1 hour manual verification across all four
run modes.
