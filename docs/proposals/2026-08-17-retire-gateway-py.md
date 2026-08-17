# 🏗️ Architect proposal — retire the legacy `gateway.py` entry point

**Status:** draft, awaiting approval — no code changes included beyond this document.

## Proposal

Delete `gateway.py`. `main.py` already reimplements 100% of its behavior and is
the only entry point actually shipped (Makefile, systemd service, README
"Running" section, `.vscode/launch.json`).

## Why now

- `CLAUDE.md` / `README.md` label `gateway.py` "Legacy entry point (still
  works)" — but it's a live, self-contained ~180-line script, not a stub.
  It independently wires `Provider → Memory → Skills → Brain → Channels →
  Heartbeat` and independently implements the update/restart signal path
  (`_ATLAS_RESTART` env var + `SIGTERM` + `os.execv`).
- `main.py` (`_run_agent` / `_run_with_heartbeat`, main.py:588-640ish)
  reimplements the *same* wiring — the two files are line-for-line near
  duplicates: channel selection, heartbeat start/stop, SIGINT/SIGTERM
  handling, the restart dance.
- Nothing calls `gateway.py` anymore. `grep -rn "import gateway"` across the
  repo (source + tests) returns zero hits. The systemd unit installed by
  `make service-install` execs `uv run python main.py run --skip-checks`;
  the Makefile, README run instructions, and `.vscode/launch.json` all point
  at `main.py` exclusively.
- The two copies have **already drifted**: `main.py`'s restart check covers
  both the file-watcher path and the tool-triggered path
  (`if _restart_requested or os.environ.pop("_ATLAS_RESTART", "") == "1"`),
  while `gateway.py` only checks the env var. With no forcing function to
  keep them in sync, any future change to channel/heartbeat wiring applied
  only to `main.py` (the file everyone actually touches) silently leaves
  `gateway.py` further out of date — dead code that *looks* alive because it
  still runs, just not correctly.

## Before / after

```
Before:
  make run  ──▶ main.py  (run/_run_agent/_run_with_heartbeat)  ─┐
  systemd   ──▶                                                  ├─▶ Provider → Memory → Skills → Brain → Channel → Heartbeat   (wired once, current)
                                                                  │
                gateway.py (main/_run_with_heartbeat)  ──────────┘─▶ Provider → Memory → Skills → Brain → Channel → Heartbeat   (wired again, unreferenced, drifting)

After:
  make run  ──▶ main.py  (run/_run_agent/_run_with_heartbeat)  ──▶ Provider → Memory → Skills → Brain → Channel → Heartbeat   (wired once)
  systemd   ──▶
```

## Migration plan (backward-compatible, single revertible commit)

1. Delete `gateway.py`.
2. Update `README.md`'s architecture listing to drop the
   `gateway.py — Legacy entry point (still works)` line.
3. Update `CLAUDE.md`'s architecture tree the same way.
4. No env var, CLI flag, or systemd unit changes needed — `make
   service-install` already targets `main.py`.

Rollback: `git revert` the single commit. The deleted file has no runtime
dependents, so revert is a clean, no-op restore.

## Performance impact

None. No runtime code path changes — `main.py run` behavior, startup time,
and memory are unaffected. Net effect: -180 lines of dead-weight surface
area maintainers have to keep mentally in sync.

## Risk: **low**

`grep -rn "import gateway\|from gateway"` across source and tests returns
zero hits. Systemd service, Makefile, README, and the VS Code launch config
already exclusively target `main.py`. The only residual risk is an
out-of-repo automation script someone runs manually that still invokes
`python gateway.py` directly — mitigated by keeping this to one revertible
commit.

## Who must approve

Repo maintainer (`cleanunicorn`) — single-owner personal-agent repo, no
separate infra/backend/security leads to loop in.

## Test strategy

- `uv run python -m pytest tests/ -v` — no test imports `gateway`; a green
  run confirms nothing in the suite depended on it.
- Manual smoke after merge: `make run` and `make cli` (both routed through
  `main.py run`) still start the agent normally.
