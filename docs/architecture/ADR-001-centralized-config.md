# ADR-001: Centralized Config Class

**Status:** Proposed  
**Date:** 2026-06-29  
**Author:** Architect

---

## Proposal

Replace 40+ scattered `os.getenv()` calls across 10+ modules with a single typed `Config`
dataclass in `config/settings.py` that validates all configuration at startup and exports
a `cfg` singleton.

---

## Why Now

The codebase has grown to 4 LLM providers, 4 channels, 9 skills, and 5 heartbeat modules —
each reading their own configuration independently:

```
brain/engine.py:46        MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
providers/settings.py:8   DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))  ← duplicate
heartbeat/__init__.py:31  MAINTENANCE_HOUR = int(os.getenv("MAINTENANCE_HOUR", "3"))
heartbeat/awareness.py    INTERVAL = int(os.getenv("AWARENESS_INTERVAL_MINUTES", "30"))
memory/store.py:26        CONTEXT_MAX_INJECTED = int(os.getenv("CONTEXT_MAX_INJECTED", "15"))
memory/store.py:49        HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "200"))
... 35+ more call sites
```

Concrete pain points:
- **Silent failures:** a typo in `LLM_MAX_TOKENS` ("abc") raises `ValueError` deep in a
  request handler, not at boot.
- **Duplicate defaults:** `"8192"` for `LLM_MAX_TOKENS` appears in both `brain/engine.py`
  and `providers/settings.py`. If one is changed, the other drifts.
- **Test pollution:** every test that exercises config-dependent code must monkey-patch
  `os.environ`, making tests order-dependent and hard to parallelize.
- **No inventory:** there is no single place to see all supported config keys.
- **Prior art ignored:** `providers/settings.py` was created to centralize provider config
  but was never adopted by the rest of the codebase.

---

## Before / After

### Before

```
                os.environ (unvalidated, scattered)
                         │
     ┌───────────────────┼───────────────────┐
     ▼                   ▼                   ▼
brain/engine.py    heartbeat/*.py      memory/store.py
os.getenv(         os.getenv(          os.getenv(
  "LLM_MAX_          "MAINTENANCE_       "CONTEXT_MAX_
   TOKENS","8192")    HOUR","3")          INJECTED","15")

(10+ more modules — each reads independently, no validation)
```

### After

```
                os.environ
                     │
                     ▼
           config/settings.py
           ┌─────────────────┐
           │  Config(         │  ← validates at startup, typed attrs
           │   llm_max_tokens │  ← fail-fast on bad values
           │   maintenance_hr │  ← single source of defaults
           │   context_max_   │
           │   injected ...   │
           │  )               │
           │  cfg = Config()  │  ← singleton imported everywhere
           └────────┬─────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
brain/engine.py  heartbeat/*.py  memory/store.py
cfg.llm_max_     cfg.maintenance  cfg.context_max_
tokens           _hour            injected
```

---

## Data Model

```python
# config/settings.py

from dataclasses import dataclass, field
import os

@dataclass
class Config:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "8192")))

    # Memory
    context_max_injected: int = field(default_factory=lambda: int(os.getenv("CONTEXT_MAX_INJECTED", "15")))
    context_summary_threshold: int = field(default_factory=lambda: int(os.getenv("CONTEXT_SUMMARY_THRESHOLD", "20")))
    history_max_messages: int = field(default_factory=lambda: int(os.getenv("HISTORY_MAX_MESSAGES", "200")))

    # Heartbeat
    maintenance_hour: int = field(default_factory=lambda: int(os.getenv("MAINTENANCE_HOUR", "3")))
    awareness_interval_minutes: int = field(default_factory=lambda: int(os.getenv("AWARENESS_INTERVAL_MINUTES", "30")))
    update_check_interval_hours: int = field(default_factory=lambda: int(os.getenv("UPDATE_CHECK_INTERVAL_HOURS", "1")))

    # Logging
    llm_log_file: str = field(default_factory=lambda: os.getenv("LLM_LOG_FILE", "logs/llm.jsonl"))

    # Embedding
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", ""))

    def __post_init__(self):
        """Validate at construction time — fail fast before any request is served."""
        if self.llm_max_tokens <= 0:
            raise ValueError(f"LLM_MAX_TOKENS must be positive, got {self.llm_max_tokens}")
        if not 0 <= self.maintenance_hour <= 23:
            raise ValueError(f"MAINTENANCE_HOUR must be 0-23, got {self.maintenance_hour}")
        if self.awareness_interval_minutes <= 0:
            raise ValueError(f"AWARENESS_INTERVAL_MINUTES must be positive")
        if self.context_max_injected <= 0:
            raise ValueError(f"CONTEXT_MAX_INJECTED must be positive")

cfg = Config()  # loaded once at import time
```

---

## API Contract Changes

None. This is an internal structural change. No public endpoints, no `providers/base.py`
interface, no skill `run()` signatures are affected.

---

## Migration Plan (backward-compatible, 5 stages)

### Stage 1 — Create `config/settings.py` (no callers yet)
- Define `Config` dataclass with all known keys
- Validate in `__post_init__`
- Export `cfg` singleton
- Add tests: `Config(llm_max_tokens=0)` raises, `Config(llm_max_tokens=100)` succeeds
- **Zero risk** — nothing uses it yet

### Stage 2 — Migrate `providers/` (lowest friction)
- `providers/settings.py` → delete; its one variable (`DEFAULT_MAX_TOKENS`) absorbed into
  `cfg.llm_max_tokens`
- Update `providers/__init__.py`, `providers/anthropic_provider.py`,
  `providers/openai_provider.py`, `providers/logging_provider.py`
- **Validation:** existing provider tests still pass

### Stage 3 — Migrate `heartbeat/`
- `heartbeat/__init__.py`, `heartbeat/awareness.py`, `heartbeat/maintenance.py`,
  `heartbeat/updater.py`
- **Validation:** heartbeat unit tests still pass

### Stage 4 — Migrate `memory/` and `brain/`
- `memory/store.py`, `memory/history.py`, `memory/retriever.py`, `memory/summariser.py`
- `brain/engine.py`, `brain/compactor.py`
- **Validation:** full test suite passes

### Stage 5 — Cleanup
- Remove any remaining `os.getenv()` calls for config keys (leave only non-config env reads
  like `ANTHROPIC_API_KEY` in the providers themselves — credentials are not config)
- Update `CLAUDE.md` config section to reference `config/settings.py`

Each stage is a separate PR or commit; rollback is trivial (revert the module-level change).

---

## Performance Impact

- **Runtime:** +0ms per request. `cfg` is a module-level singleton; attribute access is a
  dict lookup (~50ns).
- **Startup:** +1ms for `__post_init__` validation (negligible).
- **Tests:** improved — no `os.environ` mutation needed; tests construct `Config(...)` with
  explicit values and pass it via dependency injection.

---

## Risk Assessment

**Risk: LOW**

| Factor | Assessment |
|--------|-----------|
| Behavior change | None — same defaults, same semantics |
| API contract change | None |
| Data migration | None |
| Rollback | Per-module; revert a file, tests re-pass |
| Blast radius | Internal only; no external consumers |

The only non-zero risk is introducing a validation error that prevents boot on a running
instance where an env var is misconfigured. This is actually the desired behavior (fail fast
instead of silently using a wrong value mid-request), but the operator should be aware.

---

## Test Strategy

```python
# tests/test_config.py

def test_valid_config():
    cfg = Config(llm_max_tokens=4096, maintenance_hour=3)
    assert cfg.llm_max_tokens == 4096

def test_invalid_max_tokens():
    with pytest.raises(ValueError):
        Config(llm_max_tokens=0)

def test_invalid_maintenance_hour():
    with pytest.raises(ValueError):
        Config(maintenance_hour=25)

def test_defaults_match_legacy():
    cfg = Config()
    assert cfg.llm_max_tokens == 8192   # matches old os.getenv default
    assert cfg.context_max_injected == 15
    assert cfg.maintenance_hour == 3
```

---

## Who Must Approve

Single-developer project. No infra, security, or team-lead sign-off required.
No external API contracts are changed.

---

## Tradeoffs

| Cost | Benefit |
|------|---------|
| ~3-4h migration work across 5 stages | Single inventory of all config keys |
| One more import in each module | Typed config (IDE autocomplete, no string typos) |
| Credential env vars (`ANTHROPIC_API_KEY`) stay in providers — slightly inconsistent | Validation at boot; errors surface before first request |
| | Tests no longer need `os.environ` mutation |
| | Eliminates duplicate defaults (`"8192"` in two files) |

**Verdict:** Positive ROI. The migration is mechanical, the benefit compounds with every new
provider, channel, or skill added.
