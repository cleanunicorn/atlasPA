"""
paths.py

Single source of truth for all persistent data paths.

Everything the agent stores (config, memory, user files) lives under DATA_DIR
(~/agent-files/) so there is one place to back up, move, or inspect.
"""

from pathlib import Path

DATA_DIR = Path.home() / "agent-files"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_DIR = DATA_DIR / "config"
ENV_FILE = CONFIG_DIR / ".env"
JOBS_FILE = CONFIG_DIR / "jobs.json"

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_DIR = DATA_DIR / "memory"
HISTORY_DIR = MEMORY_DIR / "history"

# ── User files ────────────────────────────────────────────────────────────────
UPLOADS_DIR = DATA_DIR / "uploads"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
DOWNLOADS_DIR = DATA_DIR / "downloads"
ADDON_SKILLS_DIR = DATA_DIR / "skills"
