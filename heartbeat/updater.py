"""
heartbeat/updater.py

Periodic GitHub update checker for Atlas.

How it works:
    1. Every UPDATE_CHECK_INTERVAL_HOURS hours, run `git fetch` and compare
       the local HEAD with the remote tracking branch HEAD.
    2. If new commits are available, notify the user via notify_callback
       asking whether they'd like to update.
    3. Once an update notification has been sent, it won't repeat until
       even newer commits are detected on the remote.

Usage:
    updater = Updater(notify_callback=my_async_fn)
    await updater.start()
    ...
    await updater.stop()
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# How often to poll for updates (hours). Override with UPDATE_CHECK_INTERVAL_HOURS env var.
UPDATE_CHECK_INTERVAL_HOURS = int(os.getenv("UPDATE_CHECK_INTERVAL_HOURS", "1"))

# Project root — the git repository
_ROOT = Path(__file__).parent.parent.resolve()


def _git_env() -> dict:
    """
    Build a non-interactive env for git subprocesses.

    - GIT_TERMINAL_PROMPT=0 — never prompt for HTTPS credentials
    - GIT_SSH_COMMAND with BatchMode=yes — never prompt for ssh password / passphrase
    - StrictHostKeyChecking=accept-new — auto-accept new host keys but reject changed ones
    """
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault(
        "GIT_SSH_COMMAND",
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
    )
    return env


def _run_git(*args: str, timeout: int = 15) -> tuple[int, str]:
    """Run a git command synchronously and return (returncode, combined output)."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_ROOT,
            env=_git_env(),
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"git {' '.join(args)} timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "git not found on PATH"
    except Exception as e:
        return -1, str(e)


def _fetch() -> tuple[bool, str]:
    """Run `git fetch`. Returns (ok, output)."""
    code, out = _run_git("fetch", "--quiet")
    return code == 0, out


def check_for_update() -> tuple[bool, str]:
    """
    Fetch from the remote and compare local HEAD with the upstream tracking branch.

    Returns:
        (update_available, message) — message describes what's new or why skipped.
    """
    ok, out = _fetch()
    if not ok:
        return False, f"git fetch failed: {out}"

    # Local HEAD
    code, local_head = _run_git("rev-parse", "HEAD")
    if code != 0:
        return False, f"Could not read local HEAD: {local_head}"

    # Remote tracking branch HEAD (@{u} = upstream)
    code, remote_head = _run_git("rev-parse", "@{u}")
    if code != 0:
        return False, "No upstream branch configured — skipping update check"

    local_head = local_head.strip()
    remote_head = remote_head.strip()

    if local_head == remote_head:
        return False, "Already up to date"

    # Count commits ahead on the remote
    code, log_out = _run_git("log", "HEAD..@{u}", "--oneline", "--no-decorate")
    commit_lines = [ln for ln in log_out.splitlines() if ln.strip()]
    n = len(commit_lines)

    summary = f"{n} new commit(s)" if n else "new commits"
    if commit_lines:
        recent = commit_lines[:3]
        subjects = "\n".join(f"  • {c}" for c in recent)
        if n > 3:
            subjects += f"\n  … and {n - 3} more"
        summary = f"{n} new commit(s):\n{subjects}"

    return True, summary


class Updater:
    """
    Periodic update checker — fetches from GitHub and notifies the user when
    a new version is available.
    """

    def __init__(self, notify_callback=None):
        """
        Args:
            notify_callback:  async def notify(text: str, files: list) -> None
        """
        self.notify_callback = notify_callback
        self._scheduler = AsyncIOScheduler()
        self._last_reported_remote: str | None = None
        self._last_reported_fetch_error: str | None = None

    async def start(self) -> None:
        self._scheduler.add_job(
            self._check,
            trigger=IntervalTrigger(hours=UPDATE_CHECK_INTERVAL_HOURS),
            id="update_check",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            f"Update checker started — polling every {UPDATE_CHECK_INTERVAL_HOURS}h"
        )

    async def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("Update checker stopped")

    async def check_now(self) -> tuple[bool, str]:
        """Run an update check immediately (off-thread so it won't block the event loop)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, check_for_update)

    async def _notify_fetch_error(self, fetch_out: str) -> None:
        """
        Surface a `git fetch` failure (auth/network) to logs and the user.

        Deduped by error text so a persistently broken remote does not spam
        the user every poll interval. The dedupe key resets once a fetch
        succeeds again.
        """
        logger.warning(f"Update check: git fetch failed — {fetch_out}")

        if fetch_out == self._last_reported_fetch_error:
            return
        self._last_reported_fetch_error = fetch_out

        if not self.notify_callback:
            return

        notification = (
            "⚠️ Atlas update check failed — `git fetch` error:\n\n"
            f"```\n{fetch_out}\n```\n\n"
            "Likely causes: SSH key missing in the service env, network "
            "outage, or a changed remote host key. The repo is public — "
            "switching the remote to HTTPS avoids ssh-agent dependency:\n"
            "`git remote set-url origin https://github.com/<owner>/<repo>.git`"
        )
        try:
            await self.notify_callback(notification, [])
        except Exception as e:
            logger.exception(f"Fetch-error notify failed: {e}")

    async def _check(self) -> None:
        """One scheduled update-check tick."""
        logger.debug("Update check tick")

        loop = asyncio.get_event_loop()

        # Fetch first so auth/network failures surface separately from
        # the no-update path (which is silent).
        ok, fetch_out = await loop.run_in_executor(None, _fetch)
        if not ok:
            await self._notify_fetch_error(fetch_out)
            return

        # Reset fetch-error dedupe once a fetch succeeds again.
        self._last_reported_fetch_error = None

        update_available, message = await loop.run_in_executor(None, check_for_update)

        if not update_available:
            logger.debug(f"Update check: {message}")
            return

        # Read the remote HEAD to avoid duplicate notifications
        code, remote_head = await loop.run_in_executor(
            None, lambda: _run_git("rev-parse", "@{u}")
        )
        remote_head = remote_head.strip()

        if remote_head == self._last_reported_remote:
            logger.debug("Update already reported for this remote HEAD — skipping")
            return

        self._last_reported_remote = remote_head
        logger.info(f"Update available — notifying user: {message[:80]}")

        notification = (
            f"🆕 Atlas update available!\n\n{message}\n\n"
            "Say 'update' or ask me to run `update_self` to apply it."
        )
        if self.notify_callback:
            try:
                await self.notify_callback(notification, [])
            except Exception as e:
                logger.exception(f"Update notify failed: {e}")
        else:
            logger.info(f"Update available (no notify_callback): {message}")
