"""
heartbeat — Proactive scheduling for Atlas.

Combines three components:

    Scheduler    — runs user-defined cron / one-time jobs (heartbeat/scheduler.py)
    Awareness    — periodic autonomous LLM check for proactive actions
                   (heartbeat/awareness.py)
    Maintenance  — daily cleanup of expired jobs, context consolidation,
                   and stale data pruning (heartbeat/maintenance.py)

The Heartbeat wrapper class exposes the same interface that gateway.py
and brain tools expect, so neither needs to change.
"""

import logging
import os
from collections.abc import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from heartbeat.scheduler import Scheduler
from heartbeat.awareness import Awareness
from heartbeat.maintenance import run_maintenance

logger = logging.getLogger(__name__)

# Hour (0-23) at which daily maintenance runs.  Default: 3 AM.
MAINTENANCE_HOUR = int(os.getenv("MAINTENANCE_HOUR", "3"))


class Heartbeat:
    """
    Facade that owns the Scheduler, Awareness loop, and daily Maintenance.

    gateway.py usage:
        heartbeat = Heartbeat(brain=brain, notify_callback=telegram.push_message)
        brain.heartbeat = heartbeat
        await heartbeat.start()
        ...
        await heartbeat.stop()

    Brain tools call:
        heartbeat.reload_jobs()   — after adding/removing a cron job
    """

    def __init__(
        self,
        brain,
        notify_callback: Callable[[str, list], Awaitable[None]] | None = None,
    ):
        self._brain = brain
        self._notify_callback = notify_callback
        self._scheduler = Scheduler(brain=brain, notify_callback=notify_callback)
        self._awareness = Awareness(brain=brain, notify_callback=notify_callback)
        self._maintenance_scheduler = AsyncIOScheduler()

    async def start(self) -> None:
        await self._scheduler.start()
        await self._awareness.start()
        self._maintenance_scheduler.add_job(
            self._run_maintenance,
            trigger=CronTrigger(hour=MAINTENANCE_HOUR, minute=0),
            id="daily_maintenance",
            replace_existing=True,
        )
        self._maintenance_scheduler.start()
        logger.info(f"Daily maintenance scheduled at {MAINTENANCE_HOUR:02d}:00")

    async def stop(self) -> None:
        await self._scheduler.stop()
        await self._awareness.stop()
        if self._maintenance_scheduler.running:
            self._maintenance_scheduler.shutdown(wait=False)
        logger.info("Heartbeat stopped (scheduler + awareness + maintenance)")

    def reload_jobs(self) -> None:
        """Reload cron jobs after a brain scheduling tool call."""
        self._scheduler.reload_jobs()

    async def run_maintenance_now(self) -> str:
        """Run maintenance immediately (e.g. from a brain tool or test)."""
        return await self._run_maintenance()

    async def _run_maintenance(self) -> str:
        """Internal: execute daily maintenance tasks."""
        return await run_maintenance(
            memory=self._brain.memory,
            provider=self._brain.provider,
            on_jobs_changed=self._scheduler.reload_jobs,
            notify_callback=self._notify_callback,
        )
