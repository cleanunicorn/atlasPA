"""
heartbeat/scheduler.py

Cron scheduler — runs user-defined background jobs on a schedule.

Each job sends a synthetic prompt to the Brain and pushes the response
to all active channels via notify_callback.

Job configuration lives in config/jobs.json so both the user and the
agent (via schedule_job tool) can add/edit/remove jobs at runtime.

Usage:
    scheduler = Scheduler(brain=brain, notify_callback=my_async_fn)
    await scheduler.start()
    ...
    await scheduler.stop()

notify_callback signature:
    async def notify(text: str, files: list[tuple[Path, str]]) -> None
"""

import logging
from collections.abc import Callable, Awaitable
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from heartbeat.jobs import Job, load_jobs

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Cron job scheduler powered by APScheduler.

    Loads jobs from config/jobs.json at startup.
    Call reload_jobs() to pick up changes made at runtime (e.g. via brain tools).
    """

    def __init__(
        self,
        brain,
        notify_callback: Callable[[str, list], Awaitable[None]] | None = None,
    ):
        """
        Args:
            brain:            The Brain instance — jobs call brain.think().
            notify_callback:  Async function that delivers the response to users.
                              Signature: async def notify(text, files) -> None
                              If None, responses are only logged (useful for testing).
        """
        self.brain = brain
        self.notify_callback = notify_callback
        self._scheduler = AsyncIOScheduler()

    async def start(self) -> None:
        """Load jobs and start the scheduler."""
        self._load_jobs()
        self._scheduler.start()
        active = len(self._scheduler.get_jobs())
        logger.info(f"Heartbeat started — {active} active job(s)")

    async def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("Heartbeat stopped")

    def reload_jobs(self) -> None:
        """
        Reload all jobs from config/jobs.json.
        Called after the brain adds/removes a job via scheduling tools.
        """
        self._scheduler.remove_all_jobs()
        self._load_jobs()
        active = len(self._scheduler.get_jobs())
        logger.info(f"Heartbeat reloaded — {active} active job(s)")

    def _load_jobs(self) -> None:
        """Read config/jobs.json and register all enabled jobs."""
        jobs = load_jobs()
        for job in jobs:
            if job.enabled:
                self._register(job)

    def _register(self, job: Job) -> None:
        """Register a single job with APScheduler."""
        trigger = self._make_trigger(job.schedule)
        if trigger is None:
            logger.warning(f"Job '{job.id}': invalid schedule '{job.schedule}' — skipped")
            return
        self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            id=job.id,
            args=[job.id, job.prompt],
            replace_existing=True,
        )
        logger.info(f"Scheduled job '{job.id}': {job.schedule}")

    @staticmethod
    def _make_trigger(schedule: str):
        """Parse a cron expression or ISO datetime into an APScheduler trigger."""
        # Try ISO datetime first (one-time jobs)
        try:
            dt = datetime.fromisoformat(schedule)
            return DateTrigger(run_date=dt)
        except ValueError:
            pass

        # Cron expression: "minute hour day month day_of_week"
        parts = schedule.strip().split()
        if len(parts) != 5:
            return None
        try:
            return CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )
        except Exception:
            return None

    async def _run_job(self, job_id: str, prompt: str) -> None:
        """Execute a job: call brain.think() and push the response."""
        logger.info(f"Running job '{job_id}'")
        try:
            response, _ = await self.brain.think(prompt, conversation_history=[])
            files = self.brain.take_files()

            if self.notify_callback:
                await self.notify_callback(response, files)
            else:
                logger.info(f"Job '{job_id}' response (no notify_callback): {response[:200]}")

        except Exception as e:
            logger.exception(f"Job '{job_id}' failed: {e}")
