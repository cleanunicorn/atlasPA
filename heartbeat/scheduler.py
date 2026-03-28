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


def _build_job_note(job_id: str, schedule: str) -> str:
    """Build a per-job system-prompt suffix with context and behavioural rules."""
    return (
        f"You are running as a **scheduled background task** "
        f"(job: `{job_id}`, schedule: `{schedule}`), not in a live conversation.\n\n"
        "**Rules for this run:**\n"
        "1. **Always fetch fresh data.** Memory entries may be stale — always call "
        "the relevant skill (calendar, web search, etc.) to get current information. "
        "Do not skip a tool call just because similar data appears in memory.\n"
        "2. **Do not ask clarifying questions.** Nobody is present to answer; make "
        "reasonable assumptions and proceed.\n"
        "3. **Write a self-contained message.** Your response is pushed as a "
        "notification — there is no follow-up turn. Include everything the user needs.\n"
        "4. **Update stale memory.** After fetching fresh data, remove outdated "
        "entries with `forget` and store the updated facts with `remember`.\n"
        "5. **Stay silent if there is nothing to report.** Return an empty string "
        "rather than padding with filler."
    )


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

    def trigger_job(self, job_id: str) -> bool:
        """
        Fire a job immediately, outside its normal schedule.
        Returns False if job_id is not found in config/jobs.json.
        """
        jobs = {j.id: j for j in load_jobs()}
        if job_id not in jobs:
            return False
        job = jobs[job_id]
        import uuid
        from datetime import timezone as _tz

        self._scheduler.add_job(
            self._run_job,
            trigger=DateTrigger(run_date=datetime.now(_tz.utc)),
            id=f"{job_id}__manual__{uuid.uuid4().hex[:6]}",
            args=[job.id, job.prompt, job.schedule],
            replace_existing=False,
        )
        logger.info(f"Manually triggered job '{job_id}'")
        return True

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
            logger.warning(
                f"Job '{job.id}': invalid schedule '{job.schedule}' — skipped"
            )
            return
        self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            id=job.id,
            args=[job.id, job.prompt, job.schedule],
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

    async def _run_job(self, job_id: str, prompt: str, schedule: str = "") -> None:
        """Execute a job: call brain.think() and push the response."""
        logger.info(f"Running job '{job_id}' (schedule: {schedule!r})")
        try:
            response, _ = await self.brain.think(
                prompt,
                conversation_history=[],
                system_suffix=_build_job_note(job_id, schedule),
            )
            files = self.brain.take_files()

            if not response.strip():
                logger.info(f"Job '{job_id}' had nothing to report — skipping notify")
                return

            if self.notify_callback:
                await self.notify_callback(response, files)
            else:
                logger.info(
                    f"Job '{job_id}' response (no notify_callback): {response[:200]}"
                )

        except Exception as e:
            logger.exception(f"Job '{job_id}' failed: {e}")
