"""
heartbeat — Proactive scheduling for Atlas.

Combines two independent components:

    Scheduler  — runs user-defined cron / one-time jobs (heartbeat/scheduler.py)
    Awareness  — periodic autonomous LLM check for proactive actions
                 (heartbeat/awareness.py)

The Heartbeat wrapper class exposes the same interface that gateway.py
and brain tools expect, so neither needs to change.
"""

from collections.abc import Callable, Awaitable

from heartbeat.scheduler import Scheduler
from heartbeat.awareness import Awareness


class Heartbeat:
    """
    Facade that owns both the Scheduler and the Awareness loop.

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
        self._scheduler = Scheduler(brain=brain, notify_callback=notify_callback)
        self._awareness = Awareness(brain=brain, notify_callback=notify_callback)

    async def start(self) -> None:
        await self._scheduler.start()
        await self._awareness.start()

    async def stop(self) -> None:
        await self._scheduler.stop()
        await self._awareness.stop()

    def reload_jobs(self) -> None:
        """Reload cron jobs after a brain scheduling tool call."""
        self._scheduler.reload_jobs()
