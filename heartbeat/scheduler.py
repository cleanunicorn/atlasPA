"""
heartbeat/scheduler.py

Proactive scheduler — Phase 4 stub.

Currently starts cleanly but runs no jobs. Phase 4 will add:
  - Morning briefing (daily digest)
  - Inbox monitoring
  - Custom agent-defined cron jobs
"""

import logging

logger = logging.getLogger(__name__)


class Heartbeat:
    """
    Background scheduler for proactive agent tasks.

    Phase 4 will replace this stub with an apscheduler-based implementation
    that runs async jobs and pushes results to active channels.
    """

    def __init__(self, brain):
        """
        Args:
            brain: The Brain instance (used by jobs to generate responses).
        """
        self.brain = brain

    async def start(self) -> None:
        """Start the heartbeat scheduler. Currently a no-op stub."""
        logger.info("Heartbeat scheduler started (no active jobs — Phase 4 stub)")

    async def stop(self) -> None:
        """Stop the heartbeat scheduler."""
        logger.info("Heartbeat scheduler stopped")
