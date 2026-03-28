"""
heartbeat/jobs.py

Job definitions and persistence for the heartbeat scheduler.

Jobs are stored in ~/agent-files/config/jobs.json so the user can edit them
directly and the agent can add/remove them via built-in tools.

Each job:
    id:       Unique string identifier
    schedule: Cron expression ("0 8 * * *") or ISO datetime for one-time jobs
    prompt:   Message sent to the brain when the job fires
    enabled:  Whether the job is active (default: true)
"""

import json
import logging
from dataclasses import asdict, dataclass

from paths import JOBS_FILE

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """A scheduled task that the heartbeat runs autonomously."""

    id: str
    schedule: str  # Cron expression or ISO datetime string
    prompt: str  # Forwarded to brain.think() when the job fires
    enabled: bool = True


def load_jobs() -> list[Job]:
    """Load all jobs from config/jobs.json. Returns [] if file doesn't exist."""
    if not JOBS_FILE.exists():
        return []
    try:
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        return [Job(**j) for j in data]
    except Exception as e:
        logger.error(f"Failed to load jobs from {JOBS_FILE}: {e}")
        return []


def save_jobs(jobs: list[Job]) -> None:
    """Persist jobs to config/jobs.json."""
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(
        json.dumps([asdict(j) for j in jobs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_job(job: Job) -> None:
    """Add or replace a job (matched by id)."""
    jobs = load_jobs()
    jobs = [j for j in jobs if j.id != job.id]
    jobs.append(job)
    save_jobs(jobs)


def remove_job(job_id: str) -> bool:
    """Remove a job by id. Returns True if it was found and removed."""
    jobs = load_jobs()
    filtered = [j for j in jobs if j.id != job_id]
    if len(filtered) == len(jobs):
        return False
    save_jobs(filtered)
    return True
