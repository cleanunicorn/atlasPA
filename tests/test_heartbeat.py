"""
tests/test_heartbeat.py

Tests for Phase 4: heartbeat scheduler, job persistence, and brain scheduling tools.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from providers.base import LLMResponse, ToolCall
from tests.test_brain import make_brain


# ── Job CRUD ──────────────────────────────────────────────────────────────────


def test_upsert_and_load_jobs(tmp_path):
    """Jobs are saved to and loaded from jobs.json."""
    from heartbeat.jobs import Job, upsert_job, load_jobs

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"):
        upsert_job(Job(id="briefing", schedule="0 8 * * *", prompt="Morning!"))
        jobs = load_jobs()

    assert len(jobs) == 1
    assert jobs[0].id == "briefing"
    assert jobs[0].schedule == "0 8 * * *"
    assert jobs[0].enabled is True


def test_upsert_replaces_existing_job(tmp_path):
    """Upserting with the same id replaces the old entry."""
    from heartbeat.jobs import Job, upsert_job, load_jobs

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"):
        upsert_job(Job(id="j1", schedule="0 8 * * *", prompt="Old prompt"))
        upsert_job(Job(id="j1", schedule="0 9 * * *", prompt="New prompt"))
        jobs = load_jobs()

    assert len(jobs) == 1
    assert jobs[0].prompt == "New prompt"
    assert jobs[0].schedule == "0 9 * * *"


def test_remove_job(tmp_path):
    """remove_job deletes the matching entry and returns True."""
    from heartbeat.jobs import Job, upsert_job, remove_job, load_jobs

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"):
        upsert_job(Job(id="j1", schedule="0 8 * * *", prompt="Hello"))
        upsert_job(Job(id="j2", schedule="0 9 * * *", prompt="World"))
        removed = remove_job("j1")
        jobs = load_jobs()

    assert removed is True
    assert len(jobs) == 1
    assert jobs[0].id == "j2"


def test_remove_nonexistent_job(tmp_path):
    """remove_job returns False if the job id doesn't exist."""
    from heartbeat.jobs import remove_job

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"):
        result = remove_job("ghost")

    assert result is False


# ── Scheduler ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scheduler_loads_enabled_jobs(tmp_path):
    """Scheduler registers only enabled jobs at startup."""
    from heartbeat.jobs import Job
    from heartbeat.scheduler import Scheduler

    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(
        json.dumps(
            [
                {
                    "id": "active",
                    "schedule": "0 8 * * *",
                    "prompt": "Hi",
                    "enabled": True,
                },
                {
                    "id": "inactive",
                    "schedule": "0 9 * * *",
                    "prompt": "Bye",
                    "enabled": False,
                },
            ]
        )
    )

    mock_brain = MagicMock()
    sched = Scheduler(brain=mock_brain, notify_callback=None)

    with patch("heartbeat.jobs.JOBS_FILE", jobs_file):
        with patch("heartbeat.scheduler.load_jobs") as mock_load:
            mock_load.return_value = [
                Job(id="active", schedule="0 8 * * *", prompt="Hi", enabled=True),
                Job(id="inactive", schedule="0 9 * * *", prompt="Bye", enabled=False),
            ]
            await sched.start()

    registered = [j.id for j in sched._scheduler.get_jobs()]
    assert "active" in registered
    assert "inactive" not in registered

    await sched.stop()


@pytest.mark.asyncio
async def test_scheduler_run_job_calls_notify(tmp_memory, empty_skills):
    """When a job fires, brain.think() is called and notify_callback receives the response."""
    from heartbeat.scheduler import Scheduler as Heartbeat

    notified = []

    async def mock_notify(text, files):
        notified.append(text)

    brain, _ = make_brain(
        [LLMResponse(content="Good morning!", tool_calls=[])],
        tmp_memory,
        empty_skills,
    )

    hb = Heartbeat(brain=brain, notify_callback=mock_notify)
    await hb._run_job("test_job", "Morning briefing prompt")

    assert len(notified) == 1
    assert "Good morning!" in notified[0]


@pytest.mark.asyncio
async def test_scheduler_job_failure_doesnt_crash(tmp_memory, empty_skills):
    """A job that raises an exception doesn't crash the scheduler."""
    from heartbeat.scheduler import Scheduler as Heartbeat

    brain, _ = make_brain([], tmp_memory, empty_skills)
    # Override think to raise
    brain.think = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    hb = Heartbeat(brain=brain, notify_callback=AsyncMock())
    # Should not raise
    await hb._run_job("failing_job", "prompt")


# ── Brain scheduling tools ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_schedule_job_tool_creates_job(tmp_path, tmp_memory, empty_skills):
    """schedule_job tool writes a new job to jobs.json and reloads the scheduler."""
    from heartbeat.jobs import load_jobs

    mock_heartbeat = MagicMock()
    mock_heartbeat.reload_jobs = MagicMock()

    brain, provider = make_brain(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="schedule_job",
                        arguments={
                            "job_id": "daily_check",
                            "schedule": "0 9 * * 1-5",
                            "prompt": "Daily standup reminder",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Job scheduled!", tool_calls=[]),
        ],
        tmp_memory,
        empty_skills,
    )
    brain.heartbeat = mock_heartbeat

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"):
        response, _ = await brain.think(
            "Schedule a daily reminder", conversation_history=[]
        )
        jobs = load_jobs()

    assert any(j.id == "daily_check" for j in jobs)
    mock_heartbeat.reload_jobs.assert_called_once()
    assert "Job scheduled!" in response


@pytest.mark.asyncio
async def test_list_jobs_tool(tmp_path, tmp_memory, empty_skills):
    """list_jobs tool returns a formatted list of all jobs."""
    from heartbeat.jobs import Job, upsert_job

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"):
        upsert_job(Job(id="briefing", schedule="0 8 * * *", prompt="Morning!"))

        brain, _ = make_brain(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="tc2", name="list_jobs", arguments={})],
                    stop_reason="tool_use",
                ),
                LLMResponse(content="Here are your jobs.", tool_calls=[]),
            ],
            tmp_memory,
            empty_skills,
        )

        _, history = await brain.think(
            "Show my scheduled jobs", conversation_history=[]
        )

    tool_result = next(m for m in history if m.role == "tool")
    assert "briefing" in tool_result.content
    assert "0 8 * * *" in tool_result.content


@pytest.mark.asyncio
async def test_delete_job_tool(tmp_path, tmp_memory, empty_skills):
    """delete_job tool removes the job from jobs.json."""
    from heartbeat.jobs import Job, upsert_job, load_jobs

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"):
        upsert_job(Job(id="to_delete", schedule="0 8 * * *", prompt="Bye"))

        brain, _ = make_brain(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc3",
                            name="delete_job",
                            arguments={"job_id": "to_delete"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                LLMResponse(content="Job deleted.", tool_calls=[]),
            ],
            tmp_memory,
            empty_skills,
        )

        await brain.think("Delete the to_delete job", conversation_history=[])
        remaining = load_jobs()

    assert not any(j.id == "to_delete" for j in remaining)


# ── Awareness ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_awareness_no_action_does_not_notify(tmp_memory, empty_skills, tmp_path):
    """When LLM returns NO_ACTION, notify_callback is never called."""
    from heartbeat.awareness import Awareness

    notified = []

    async def mock_notify(text, files):
        notified.append(text)

    brain, _ = make_brain(
        [LLMResponse(content="NO_ACTION", tool_calls=[])],
        tmp_memory,
        empty_skills,
    )

    with patch("heartbeat.awareness.LOG_FILE", tmp_path / "awareness_log.json"):
        awareness = Awareness(brain=brain, notify_callback=mock_notify)
        await awareness._check()

    assert len(notified) == 0


@pytest.mark.asyncio
async def test_awareness_triggered_calls_notify(tmp_memory, empty_skills, tmp_path):
    """When LLM returns a real message, notify_callback is called."""
    from heartbeat.awareness import Awareness

    notified = []

    async def mock_notify(text, files):
        notified.append(text)

    brain, _ = make_brain(
        [LLMResponse(content="Don't forget your meeting at 3pm!", tool_calls=[])],
        tmp_memory,
        empty_skills,
    )

    with patch("heartbeat.awareness.LOG_FILE", tmp_path / "awareness_log.json"):
        awareness = Awareness(brain=brain, notify_callback=mock_notify)
        await awareness._check()

    assert len(notified) == 1
    assert "meeting" in notified[0]


@pytest.mark.asyncio
async def test_awareness_logs_entries(tmp_memory, empty_skills, tmp_path):
    """_check() appends an entry to the log regardless of whether it triggered."""
    import json
    from heartbeat.awareness import Awareness

    log_file = tmp_path / "awareness_log.json"
    brain, _ = make_brain(
        [LLMResponse(content="NO_ACTION", tool_calls=[])],
        tmp_memory,
        empty_skills,
    )

    with patch("heartbeat.awareness.LOG_FILE", log_file):
        awareness = Awareness(brain=brain, notify_callback=None)
        await awareness._check()

    entries = json.loads(log_file.read_text())
    assert len(entries) == 1
    assert entries[0]["triggered"] is False


@pytest.mark.asyncio
async def test_awareness_failure_doesnt_crash(tmp_memory, empty_skills, tmp_path):
    """An LLM exception during awareness check is swallowed (no crash)."""
    from heartbeat.awareness import Awareness

    brain, _ = make_brain([], tmp_memory, empty_skills)
    brain.think = AsyncMock(side_effect=RuntimeError("LLM down"))

    with patch("heartbeat.awareness.LOG_FILE", tmp_path / "awareness_log.json"):
        awareness = Awareness(brain=brain, notify_callback=AsyncMock())
        await awareness._check()  # Should not raise


@pytest.mark.asyncio
async def test_heartbeat_wrapper_starts_both_components(
    tmp_memory, empty_skills, tmp_path
):
    """Heartbeat facade starts both Scheduler and Awareness."""
    from heartbeat import Heartbeat

    brain, _ = make_brain([], tmp_memory, empty_skills)

    with patch("heartbeat.awareness.LOG_FILE", tmp_path / "awareness_log.json"):
        hb = Heartbeat(brain=brain, notify_callback=None)
        await hb.start()
        assert hb._scheduler._scheduler.running
        assert hb._awareness._scheduler.running
        await hb.stop()
