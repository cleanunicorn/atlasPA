"""
tests/test_maintenance.py

Tests for the daily maintenance module (heartbeat/maintenance.py).
Mocks DSPy calls for context/awareness consolidation.
"""

import hashlib
import json
import types
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch



# ── DSPy mock helpers ────────────────────────────────────────────────────────


def _make_mock_cot(consolidated_json: str):
    """Return a mock dspy.ChainOfThought that returns a fixed consolidation."""

    class MockCOT:
        def __init__(self, sig):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(consolidated_json=consolidated_json)

    return MockCOT


def _make_mock_predict(keep_indices_json: str):
    """Return a mock dspy.Predict that returns fixed keep indices."""

    class MockPredict:
        def __init__(self, sig):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(keep_indices_json=keep_indices_json)

    return MockPredict


# ── Expired job cleanup ──────────────────────────────────────────────────────


def test_cleanup_expired_onetime_jobs(tmp_path):
    """One-time jobs with a past datetime are removed."""
    from heartbeat.maintenance import _cleanup_expired_jobs

    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()

    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(
        json.dumps(
            [
                {"id": "expired", "schedule": past, "prompt": "old", "enabled": True},
                {
                    "id": "upcoming",
                    "schedule": future,
                    "prompt": "new",
                    "enabled": True,
                },
                {
                    "id": "cron",
                    "schedule": "0 8 * * *",
                    "prompt": "daily",
                    "enabled": True,
                },
            ]
        )
    )

    with patch("heartbeat.jobs.JOBS_FILE", jobs_file):
        expired = _cleanup_expired_jobs()

    assert expired == ["expired"]

    remaining = json.loads(jobs_file.read_text())
    ids = [j["id"] for j in remaining]
    assert "expired" not in ids
    assert "upcoming" in ids
    assert "cron" in ids


def test_cleanup_no_expired_jobs(tmp_path):
    """No jobs are removed when none are expired."""
    from heartbeat.maintenance import _cleanup_expired_jobs

    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(
        json.dumps(
            [{"id": "future", "schedule": future, "prompt": "hi", "enabled": True}]
        )
    )

    with patch("heartbeat.jobs.JOBS_FILE", jobs_file):
        expired = _cleanup_expired_jobs()

    assert expired == []


def test_cleanup_empty_jobs_file(tmp_path):
    """Handles missing jobs file gracefully."""
    from heartbeat.maintenance import _cleanup_expired_jobs

    with patch("heartbeat.jobs.JOBS_FILE", tmp_path / "nonexistent.json"):
        expired = _cleanup_expired_jobs()

    assert expired == []


# ── Context consolidation (DSPy) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_context_merges_duplicates(tmp_memory):
    """DSPy consolidation merges redundant entries."""
    from heartbeat.maintenance import _consolidate_context

    # Add redundant entries (mirrors real context.md pattern)
    tmp_memory.append_context("User works at Acme Corp as an engineer.")
    tmp_memory.append_context("User works at Acme Corp as an engineer. Likes Python.")
    tmp_memory.append_context("User likes hiking on weekends.")
    tmp_memory.append_context("User has a dog named Max.")
    tmp_memory.append_context("User prefers morning meetings.")

    entries_before = tmp_memory.parse_context_entries()
    assert len(entries_before) == 5

    # Mock DSPy to return merged output
    consolidated = json.dumps(
        [
            {
                "timestamp": "2026-03-28 12:00",
                "content": "User works at Acme Corp as an engineer. Likes Python.",
            },
            {
                "timestamp": "2026-03-28 12:00",
                "content": "User likes hiking on weekends.",
            },
            {"timestamp": "2026-03-28 12:00", "content": "User has a dog named Max."},
            {
                "timestamp": "2026-03-28 12:00",
                "content": "User prefers morning meetings.",
            },
        ]
    )

    with patch(
        "heartbeat.maintenance.dspy.ChainOfThought",
        _make_mock_cot(consolidated),
    ):
        before, after = await _consolidate_context(tmp_memory)

    assert before == 5
    assert after == 4  # merged the two Acme entries

    entries_after = tmp_memory.parse_context_entries()
    assert len(entries_after) == 4
    assert any(
        "Acme Corp" in e.content and "Python" in e.content for e in entries_after
    )


@pytest.mark.asyncio
async def test_consolidate_context_preserves_background(tmp_memory):
    """Background block is preserved through consolidation."""
    from heartbeat.maintenance import _consolidate_context

    # Write a context with a background block
    tmp_memory.replace_context_entries(
        "User is a senior engineer with 10 years experience.",
        [
            __import__("memory.retriever", fromlist=["ContextEntry"]).ContextEntry(
                timestamp="2026-03-28 12:00", content="Entry A."
            ),
            __import__("memory.retriever", fromlist=["ContextEntry"]).ContextEntry(
                timestamp="2026-03-28 12:01", content="Entry B."
            ),
            __import__("memory.retriever", fromlist=["ContextEntry"]).ContextEntry(
                timestamp="2026-03-28 12:02", content="Entry C."
            ),
            __import__("memory.retriever", fromlist=["ContextEntry"]).ContextEntry(
                timestamp="2026-03-28 12:03", content="Entry D."
            ),
            __import__("memory.retriever", fromlist=["ContextEntry"]).ContextEntry(
                timestamp="2026-03-28 12:04", content="Entry E."
            ),
        ],
    )

    consolidated = json.dumps(
        [
            {"timestamp": "2026-03-28 12:04", "content": "Merged A-E content."},
        ]
    )

    with patch(
        "heartbeat.maintenance.dspy.ChainOfThought",
        _make_mock_cot(consolidated),
    ):
        before, after = await _consolidate_context(tmp_memory)

    assert before == 5
    assert after == 1

    entries_after = tmp_memory.parse_context_entries()
    # Background block + 1 consolidated entry
    background = [e for e in entries_after if not e.timestamp]
    assert len(background) == 1
    assert "senior engineer" in background[0].content


@pytest.mark.asyncio
async def test_consolidate_context_skips_below_threshold(tmp_memory):
    """No consolidation when entry count is below threshold."""
    from heartbeat.maintenance import _consolidate_context

    tmp_memory.append_context("Single entry.")
    before, after = await _consolidate_context(tmp_memory)

    assert before == 0
    assert after == 0


@pytest.mark.asyncio
async def test_consolidate_context_handles_fenced_json(tmp_memory):
    """Handles LLM wrapping JSON in markdown code fences."""
    from heartbeat.maintenance import _consolidate_context

    for i in range(6):
        tmp_memory.append_context(f"Fact {i}.")

    fenced = '```json\n[{"timestamp": "2026-03-28 12:00", "content": "All facts merged."}]\n```'

    with patch(
        "heartbeat.maintenance.dspy.ChainOfThought",
        _make_mock_cot(fenced),
    ):
        before, after = await _consolidate_context(tmp_memory)

    assert before == 6
    assert after == 1


@pytest.mark.asyncio
async def test_consolidate_context_aborts_on_empty_result(tmp_memory):
    """Consolidation aborts if LLM returns empty list."""
    from heartbeat.maintenance import _consolidate_context

    for i in range(6):
        tmp_memory.append_context(f"Entry {i}.")

    entries_before = tmp_memory.parse_context_entries()

    with (
        patch("heartbeat.maintenance.dspy.ChainOfThought", _make_mock_cot("[]")),
        pytest.raises(ValueError, match="invalid consolidation"),
    ):
        await _consolidate_context(tmp_memory)

    # Memory should be unchanged
    entries_after = tmp_memory.parse_context_entries()
    assert len(entries_after) == len(entries_before)


@pytest.mark.asyncio
async def test_consolidate_context_none_from_dspy(tmp_memory):
    """Consolidation raises ValueError (not AttributeError) when DSPy returns None."""
    from heartbeat.maintenance import _consolidate_context

    for i in range(6):
        tmp_memory.append_context(f"Entry {i}.")

    entries_before = tmp_memory.parse_context_entries()

    # Simulate DSPy returning None for the output field (LLM server down, etc.)
    with (
        patch(
            "heartbeat.maintenance.dspy.ChainOfThought",
            _make_mock_cot(None),
        ),
        pytest.raises(ValueError, match="empty/None"),
    ):
        await _consolidate_context(tmp_memory)

    # Memory must be unchanged
    entries_after = tmp_memory.parse_context_entries()
    assert len(entries_after) == len(entries_before)


@pytest.mark.asyncio
async def test_consolidate_awareness_none_from_dspy(tmp_path):
    """Awareness consolidation raises ValueError (not AttributeError) when DSPy returns None."""
    from heartbeat.maintenance import _consolidate_awareness

    log_file = tmp_path / "awareness_log.json"
    entries = []
    for i in range(12):
        ts = (datetime.now(timezone.utc) - timedelta(hours=12 - i)).isoformat()
        entries.append({"ts": ts, "triggered": False, "summary": "no action"})
    log_file.write_text(json.dumps(entries))

    with (
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", log_file),
        patch("heartbeat.maintenance.dspy.Predict", _make_mock_predict(None)),
        pytest.raises(ValueError, match="empty/None"),
    ):
        await _consolidate_awareness()

    # Log must be unchanged
    remaining = json.loads(log_file.read_text())
    assert len(remaining) == 12


# ── Awareness consolidation (DSPy) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_awareness_keeps_triggered(tmp_path):
    """DSPy keeps triggered entries and prunes stale no-action ones."""
    from heartbeat.maintenance import _consolidate_awareness

    log_file = tmp_path / "awareness_log.json"
    entries = []
    for i in range(12):
        ts = (datetime.now(timezone.utc) - timedelta(hours=12 - i)).isoformat()
        entries.append(
            {
                "ts": ts,
                "triggered": (i == 5),  # only entry 5 triggered
                "summary": "sent reminder" if i == 5 else "no action",
            }
        )
    log_file.write_text(json.dumps(entries))

    # LLM decides to keep entry 5 (triggered) and entries 10, 11 (recent)
    with (
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", log_file),
        patch(
            "heartbeat.maintenance.dspy.Predict",
            _make_mock_predict("[5, 10, 11]"),
        ),
    ):
        before, after = await _consolidate_awareness()

    assert before == 12
    assert after == 3

    kept = json.loads(log_file.read_text())
    assert len(kept) == 3
    assert any(e["triggered"] for e in kept)


@pytest.mark.asyncio
async def test_consolidate_awareness_skips_below_threshold(tmp_path):
    """No consolidation when log is below threshold."""
    from heartbeat.maintenance import _consolidate_awareness

    log_file = tmp_path / "awareness_log.json"
    log_file.write_text(
        json.dumps(
            [
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "triggered": False,
                    "summary": "no action",
                }
            ]
        )
    )

    with patch("heartbeat.maintenance.AWARENESS_LOG_FILE", log_file):
        before, after = await _consolidate_awareness()

    assert before == 0
    assert after == 0


@pytest.mark.asyncio
async def test_consolidate_awareness_missing_file(tmp_path):
    """Handles missing log file gracefully."""
    from heartbeat.maintenance import _consolidate_awareness

    with patch("heartbeat.maintenance.AWARENESS_LOG_FILE", tmp_path / "nope.json"):
        before, after = await _consolidate_awareness()

    assert before == 0
    assert after == 0


@pytest.mark.asyncio
async def test_consolidate_awareness_safety_keeps_last(tmp_path):
    """If LLM returns empty keep list, at least the most recent entry is kept."""
    from heartbeat.maintenance import _consolidate_awareness

    log_file = tmp_path / "awareness_log.json"
    entries = []
    for i in range(12):
        ts = (datetime.now(timezone.utc) - timedelta(hours=12 - i)).isoformat()
        entries.append({"ts": ts, "triggered": False, "summary": "no action"})
    log_file.write_text(json.dumps(entries))

    with (
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", log_file),
        patch("heartbeat.maintenance.dspy.Predict", _make_mock_predict("[]")),
    ):
        before, after = await _consolidate_awareness()

    assert before == 12
    assert after == 1  # safety net: kept the last entry


@pytest.mark.asyncio
async def test_consolidate_awareness_invalid_indices_ignored(tmp_path):
    """Out-of-range indices from LLM are silently filtered."""
    from heartbeat.maintenance import _consolidate_awareness

    log_file = tmp_path / "awareness_log.json"
    entries = []
    for i in range(12):
        ts = (datetime.now(timezone.utc) - timedelta(hours=12 - i)).isoformat()
        entries.append({"ts": ts, "triggered": False, "summary": "no action"})
    log_file.write_text(json.dumps(entries))

    # Indices 99 and -1 are invalid; only 0 and 11 are valid
    with (
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", log_file),
        patch(
            "heartbeat.maintenance.dspy.Predict",
            _make_mock_predict("[0, 11, 99, -1]"),
        ),
    ):
        before, after = await _consolidate_awareness()

    assert before == 12
    assert after == 2  # only 0 and 11 are in range(12)


# ── Embedding cache pruning ──────────────────────────────────────────────────


def test_prune_stale_embeddings(tmp_path, tmp_memory):
    """Stale embedding entries (for deleted context) are removed."""
    from heartbeat.maintenance import _prune_embedding_cache

    tmp_memory.append_context("User likes hiking.")
    entries = tmp_memory.parse_context_entries()
    real_hash = hashlib.sha256(entries[0].content.encode("utf-8")).hexdigest()

    cache_file = tmp_path / "embeddings.json"
    cache_data = {
        real_hash: [0.1, 0.2, 0.3],
        "deadbeef" * 8: [0.4, 0.5, 0.6],
    }
    cache_file.write_text(json.dumps(cache_data))

    with patch("heartbeat.maintenance.EMBEDDINGS_FILE", cache_file):
        pruned = _prune_embedding_cache(tmp_memory)

    assert pruned == 1
    remaining = json.loads(cache_file.read_text())
    assert real_hash in remaining
    assert "deadbeef" * 8 not in remaining


def test_prune_embeddings_nothing_stale(tmp_path, tmp_memory):
    """No pruning when all cache entries are current."""
    from heartbeat.maintenance import _prune_embedding_cache

    tmp_memory.append_context("Fresh entry.")
    entries = tmp_memory.parse_context_entries()
    h = hashlib.sha256(entries[0].content.encode("utf-8")).hexdigest()

    cache_file = tmp_path / "embeddings.json"
    cache_file.write_text(json.dumps({h: [0.1, 0.2]}))

    with patch("heartbeat.maintenance.EMBEDDINGS_FILE", cache_file):
        pruned = _prune_embedding_cache(tmp_memory)

    assert pruned == 0


def test_prune_embeddings_missing_file(tmp_path, tmp_memory):
    """Handles missing embeddings file gracefully."""
    from heartbeat.maintenance import _prune_embedding_cache

    with patch("heartbeat.maintenance.EMBEDDINGS_FILE", tmp_path / "nope.json"):
        pruned = _prune_embedding_cache(tmp_memory)

    assert pruned == 0


# ── Full run_maintenance ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_maintenance_full(tmp_path, tmp_memory):
    """run_maintenance orchestrates all cleanup tasks."""
    from heartbeat.maintenance import run_maintenance

    # Set up expired job
    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(
        json.dumps(
            [{"id": "old_job", "schedule": past, "prompt": "gone", "enabled": True}]
        )
    )

    # Add enough context entries to trigger consolidation
    for i in range(6):
        tmp_memory.append_context(f"Context entry {i}.")

    # Set up awareness log above threshold
    log_file = tmp_path / "awareness_log.json"
    awareness_entries = []
    for i in range(12):
        ts = (datetime.now(timezone.utc) - timedelta(hours=12 - i)).isoformat()
        awareness_entries.append({"ts": ts, "triggered": False, "summary": "no action"})
    log_file.write_text(json.dumps(awareness_entries))

    # Set up stale embedding
    cache_file = tmp_path / "embeddings.json"
    cache_file.write_text(json.dumps({"stale_hash": [0.1, 0.2]}))

    reload_called = []

    # Mock DSPy: context consolidated to 3, awareness keeps indices [10, 11]
    context_result = json.dumps(
        [
            {"timestamp": "2026-03-28 12:00", "content": "Merged context A."},
            {"timestamp": "2026-03-28 12:00", "content": "Merged context B."},
            {"timestamp": "2026-03-28 12:00", "content": "Merged context C."},
        ]
    )

    with (
        patch("heartbeat.jobs.JOBS_FILE", jobs_file),
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", log_file),
        patch("heartbeat.maintenance.EMBEDDINGS_FILE", cache_file),
        patch(
            "heartbeat.maintenance.dspy.ChainOfThought",
            _make_mock_cot(context_result),
        ),
        patch(
            "heartbeat.maintenance.dspy.Predict",
            _make_mock_predict("[10, 11]"),
        ),
    ):
        summary = await run_maintenance(
            memory=tmp_memory,
            on_jobs_changed=lambda: reload_called.append(True),
        )

    assert "expired job" in summary.lower()
    assert "context" in summary.lower()
    assert "awareness" in summary.lower()
    assert "embedding" in summary.lower()
    assert len(reload_called) == 1


@pytest.mark.asyncio
async def test_run_maintenance_nothing_to_do(tmp_path, tmp_memory):
    """When there's nothing to clean, returns 'Nothing to clean up'."""
    from heartbeat.maintenance import run_maintenance

    with (
        patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"),
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", tmp_path / "no_log.json"),
        patch("heartbeat.maintenance.EMBEDDINGS_FILE", tmp_path / "no_cache.json"),
    ):
        summary = await run_maintenance(memory=tmp_memory)

    assert summary == "Nothing to clean up"


@pytest.mark.asyncio
async def test_run_maintenance_notifies_user(tmp_path, tmp_memory):
    """When there are results and a notify_callback, the user is informed."""
    from heartbeat.maintenance import run_maintenance

    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(
        json.dumps([{"id": "done", "schedule": past, "prompt": "x", "enabled": True}])
    )

    notified = []

    async def mock_notify(text, files):
        notified.append(text)

    with (
        patch("heartbeat.jobs.JOBS_FILE", jobs_file),
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", tmp_path / "no.json"),
        patch("heartbeat.maintenance.EMBEDDINGS_FILE", tmp_path / "no.json"),
    ):
        await run_maintenance(
            memory=tmp_memory,
            notify_callback=mock_notify,
        )

    assert len(notified) == 1
    assert "maintenance" in notified[0].lower()


@pytest.mark.asyncio
async def test_run_maintenance_dspy_failure_doesnt_crash(tmp_path, tmp_memory):
    """If DSPy consolidation fails, maintenance continues with other tasks."""
    from heartbeat.maintenance import run_maintenance

    # Add entries to trigger consolidation attempts
    for i in range(6):
        tmp_memory.append_context(f"Entry {i}.")

    log_file = tmp_path / "awareness_log.json"
    entries = []
    for i in range(12):
        ts = (datetime.now(timezone.utc) - timedelta(hours=12 - i)).isoformat()
        entries.append({"ts": ts, "triggered": False, "summary": "no action"})
    log_file.write_text(json.dumps(entries))

    # Make DSPy calls raise
    def _failing_cot(sig):
        def _call(**kwargs):
            raise RuntimeError("LLM unavailable")

        cot = types.SimpleNamespace()
        cot.__call__ = _call
        return cot

    def _failing_predict(sig):
        def _call(**kwargs):
            raise RuntimeError("LLM unavailable")

        pred = types.SimpleNamespace()
        pred.__call__ = _call
        return pred

    with (
        patch("heartbeat.jobs.JOBS_FILE", tmp_path / "jobs.json"),
        patch("heartbeat.maintenance.AWARENESS_LOG_FILE", log_file),
        patch("heartbeat.maintenance.EMBEDDINGS_FILE", tmp_path / "no.json"),
        patch("heartbeat.maintenance.dspy.ChainOfThought", _failing_cot),
        patch("heartbeat.maintenance.dspy.Predict", _failing_predict),
    ):
        summary = await run_maintenance(memory=tmp_memory)

    # Should report failures but not crash
    assert "failed" in summary.lower()


# ── Heartbeat integration ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_schedules_maintenance(tmp_memory, tmp_path):
    """Heartbeat facade starts the maintenance scheduler alongside others."""
    from heartbeat import Heartbeat
    from tests.test_brain import make_brain

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    with (
        patch("skills.registry.CORE_SKILLS_DIR", skills_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", tmp_path / "addon_skills"),
    ):
        from skills.registry import SkillRegistry

        skills = SkillRegistry()

    brain, _ = make_brain([], tmp_memory, skills)

    with patch("heartbeat.awareness.LOG_FILE", tmp_path / "awareness_log.json"):
        hb = Heartbeat(brain=brain, notify_callback=None)
        await hb.start()

        assert hb._maintenance_scheduler.running
        maint_jobs = hb._maintenance_scheduler.get_jobs()
        assert any(j.id == "daily_maintenance" for j in maint_jobs)

        await hb.stop()
