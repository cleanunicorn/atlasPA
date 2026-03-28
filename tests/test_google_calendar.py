"""
tests/test_google_calendar.py

Tests for the google_calendar skill (multi-account).
All Google API calls are mocked — no real credentials needed.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── Mock helpers ───────────────────────────────────────────────────────────────


def make_service(events=None):
    """Build a minimal mock Google Calendar service for one account."""
    svc = MagicMock()
    svc.calendarList().list().execute.return_value = {
        "items": [
            {"id": "primary", "summary": "Main", "primary": True},
            {"id": "secondary@group", "summary": "Secondary"},
        ]
    }
    svc.events().list().execute.return_value = {"items": events or []}
    svc.events().insert().execute.return_value = {
        "id": "new_id",
        "summary": "Test Event",
        "start": {"dateTime": "2026-03-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T10:00:00+00:00"},
    }
    svc.events().get().execute.return_value = {
        "id": "evt1",
        "summary": "Old Title",
        "start": {"dateTime": "2026-03-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T10:00:00+00:00"},
    }
    svc.events().update().execute.return_value = {
        "id": "evt1",
        "summary": "New Title",
        "start": {"dateTime": "2026-03-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T10:00:00+00:00"},
    }
    return svc


def _fake_configs(names=("personal", "work")):
    """Return fake account config dicts (creds/token files don't need to exist for patching)."""
    return [
        {
            "name": n,
            "creds_file": Path(f"/fake/config/google_credentials_{n}.json"),
            "token_file": Path(f"/fake/config/google_token_{n}.json"),
        }
        for n in names
    ]


def patch_services(service_map: dict):
    """
    Patch _get_services to return [(name, service), ...] based on service_map.
    service_map: {"personal": mock_svc, "work": mock_svc2}
    """

    def fake_get_services(account_name):
        if account_name == "all":
            return list(service_map.items())
        if account_name in service_map:
            return [(account_name, service_map[account_name])]
        raise ValueError(f"Unknown account '{account_name}'")

    return patch(
        "skills.google_calendar.tool._get_services", side_effect=fake_get_services
    )


# ── list_accounts ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_accounts(tmp_path):
    from skills.google_calendar.tool import run

    configs = [
        {
            "name": "personal",
            "creds_file": tmp_path / "creds_personal.json",
            "token_file": tmp_path / "tok.json",
        },
        {
            "name": "work",
            "creds_file": tmp_path / "creds_work.json",
            "token_file": tmp_path / "tok2.json",
        },
    ]
    (tmp_path / "creds_personal.json").write_text("{}")  # exists
    # creds_work.json does NOT exist → shows warning

    with patch(
        "skills.google_calendar.tool._load_account_configs", return_value=configs
    ):
        result = await run(action="list_accounts")

    assert "personal" in result
    assert "work" in result
    assert "credentials found" in result
    assert "credentials missing" in result


# ── list_calendars ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_calendars_multi_account():
    from skills.google_calendar.tool import run

    svc1, svc2 = make_service(), make_service()
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="list_calendars")

    assert "personal" in result
    assert "work" in result
    assert "Main" in result


@pytest.mark.asyncio
async def test_list_calendars_single_account():
    from skills.google_calendar.tool import run

    svc = make_service()
    with patch_services({"personal": svc}):
        result = await run(action="list_calendars", account="personal")

    assert "Main" in result
    # Single account — no account header needed in output
    assert "primary" in result or "Main" in result


# ── list_events ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_events_no_events():
    from skills.google_calendar.tool import run

    svc = make_service(events=[])
    with patch_services({"personal": svc}):
        result = await run(action="list_events", account="personal")

    assert "No events" in result


@pytest.mark.asyncio
async def test_list_events_shows_account_label():
    from skills.google_calendar.tool import run

    events = [
        {
            "id": "ev1",
            "summary": "Team standup",
            "start": {"dateTime": "2026-03-16T09:00:00+00:00"},
            "end": {"dateTime": "2026-03-16T09:30:00+00:00"},
        }
    ]
    svc1 = make_service(events=events)
    svc2 = make_service(events=[])
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="list_events")

    assert "Team standup" in result
    # Multi-account: label should include account name
    assert "personal" in result


@pytest.mark.asyncio
async def test_list_events_single_account_no_prefix():
    from skills.google_calendar.tool import run

    events = [
        {
            "id": "ev1",
            "summary": "Solo event",
            "start": {"dateTime": "2026-03-16T09:00:00+00:00"},
            "end": {"dateTime": "2026-03-16T09:30:00+00:00"},
        }
    ]
    svc = make_service(events=events)
    with patch_services({"personal": svc}):
        result = await run(action="list_events", account="personal")

    assert "Solo event" in result
    # Single account: label is just the calendar name, no "personal /" prefix
    assert "personal / " not in result


# ── rsvp_filter / RSVP display ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_events_shows_rsvp_status():
    from skills.google_calendar.tool import run

    events = [
        {
            "id": "ev1",
            "summary": "Team call",
            "start": {"dateTime": "2026-03-16T09:00:00+00:00"},
            "end": {"dateTime": "2026-03-16T09:30:00+00:00"},
            "attendees": [
                {"self": True, "email": "me@example.com", "responseStatus": "accepted"}
            ],
        }
    ]
    svc = make_service(events=events)
    with patch_services({"personal": svc}):
        result = await run(action="list_events", account="personal")

    assert "Team call" in result
    assert "accepted" in result


@pytest.mark.asyncio
async def test_list_events_pending_rsvp_emoji():
    from skills.google_calendar.tool import run

    events = [
        {
            "id": "ev1",
            "summary": "Board meeting",
            "start": {"dateTime": "2026-03-16T10:00:00+00:00"},
            "end": {"dateTime": "2026-03-16T11:00:00+00:00"},
            "attendees": [
                {
                    "self": True,
                    "email": "me@example.com",
                    "responseStatus": "needsAction",
                }
            ],
        }
    ]
    svc = make_service(events=events)
    with patch_services({"personal": svc}):
        result = await run(action="list_events", account="personal")

    assert "📨" in result
    assert "needsAction" in result


@pytest.mark.asyncio
async def test_list_events_rsvp_filter_needs_action():
    from skills.google_calendar.tool import run

    events = [
        {
            "id": "ev1",
            "summary": "Pending invite",
            "start": {"dateTime": "2026-03-16T09:00:00+00:00"},
            "end": {"dateTime": "2026-03-16T09:30:00+00:00"},
            "attendees": [
                {
                    "self": True,
                    "email": "me@example.com",
                    "responseStatus": "needsAction",
                }
            ],
        },
        {
            "id": "ev2",
            "summary": "Already accepted",
            "start": {"dateTime": "2026-03-16T10:00:00+00:00"},
            "end": {"dateTime": "2026-03-16T11:00:00+00:00"},
            "attendees": [
                {"self": True, "email": "me@example.com", "responseStatus": "accepted"}
            ],
        },
    ]
    svc = make_service(events=events)
    with patch_services({"personal": svc}):
        result = await run(
            action="list_events", account="personal", rsvp_filter="needsAction"
        )

    assert "Pending invite" in result
    assert "Already accepted" not in result


@pytest.mark.asyncio
async def test_list_events_rsvp_filter_no_matches():
    from skills.google_calendar.tool import run

    events = [
        {
            "id": "ev1",
            "summary": "Regular event",
            "start": {"dateTime": "2026-03-16T09:00:00+00:00"},
            "end": {"dateTime": "2026-03-16T09:30:00+00:00"},
            # no attendees → no RSVP status
        }
    ]
    svc = make_service(events=events)
    with patch_services({"personal": svc}):
        result = await run(
            action="list_events", account="personal", rsvp_filter="needsAction"
        )

    assert "No events" in result


# ── create_event ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_event_requires_account_when_multi():
    from skills.google_calendar.tool import run

    svc1, svc2 = make_service(), make_service()
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(
            action="create_event",
            account="all",
            summary="Meeting",
            start="2026-03-20T10:00:00",
            end="2026-03-20T11:00:00",
        )

    assert "Error" in result
    assert "account" in result.lower()


@pytest.mark.asyncio
async def test_create_event_with_specific_account():
    from skills.google_calendar.tool import run

    svc = make_service()
    with patch_services({"personal": svc}):
        result = await run(
            action="create_event",
            account="personal",
            summary="Dentist",
            start="2026-03-20T14:00:00",
            end="2026-03-20T15:00:00",
        )

    assert "✅" in result


@pytest.mark.asyncio
async def test_create_event_missing_fields():
    from skills.google_calendar.tool import run

    svc = make_service()
    with patch_services({"personal": svc}):
        result = await run(
            action="create_event", account="personal", summary="No times"
        )

    assert "Error" in result


# ── rsvp_event ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rsvp_event_accepted():
    from skills.google_calendar.tool import run

    svc = MagicMock()
    svc.events().get().execute.return_value = {
        "id": "evt1",
        "summary": "Team Standup",
        "start": {"dateTime": "2026-03-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T09:30:00+00:00"},
        "attendees": [
            {"self": True, "email": "me@example.com", "responseStatus": "needsAction"}
        ],
    }
    svc.events().patch().execute.return_value = {
        "id": "evt1",
        "summary": "Team Standup",
        "start": {"dateTime": "2026-03-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T09:30:00+00:00"},
        "attendees": [
            {"self": True, "email": "me@example.com", "responseStatus": "accepted"}
        ],
    }
    with patch_services({"work": svc}):
        result = await run(
            action="rsvp_event", account="work", event_id="evt1", response="accepted"
        )

    assert "✅" in result
    assert "accepted" in result


@pytest.mark.asyncio
async def test_rsvp_event_declined():
    from skills.google_calendar.tool import run

    svc = MagicMock()
    svc.events().get().execute.return_value = {
        "id": "evt1",
        "summary": "All Hands",
        "start": {"dateTime": "2026-03-15T10:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T11:00:00+00:00"},
        "attendees": [
            {"self": True, "email": "me@example.com", "responseStatus": "needsAction"}
        ],
    }
    svc.events().patch().execute.return_value = {
        "id": "evt1",
        "summary": "All Hands",
        "start": {"dateTime": "2026-03-15T10:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T11:00:00+00:00"},
        "attendees": [
            {"self": True, "email": "me@example.com", "responseStatus": "declined"}
        ],
    }
    with patch_services({"work": svc}):
        result = await run(
            action="rsvp_event", account="work", event_id="evt1", response="declined"
        )

    assert "❌" in result
    assert "declined" in result


@pytest.mark.asyncio
async def test_rsvp_event_missing_event_id():
    from skills.google_calendar.tool import run

    svc = MagicMock()
    with patch_services({"personal": svc}):
        result = await run(action="rsvp_event", account="personal", response="accepted")

    assert "Error" in result
    assert "event_id" in result


@pytest.mark.asyncio
async def test_rsvp_event_missing_response():
    from skills.google_calendar.tool import run

    svc = MagicMock()
    with patch_services({"personal": svc}):
        result = await run(action="rsvp_event", account="personal", event_id="evt1")

    assert "Error" in result
    assert "response" in result


@pytest.mark.asyncio
async def test_rsvp_event_invalid_response():
    from skills.google_calendar.tool import run

    svc = MagicMock()
    svc.events().get().execute.return_value = {
        "id": "evt1",
        "summary": "Meeting",
        "start": {"dateTime": "2026-03-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T10:00:00+00:00"},
        "attendees": [
            {"self": True, "email": "me@example.com", "responseStatus": "needsAction"}
        ],
    }
    with patch_services({"personal": svc}):
        result = await run(
            action="rsvp_event", account="personal", event_id="evt1", response="maybe"
        )

    assert "Error" in result


@pytest.mark.asyncio
async def test_rsvp_event_no_attendee_self():
    from skills.google_calendar.tool import run

    svc = MagicMock()
    svc.events().get().execute.return_value = {
        "id": "evt1",
        "summary": "Personal reminder",
        "start": {"dateTime": "2026-03-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-03-15T10:00:00+00:00"},
        "attendees": [],  # no self attendee
    }
    with patch_services({"personal": svc}):
        result = await run(
            action="rsvp_event",
            account="personal",
            event_id="evt1",
            response="accepted",
        )

    assert "Error" in result


@pytest.mark.asyncio
async def test_rsvp_event_requires_account_when_multi():
    from skills.google_calendar.tool import run

    svc1, svc2 = MagicMock(), MagicMock()
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="rsvp_event", event_id="evt1", response="accepted")

    assert "Error" in result
    assert "account" in result.lower()


# ── update / delete ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_event_requires_account_when_multi():
    from skills.google_calendar.tool import run

    svc1, svc2 = make_service(), make_service()
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="update_event", event_id="evt1")

    assert "Error" in result


@pytest.mark.asyncio
async def test_update_event_success():
    from skills.google_calendar.tool import run

    svc = make_service()
    with patch_services({"work": svc}):
        result = await run(
            action="update_event", account="work", event_id="evt1", summary="New Title"
        )

    assert "✅" in result
    assert "New Title" in result


@pytest.mark.asyncio
async def test_delete_event_missing_id():
    from skills.google_calendar.tool import run

    svc = make_service()
    with patch_services({"personal": svc}):
        result = await run(action="delete_event", account="personal")

    assert "Error" in result


@pytest.mark.asyncio
async def test_delete_event_success():
    from skills.google_calendar.tool import run

    svc = make_service()
    with patch_services({"personal": svc}):
        result = await run(
            action="delete_event",
            account="personal",
            event_id="evt1",
            calendar_id="primary",
        )

    assert "✅" in result


# ── find_conflicts ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_conflicts_no_overlap():
    from skills.google_calendar.tool import run

    svc1 = make_service(
        events=[
            {
                "id": "e1",
                "summary": "Morning",
                "start": {"dateTime": "2026-03-16T08:00:00+00:00"},
                "end": {"dateTime": "2026-03-16T09:00:00+00:00"},
            }
        ]
    )
    svc2 = make_service(
        events=[
            {
                "id": "e2",
                "summary": "Afternoon",
                "start": {"dateTime": "2026-03-16T14:00:00+00:00"},
                "end": {"dateTime": "2026-03-16T15:00:00+00:00"},
            }
        ]
    )
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="find_conflicts")

    assert "No scheduling conflicts" in result


@pytest.mark.asyncio
async def test_find_conflicts_detects_cross_account_overlap():
    from skills.google_calendar.tool import run

    ev = {
        "id": "e1",
        "summary": "Meeting",
        "start": {"dateTime": "2026-03-16T10:00:00+00:00"},
        "end": {"dateTime": "2026-03-16T11:30:00+00:00"},
    }
    ev2 = {
        "id": "e2",
        "summary": "Other meeting",
        "start": {"dateTime": "2026-03-16T11:00:00+00:00"},
        "end": {"dateTime": "2026-03-16T12:00:00+00:00"},
    }
    svc1 = make_service(events=[ev])
    svc2 = make_service(events=[ev2])
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="find_conflicts")

    assert "conflict" in result.lower()
    assert "Meeting" in result


# ── find_duplicates ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_duplicates_no_candidates():
    from skills.google_calendar.tool import run

    svc1 = make_service(
        events=[
            {
                "id": "e1",
                "summary": "Morning run",
                "start": {"dateTime": "2026-03-16T07:00:00+00:00"},
                "end": {"dateTime": "2026-03-16T08:00:00+00:00"},
            }
        ]
    )
    svc2 = make_service(
        events=[
            {
                "id": "e2",
                "summary": "Afternoon meeting",
                "start": {"dateTime": "2026-03-16T15:00:00+00:00"},
                "end": {"dateTime": "2026-03-16T16:00:00+00:00"},
            }
        ]
    )
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="find_duplicates")

    assert "No potential duplicate" in result


@pytest.mark.asyncio
async def test_find_duplicates_cross_account():
    from skills.google_calendar.tool import run

    svc1 = make_service(
        events=[
            {
                "id": "p1",
                "summary": "Dentist",
                "start": {"dateTime": "2026-03-18T14:00:00+00:00"},
                "end": {"dateTime": "2026-03-18T15:00:00+00:00"},
            }
        ]
    )
    svc2 = make_service(
        events=[
            {
                "id": "w1",
                "summary": "dentist appt (personal)",
                "start": {"dateTime": "2026-03-18T14:00:00+00:00"},
                "end": {"dateTime": "2026-03-18T15:00:00+00:00"},
            }
        ]
    )
    with patch_services({"personal": svc1, "work": svc2}):
        result = await run(action="find_duplicates")

    assert "Group" in result
    assert "Dentist" in result
    assert "dentist appt" in result


# ── setup_account ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_setup_account_missing_credentials_path():
    from skills.google_calendar.tool import run

    result = await run(action="setup_account", account_name="personal")
    assert "Error" in result
    assert "credentials_path" in result


@pytest.mark.asyncio
async def test_setup_account_missing_account_name():
    from skills.google_calendar.tool import run

    result = await run(action="setup_account", credentials_path="/some/path.json")
    assert "Error" in result
    assert "account_name" in result


@pytest.mark.asyncio
async def test_setup_account_file_not_found():
    from skills.google_calendar.tool import run

    result = await run(
        action="setup_account",
        credentials_path="/nonexistent/path.json",
        account_name="personal",
    )
    assert "Error" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_setup_account_invalid_json(tmp_path):
    from skills.google_calendar.tool import run

    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")
    result = await run(
        action="setup_account", credentials_path=str(bad), account_name="personal"
    )
    assert "Error" in result
    assert "JSON" in result


@pytest.mark.asyncio
async def test_setup_account_wrong_json_structure(tmp_path):
    from skills.google_calendar.tool import run

    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"foo": "bar"}')
    result = await run(
        action="setup_account", credentials_path=str(wrong), account_name="personal"
    )
    assert "Error" in result
    assert "credentials" in result.lower() or "installed" in result


@pytest.mark.asyncio
async def test_setup_account_success(tmp_path):
    """Valid credentials file is copied to config/ and OAuth runs."""
    from skills.google_calendar.tool import run

    # Source file lives in a separate upload dir (not config/)
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    creds_file = upload_dir / "google_credentials_personal.json"
    creds_file.write_text('{"installed": {"client_id": "x", "client_secret": "y"}}')

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    fake_service = make_service()

    with (
        patch("skills.google_calendar.tool._CONFIG_DIR", config_dir),
        patch(
            "skills.google_calendar.tool._ACCOUNTS_FILE",
            config_dir / "google_accounts.json",
        ),
        patch("skills.google_calendar.tool._get_service", return_value=fake_service),
    ):
        result = await run(
            action="setup_account",
            credentials_path=str(creds_file),
            account_name="personal",
        )

    assert "✅" in result
    assert "personal" in result
    import json

    accounts = json.loads((config_dir / "google_accounts.json").read_text())
    assert any(a["name"] == "personal" for a in accounts)


# ── account config errors ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_credentials_returns_setup_message():
    from skills.google_calendar.tool import run

    with patch(
        "skills.google_calendar.tool._get_services",
        side_effect=FileNotFoundError("Missing credentials for account 'personal'"),
    ):
        result = await run(action="list_events")

    assert "Setup required" in result


@pytest.mark.asyncio
async def test_unknown_account_returns_error():
    from skills.google_calendar.tool import run

    with patch(
        "skills.google_calendar.tool._get_services",
        side_effect=ValueError("Unknown account 'bogus'"),
    ):
        result = await run(action="list_events", account="bogus")

    assert "Config error" in result


# ── unknown action ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_action():
    from skills.google_calendar.tool import run

    svc = make_service()
    with patch_services({"personal": svc}):
        result = await run(action="explode")

    assert "Unknown action" in result
