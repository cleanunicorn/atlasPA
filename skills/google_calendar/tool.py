"""
skills/google_calendar/tool.py

Google Calendar skill — read and manage events across multiple Google accounts
and calendars.

Multi-account setup:
    Create config/google_accounts.json listing each account:

        [
          {"name": "personal", "credentials": "google_credentials_personal.json"},
          {"name": "work",     "credentials": "google_credentials_work.json"}
        ]

    Each entry needs its own OAuth credentials file downloaded from the Google
    Cloud Console. Token files (google_token_<name>.json) are auto-created on
    first use.

Single-account setup (backward compat):
    If google_accounts.json doesn't exist, the skill falls back to
    config/google_credentials.json + config/google_token.json as a single
    account named "default".

Auth flow:
    First use of each account opens a browser for OAuth2 consent.
    Tokens are saved and refreshed automatically.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
from paths import CONFIG_DIR as _CONFIG_DIR

_ACCOUNTS_FILE = _CONFIG_DIR / "google_accounts.json"
_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# ── Parameters ─────────────────────────────────────────────────────────────────
PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "list_accounts",
                "setup_account",
                "list_calendars",
                "list_events",
                "create_event",
                "update_event",
                "delete_event",
                "rsvp_event",
                "find_conflicts",
                "find_duplicates",
            ],
            "description": "The calendar operation to perform.",
        },
        "account": {
            "type": "string",
            "description": (
                "Which Google account to use. 'all' (default) covers every configured "
                "account. Use the account name from list_accounts to target one "
                "(e.g. 'personal' or 'work'). Required for write operations when "
                "multiple accounts are configured."
            ),
        },
        "calendar_id": {
            "type": "string",
            "description": (
                "Calendar ID within the account. Use 'all' for all calendars (default), "
                "'primary' for the main calendar, or a specific ID from list_calendars."
            ),
        },
        "time_min": {
            "type": "string",
            "description": (
                "Start of time window (ISO 8601). Default: now. "
                "Examples: '2026-03-15', '2026-03-15T09:00:00'."
            ),
        },
        "time_max": {
            "type": "string",
            "description": (
                "End of time window (ISO 8601). Default: 7 days from now. "
                "Examples: '2026-03-22', '2026-03-22T23:59:59'."
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of events to return (default 20, max 250).",
        },
        "summary": {
            "type": "string",
            "description": "Event title (required for create_event).",
        },
        "start": {
            "type": "string",
            "description": "Event start (ISO 8601, required for create_event).",
        },
        "end": {
            "type": "string",
            "description": "Event end (ISO 8601, required for create_event).",
        },
        "description": {
            "type": "string",
            "description": "Event description / notes.",
        },
        "location": {
            "type": "string",
            "description": "Event location string.",
        },
        "rsvp_filter": {
            "type": "string",
            "enum": ["needsAction", "accepted", "declined", "tentative"],
            "description": (
                "Filter list_events to only show events with this RSVP status. "
                "Use 'needsAction' to see pending invitations that need a response."
            ),
        },
        "event_id": {
            "type": "string",
            "description": "Google Calendar event ID (required for update_event / delete_event / rsvp_event).",
        },
        "response": {
            "type": "string",
            "enum": ["accepted", "declined", "tentative"],
            "description": "RSVP response status (required for rsvp_event).",
        },
        "credentials_path": {
            "type": "string",
            "description": (
                "Absolute path to the downloaded Google OAuth credentials JSON file "
                "(required for setup_account). Usually ~/agent-files/uploads/<filename>."
            ),
        },
        "account_name": {
            "type": "string",
            "description": (
                "Short name for the account being set up, e.g. 'personal' or 'work' "
                "(required for setup_account). Used as the key in google_accounts.json."
            ),
        },
    },
    "required": ["action"],
}


# ── Account config ─────────────────────────────────────────────────────────────


def _load_account_configs() -> list[dict]:
    """
    Return a list of account dicts, each with:
        name, creds_file (Path), token_file (Path)

    Reads google_accounts.json if present; otherwise falls back to the legacy
    single-file setup (google_credentials.json / google_token.json).
    """
    if _ACCOUNTS_FILE.exists():
        try:
            entries = json.loads(_ACCOUNTS_FILE.read_text())
        except Exception as e:
            raise ValueError(f"Invalid config/google_accounts.json: {e}")

        accounts = []
        for entry in entries:
            name = entry.get("name", "").strip()
            creds_filename = entry.get("credentials", "")
            if not name or not creds_filename:
                continue
            creds_file = _CONFIG_DIR / creds_filename
            token_file = _CONFIG_DIR / f"google_token_{name}.json"
            accounts.append(
                {"name": name, "creds_file": creds_file, "token_file": token_file}
            )

        if not accounts:
            raise ValueError(
                "config/google_accounts.json exists but contains no valid entries."
            )
        return accounts

    # Legacy single-account fallback
    creds_file = _CONFIG_DIR / "google_credentials.json"
    token_file = _CONFIG_DIR / "google_token.json"
    return [{"name": "default", "creds_file": creds_file, "token_file": token_file}]


# ── Auth helper ────────────────────────────────────────────────────────────────


def _get_service(account: dict):
    """Return an authenticated Google Calendar service for one account."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError(
            "Google client libraries not installed. "
            "Run: uv add google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )

    creds_file: Path = account["creds_file"]
    token_file: Path = account["token_file"]
    name: str = account["name"]

    if not creds_file.exists():
        raise FileNotFoundError(
            f"Missing credentials for account '{name}': {creds_file}. "
            "Download OAuth 2.0 credentials (Desktop app) from Google Cloud Console."
        )

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), _SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _get_services(account_name: str) -> list[tuple[str, object]]:
    """
    Return [(account_name, service), ...] for the requested account(s).
    account_name = "all" → all configured accounts.
    """
    configs = _load_account_configs()

    if account_name != "all":
        match = next((a for a in configs if a["name"] == account_name), None)
        if match is None:
            known = ", ".join(a["name"] for a in configs)
            raise ValueError(
                f"Unknown account '{account_name}'. Configured accounts: {known}"
            )
        configs = [match]

    results = []
    errors = []
    for cfg in configs:
        try:
            svc = _get_service(cfg)
            results.append((cfg["name"], svc))
        except FileNotFoundError as e:
            errors.append(str(e))

    if not results:
        raise FileNotFoundError("\n".join(errors))

    return results


# ── Date helpers ───────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ahead_iso(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _ensure_tz(s: str) -> str:
    """Ensure an ISO 8601 datetime string has timezone info (appends +00:00 if missing)."""
    if not s:
        return s
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return s


def _parse_dt(s: str) -> dict:
    if "T" in s or len(s) > 10:
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return {"dateTime": dt.isoformat(), "timeZone": "UTC"}
        except ValueError:
            return {"dateTime": s}
    return {"date": s}


def _label(account_name: str, calendar_name: str, multi_account: bool) -> str:
    """Build the [Account / Calendar] label for event output."""
    if multi_account and account_name != "default":
        return f"{account_name} / {calendar_name}"
    return calendar_name


_RSVP_EMOJI = {
    "accepted": "✅",
    "declined": "❌",
    "tentative": "❓",
    "needsAction": "📨",
}


def _self_rsvp(event: dict) -> str | None:
    """Return the current user's responseStatus for an event, or None if not an invite."""
    for attendee in event.get("attendees", []):
        if attendee.get("self"):
            return attendee.get("responseStatus")
    return None


def _fmt_event(event: dict) -> str:
    summary = event.get("summary", "(no title)")
    start = event.get("start", {})
    dt_str = start.get("dateTime", start.get("date", "?"))
    try:
        dt = datetime.fromisoformat(dt_str)
        dt_str = (
            dt.strftime("%a %d %b %Y, %H:%M")
            if "T" in dt_str
            else dt.strftime("%a %d %b %Y")
        )
    except Exception:
        pass

    end = event.get("end", {})
    end_str = end.get("dateTime", end.get("date", ""))
    try:
        end_dt = datetime.fromisoformat(end_str)
        end_str = end_dt.strftime("%H:%M") if "T" in end_str else ""
    except Exception:
        end_str = ""

    loc = event.get("location", "")
    eid = event.get("id", "")
    label = event.get("_label", "")
    rsvp = _self_rsvp(event)

    line1 = f"• {dt_str}"
    if end_str:
        line1 += f"–{end_str}"
    if rsvp:
        line1 += f"  {_RSVP_EMOJI.get(rsvp, '')} {summary}"
    else:
        line1 += f"  {summary}"

    line2_parts = []
    if rsvp:
        line2_parts.append(f"rsvp:{rsvp}")
    if loc:
        line2_parts.append(f"📍 {loc}")
    if label:
        line2_parts.append(f"[{label}]")
    line2_parts.append(f"id:{eid}")

    return line1 + "\n  " + "  ".join(line2_parts)


# ── Helpers for gathering events across accounts ────────────────────────────────


def _gather_timed_events(
    services: list[tuple[str, object]],
    time_min: str,
    time_max: str,
    calendar_id: str = "all",
    max_results: int = 250,
) -> list[dict]:
    """
    Collect all timed (non all-day) events from the given services.
    Tags each event with _label, _account_name, _calendar_name, _start_dt, _end_dt.
    ID-deduplicates within each account (not across — same event on two accounts
    IS a duplicate candidate).
    """
    multi_account = len(services) > 1
    all_events: list[dict] = []

    for acct_name, svc in services:
        if calendar_id == "all":
            cal_list = svc.calendarList().list().execute().get("items", [])
        else:
            cal_list = [{"id": calendar_id, "summary": calendar_id}]

        seen_in_account: set[str] = set()
        for cal in cal_list:
            try:
                result = (
                    svc.events()
                    .list(
                        calendarId=cal["id"],
                        timeMin=time_min,
                        timeMax=time_max,
                        maxResults=max_results,
                        singleEvents=True,
                        orderBy="startTime",
                    )
                    .execute()
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to gather events for calendar %s: %s", cal.get("id"), e
                )
                continue

            cal_name = cal.get("summary", cal["id"])
            for ev in result.get("items", []):
                if "dateTime" not in ev.get("start", {}):
                    continue
                eid = ev.get("id", "")
                if eid and eid in seen_in_account:
                    continue
                seen_in_account.add(eid)
                try:
                    ev["_start_dt"] = datetime.fromisoformat(ev["start"]["dateTime"])
                    ev["_end_dt"] = datetime.fromisoformat(ev["end"]["dateTime"])
                except Exception:
                    continue
                ev["_account_name"] = acct_name
                ev["_calendar_name"] = cal_name
                ev["_label"] = _label(acct_name, cal_name, multi_account)
                all_events.append(ev)

    return all_events


# ── Actions ────────────────────────────────────────────────────────────────────


def _setup_account(credentials_path: str, account_name: str) -> str:
    """
    Install a Google OAuth credentials file and authenticate the account.

    Steps:
      1. Validate the JSON looks like a Google OAuth credentials file.
      2. Copy it to config/google_credentials_<account_name>.json.
      3. Add / update the entry in config/google_accounts.json.
      4. Run the OAuth browser flow to obtain and save a token.
    """
    import shutil

    src = Path(credentials_path).expanduser()
    if not src.exists():
        return f"Error: file not found at {credentials_path}"

    # Validate JSON structure
    try:
        data = json.loads(src.read_text())
    except json.JSONDecodeError:
        return "Error: the file is not valid JSON."

    if "installed" not in data and "web" not in data:
        return (
            "Error: this doesn't look like a Google OAuth 2.0 credentials file. "
            "Expected an 'installed' or 'web' key. Download the correct file from "
            "Google Cloud Console → APIs & Services → Credentials → OAuth 2.0 Client IDs → Download JSON."
        )

    account_name = account_name.strip().lower().replace(" ", "_")
    if not account_name:
        return "Error: account_name is required (e.g. 'personal' or 'work')."

    # Copy to config/
    creds_filename = f"google_credentials_{account_name}.json"
    target = _CONFIG_DIR / creds_filename
    shutil.copy2(src, target)

    # Update google_accounts.json
    if _ACCOUNTS_FILE.exists():
        try:
            accounts = json.loads(_ACCOUNTS_FILE.read_text())
        except Exception:
            accounts = []
    else:
        accounts = []

    accounts = [a for a in accounts if a.get("name") != account_name]
    accounts.append({"name": account_name, "credentials": creds_filename})
    _ACCOUNTS_FILE.write_text(json.dumps(accounts, indent=2))

    # Trigger OAuth flow (opens browser on local machine)
    cfg = {
        "name": account_name,
        "creds_file": target,
        "token_file": _CONFIG_DIR / f"google_token_{account_name}.json",
    }
    try:
        _get_service(cfg)
        return (
            f"✅ Account '{account_name}' configured and authenticated!\n"
            f"   Credentials saved to config/{creds_filename}\n"
            f"   Token saved to config/google_token_{account_name}.json\n"
            f"   You can now use list_calendars or list_events with account='{account_name}'."
        )
    except Exception as e:
        return (
            f"Credentials saved for account '{account_name}', but OAuth failed: {e}\n"
            "Make sure a browser is available on this machine. "
            f"You can retry by calling list_calendars with account='{account_name}'."
        )


def _list_accounts(configs: list[dict]) -> str:
    lines = ["Configured Google accounts:\n"]
    for cfg in configs:
        status = (
            "✅ credentials found"
            if cfg["creds_file"].exists()
            else "⚠️  credentials missing"
        )
        token_status = (
            "token saved" if cfg["token_file"].exists() else "not yet authenticated"
        )
        lines.append(f"  • {cfg['name']}  [{status}, {token_status}]")
        lines.append(f"    credentials: {cfg['creds_file'].name}")
    return "\n".join(lines)


def _list_calendars(services: list[tuple[str, object]], multi_account: bool) -> str:
    lines = []
    for acct_name, svc in services:
        if multi_account:
            lines.append(f"Account: {acct_name}")
        items = svc.calendarList().list().execute().get("items", [])
        if not items:
            lines.append("  (no calendars found)")
            continue
        for c in items:
            primary = " (primary)" if c.get("primary") else ""
            lines.append(f"  • {c['summary']}{primary}")
            lines.append(f"    id: {c['id']}")
        if multi_account:
            lines.append("")
    return "\n".join(lines) if lines else "No calendars found."


def _list_events_multi(
    services: list[tuple[str, object]],
    calendar_id: str,
    time_min: str,
    time_max: str,
    max_results: int,
    rsvp_filter: str = "",
) -> str:
    multi_account = len(services) > 1
    all_events: list[dict] = []

    for acct_name, svc in services:
        if calendar_id == "all":
            cal_list = svc.calendarList().list().execute().get("items", [])
        else:
            cal_list = [{"id": calendar_id, "summary": calendar_id}]

        seen: set[str] = set()
        for cal in cal_list:
            try:
                result = (
                    svc.events()
                    .list(
                        calendarId=cal["id"],
                        timeMin=time_min,
                        timeMax=time_max,
                        maxResults=max_results,
                        singleEvents=True,
                        orderBy="startTime",
                    )
                    .execute()
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to list events for calendar %s: %s", cal.get("id"), e
                )
                continue
            cal_name = cal.get("summary", cal["id"])
            for ev in result.get("items", []):
                eid = ev.get("id", "")
                if eid and eid in seen:
                    continue
                seen.add(eid)
                ev["_label"] = _label(acct_name, cal_name, multi_account)
                all_events.append(ev)

    if rsvp_filter:
        all_events = [ev for ev in all_events if _self_rsvp(ev) == rsvp_filter]

    if not all_events:
        if rsvp_filter:
            return f"No events with RSVP status '{rsvp_filter}' found in the specified time range."
        return "No events found in the specified time range."

    def sort_key(ev):
        s = ev.get("start", {})
        return s.get("dateTime", s.get("date", ""))

    all_events.sort(key=sort_key)
    all_events = all_events[:max_results]

    lines = [f"Events ({len(all_events)} found):\n"]
    for ev in all_events:
        lines.append(_fmt_event(ev))
    return "\n".join(lines)


def _create_event(
    service,
    calendar_id: str,
    summary: str,
    start: str,
    end: str,
    description: str,
    location: str,
) -> str:
    body: dict = {"summary": summary, "start": _parse_dt(start), "end": _parse_dt(end)}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    event = service.events().insert(calendarId=calendar_id, body=body).execute()
    return (
        f"✅ Event created: {event.get('summary')}\n"
        f"   Start: {event['start'].get('dateTime', event['start'].get('date'))}\n"
        f"   Calendar: {calendar_id}\n"
        f"   ID: {event['id']}"
    )


def _update_event(service, calendar_id: str, event_id: str, **fields) -> str:
    event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
    for field in ("summary", "description", "location"):
        if fields.get(field):
            event[field] = fields[field]
    if fields.get("start"):
        event["start"] = _parse_dt(fields["start"])
    if fields.get("end"):
        event["end"] = _parse_dt(fields["end"])
    updated = (
        service.events()
        .update(calendarId=calendar_id, eventId=event_id, body=event)
        .execute()
    )
    return (
        f"✅ Event updated: {updated.get('summary')}\n"
        f"   Start: {updated['start'].get('dateTime', updated['start'].get('date'))}\n"
        f"   ID: {event_id}"
    )


def _delete_event(service, calendar_id: str, event_id: str) -> str:
    service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
    return f"✅ Event {event_id} deleted from calendar '{calendar_id}'."


def _rsvp_event(service, calendar_id: str, event_id: str, response: str) -> str:
    """Accept, decline, or mark tentative an event invitation."""
    valid = {"accepted", "declined", "tentative"}
    if response not in valid:
        return f"Error: response must be one of {sorted(valid)}, got '{response}'."

    event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
    attendees = event.get("attendees", [])

    # Find the self attendee and update their status
    self_attendee = next((a for a in attendees if a.get("self")), None)
    if self_attendee is None:
        # Not an invite — patch anyway (some events have no attendee list)
        return (
            f"Error: no attendee entry found for this account on event '{event.get('summary', event_id)}'. "
            "This event may not be an invitation."
        )

    self_attendee["responseStatus"] = response
    event["attendees"] = attendees

    updated = (
        service.events()
        .patch(
            calendarId=calendar_id,
            eventId=event_id,
            body={"attendees": attendees},
            sendUpdates="all",
        )
        .execute()
    )

    emoji = {"accepted": "✅", "declined": "❌", "tentative": "❓"}[response]
    return (
        f"{emoji} RSVP recorded: {response} for '{updated.get('summary', event_id)}'\n"
        f"   Calendar: {calendar_id}  ID: {event_id}"
    )


def _find_conflicts_multi(
    services: list[tuple[str, object]], time_min: str, time_max: str
) -> str:
    all_events = _gather_timed_events(services, time_min, time_max)
    all_events.sort(key=lambda e: e["_start_dt"])

    conflicts = []
    for i, a in enumerate(all_events):
        for b in all_events[i + 1 :]:
            if b["_start_dt"] >= a["_end_dt"]:
                break
            # Only flag if from different sources (account+calendar)
            a_src = (a["_account_name"], a["_calendar_name"])
            b_src = (b["_account_name"], b["_calendar_name"])
            if a_src != b_src:
                conflicts.append((a, b))

    if not conflicts:
        return "No scheduling conflicts found."

    lines = [f"⚠️  {len(conflicts)} conflict(s) found:\n"]
    for a, b in conflicts:
        lines.append(
            f"  CONFLICT:\n"
            f"    • {a.get('summary', '?')} [{a['_label']}] "
            f"{a['_start_dt'].strftime('%d %b %H:%M')}–{a['_end_dt'].strftime('%H:%M')}\n"
            f"    • {b.get('summary', '?')} [{b['_label']}] "
            f"{b['_start_dt'].strftime('%d %b %H:%M')}–{b['_end_dt'].strftime('%H:%M')}"
        )
    return "\n".join(lines)


def _find_duplicates_multi(
    services: list[tuple[str, object]], time_min: str, time_max: str
) -> str:
    """
    Surface candidate duplicate events for LLM reasoning.

    Two events are candidates if they:
      - Come from different sources (different account OR different calendar), AND
      - Overlap in time OR start within DUPLICATE_GAP_MINUTES of each other.

    The LLM then reasons about the titles to determine if they are truly the
    same event copied across calendars/accounts.
    """
    DUPLICATE_GAP_MINUTES = 15

    all_events = _gather_timed_events(services, time_min, time_max)
    all_events.sort(key=lambda e: e["_start_dt"])

    gap = timedelta(minutes=DUPLICATE_GAP_MINUTES)
    groups: list[list[dict]] = []
    used: set[int] = set()

    for i, a in enumerate(all_events):
        if i in used:
            continue
        group = [a]
        a_src = (a["_account_name"], a["_calendar_name"])
        for j, b in enumerate(all_events[i + 1 :], start=i + 1):
            if j in used:
                continue
            b_src = (b["_account_name"], b["_calendar_name"])
            if a_src == b_src:
                continue  # Same source — not a cross-calendar duplicate
            overlap = a["_start_dt"] < b["_end_dt"] and a["_end_dt"] > b["_start_dt"]
            close = abs(a["_start_dt"] - b["_start_dt"]) <= gap
            if overlap or close:
                group.append(b)
                used.add(j)
        if len(group) > 1:
            used.add(i)
            groups.append(group)

    if not groups:
        return "No potential duplicate events found across accounts/calendars."

    lines = [
        f"Found {len(groups)} group(s) of events that may be duplicates.\n"
        "Review each group — these events come from different accounts or calendars "
        "and overlap or start within 15 minutes of each other.\n"
        "Use your judgment (titles, descriptions) to decide if they are the same event:\n"
    ]
    for idx, group in enumerate(groups, 1):
        lines.append(f"Group {idx}:")
        for ev in group:
            start_str = ev["_start_dt"].strftime("%a %d %b %Y, %H:%M")
            end_str = ev["_end_dt"].strftime("%H:%M")
            lines.append(
                f"  • [{ev['_label']}] {ev.get('summary', '(no title)')} "
                f"{start_str}–{end_str}  id:{ev.get('id', '?')}"
            )
        lines.append("")
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────


async def run(
    action: str,
    account: str = "all",
    calendar_id: str = "all",
    time_min: str = "",
    time_max: str = "",
    max_results: int = 20,
    summary: str = "",
    start: str = "",
    end: str = "",
    description: str = "",
    location: str = "",
    rsvp_filter: str = "",
    event_id: str = "",
    response: str = "",
    credentials_path: str = "",
    account_name: str = "",
    **_,
) -> str:
    """Dispatch to the appropriate Google Calendar operation."""

    # Actions that don't need an existing authenticated service
    if action == "list_accounts":
        try:
            configs = await asyncio.to_thread(_load_account_configs)
            return _list_accounts(configs)
        except Exception as e:
            return f"Error reading account config: {e}"

    if action == "setup_account":
        if not credentials_path:
            return "Error: setup_account requires 'credentials_path' (path to the downloaded JSON file)."
        if not account_name:
            return "Error: setup_account requires 'account_name' (e.g. 'personal' or 'work')."
        return await asyncio.to_thread(_setup_account, credentials_path, account_name)

    # All other actions need at least one authenticated service
    try:
        services = await asyncio.to_thread(_get_services, account)
    except FileNotFoundError as e:
        return f"Setup required: {e}"
    except ValueError as e:
        return f"Config error: {e}"
    except Exception as e:
        return f"Auth error: {e}"

    t_min = _ensure_tz(time_min) if time_min else _now_iso()
    t_max = _ensure_tz(time_max) if time_max else _days_ahead_iso(7)
    max_results = min(max(1, max_results), 250)
    cal = calendar_id or "all"
    multi_account = len(services) > 1

    try:
        if action == "list_calendars":
            return await asyncio.to_thread(_list_calendars, services, multi_account)

        elif action == "list_events":
            return await asyncio.to_thread(
                _list_events_multi,
                services,
                cal,
                t_min,
                t_max,
                max_results,
                rsvp_filter,
            )

        elif action in ("create_event", "update_event", "delete_event", "rsvp_event"):
            # Write operations need exactly one account+service
            if multi_account:
                return (
                    f"Error: '{action}' requires a specific account when multiple are configured. "
                    f"Re-call with account='personal' or account='work' (use list_accounts to see options)."
                )
            acct_name, svc = services[0]
            target_cal = cal if cal != "all" else "primary"

            if action == "create_event":
                if not summary or not start or not end:
                    return "Error: create_event requires 'summary', 'start', and 'end'."
                return await asyncio.to_thread(
                    _create_event,
                    svc,
                    target_cal,
                    summary,
                    start,
                    end,
                    description,
                    location,
                )
            elif action == "update_event":
                if not event_id:
                    return "Error: update_event requires 'event_id'."
                return await asyncio.to_thread(
                    _update_event,
                    svc,
                    target_cal,
                    event_id,
                    summary=summary,
                    start=start,
                    end=end,
                    description=description,
                    location=location,
                )
            elif action == "delete_event":
                if not event_id:
                    return "Error: delete_event requires 'event_id'."
                return await asyncio.to_thread(_delete_event, svc, target_cal, event_id)
            else:  # rsvp_event
                if not event_id:
                    return "Error: rsvp_event requires 'event_id'."
                if not response:
                    return "Error: rsvp_event requires 'response' (accepted / declined / tentative)."
                return await asyncio.to_thread(
                    _rsvp_event, svc, target_cal, event_id, response
                )

        elif action == "find_conflicts":
            return await asyncio.to_thread(
                _find_conflicts_multi, services, t_min, t_max
            )

        elif action == "find_duplicates":
            return await asyncio.to_thread(
                _find_duplicates_multi, services, t_min, t_max
            )

        else:
            return (
                f"Unknown action '{action}'. Valid actions: list_accounts, setup_account, "
                "list_calendars, list_events, create_event, update_event, delete_event, "
                "rsvp_event, find_conflicts, find_duplicates."
            )

    except Exception as e:
        return f"Google Calendar error ({action}): {e}"
