# google_calendar

Read and manage Google Calendar events across multiple Google accounts and calendars.

## Actions

| action            | What it does                                                                   |
|-------------------|--------------------------------------------------------------------------------|
| `list_accounts`   | Show all configured Google accounts and their auth status.                     |
| `setup_account`   | Install a credentials file and authenticate a new Google account.              |
| `list_calendars`  | List all calendars across all (or a specific) account.                         |
| `list_events`     | List upcoming events. Defaults to all accounts + all calendars, next 7 days.   |
| `create_event`    | Create a new event. Requires specifying the target account.                    |
| `update_event`    | Update an existing event. Requires specifying the target account.              |
| `delete_event`    | Delete an event. Requires specifying the target account.                       |
| `rsvp_event`      | Accept, decline, or mark tentative an event invitation.                        |
| `find_conflicts`  | Find overlapping events across all accounts and calendars.                     |
| `find_duplicates` | Find events that may be the same meeting copied across accounts or calendars.  |

## Parameters

- `account` — `"all"` (default) or a specific account name like `"personal"` / `"work"`.
  Required for `create_event`, `update_event`, `delete_event`.
- `calendar_id` — `"all"` (default), `"primary"`, or a specific calendar ID from `list_calendars`.
- `response` — `"accepted"`, `"declined"`, or `"tentative"` (required for `rsvp_event`).
- `rsvp_filter` — Filter `list_events` by RSVP status: `"needsAction"`, `"accepted"`, `"declined"`, `"tentative"`.
- `time_min` / `time_max` — ISO 8601 date or datetime. Default: now → +7 days.

## When to use

- "What do I have this week?" → `list_events`
- "Show all my calendars" → `list_calendars`
- "Show only my work calendar" → `list_events` with `account="work"`
- "Schedule a dentist appointment on my personal calendar" → `create_event` with `account="personal"`
- "Do I have any conflicts next week?" → `find_conflicts`
- "Are there any duplicate events across my calendars?" → `find_duplicates`
- "Which accounts are set up?" → `list_accounts`
- "I sent you a credentials file, set it up as my work account" → `setup_account`
- "Accept the team standup invite" → `rsvp_event` with `response="accepted"`
- "Decline the dentist appointment" → `rsvp_event` with `response="declined"`
- "What invites haven't I responded to?" → `list_events` with `rsvp_filter="needsAction"`
- "Which events did I accept this week?" → `list_events` with `rsvp_filter="accepted"`

## Duplicate detection

The user keeps multiple accounts in sync. The same meeting may appear under
different names across accounts/calendars (e.g. "Dentist" on personal and
"dentist appt" on work). Use `find_duplicates` to surface candidate groups,
then reason about whether each group is a single real event or two distinct ones.
When you confirm a duplicate, offer to delete the redundant copy — but always
ask which copy to keep before deleting.

## Setup

### Step 1 — Create config/google_accounts.json

```json
[
  {"name": "personal", "credentials": "google_credentials_personal.json"},
  {"name": "work",     "credentials": "google_credentials_work.json"}
]
```

Add one entry per Google account. The `name` is how you'll refer to the account.

### Step 2 — Download credentials for each account

For each account:
1. Go to console.cloud.google.com (sign in with that Google account)
2. Create a project → enable Google Calendar API
3. APIs & Services → Credentials → Create OAuth client ID → Desktop app
4. Download JSON → save as the filename listed in google_accounts.json
   (e.g. `config/google_credentials_personal.json`)

### Step 3 — Authenticate each account

Run the agent and call `list_calendars`. A browser window opens for each account
that hasn't been authenticated yet. Token files are saved automatically.

### Single-account fallback

If you only have one Google account, skip google_accounts.json and just place
`config/google_credentials.json`. The skill uses it automatically as account "default".

## Notes

- Token files (`google_token_<name>.json`) are auto-created and gitignored.
- To re-authenticate an account, delete its token file.
- Events are labelled `[account / calendar]` in multi-account mode.
