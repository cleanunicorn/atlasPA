"""
tests/test_channels.py

Tests for Phase 5 channels: Discord and Web UI.
"""

import pytest
from unittest.mock import AsyncMock, patch

from providers.base import LLMResponse
from tests.test_brain import make_brain


# ── Discord ────────────────────────────────────────────────────────────────────


@pytest.fixture
def discord_brain(tmp_memory, empty_skills):
    brain, _ = make_brain(
        [LLMResponse(content="Hello from Discord!", tool_calls=[])],
        tmp_memory,
        empty_skills,
    )
    return brain


def make_discord_bot(brain):
    """Create a DiscordBot with a fake token (no real connection)."""
    with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "fake-token-for-tests"}):
        # Patch discord.Client to avoid actual network setup
        with patch("channels.discord.bot.discord.Client"):
            with patch("channels.discord.bot.app_commands.CommandTree"):
                from channels.discord.bot import DiscordBot

                bot = DiscordBot(brain=brain)
    return bot


def test_discord_bot_requires_token(discord_brain):
    """DiscordBot raises ValueError if DISCORD_BOT_TOKEN is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("channels.discord.bot.discord.Client"):
            with patch("channels.discord.bot.app_commands.CommandTree"):
                from channels.discord.bot import DiscordBot

                with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
                    DiscordBot(brain=discord_brain)


def test_discord_bot_allowed_users_parsing(discord_brain):
    """DISCORD_ALLOWED_USERS env var is correctly parsed into a set of ints."""
    with patch.dict(
        "os.environ",
        {
            "DISCORD_BOT_TOKEN": "fake",
            "DISCORD_ALLOWED_USERS": "111,222, 333",
        },
    ):
        with patch("channels.discord.bot.discord.Client"):
            with patch("channels.discord.bot.app_commands.CommandTree"):
                from channels.discord.bot import DiscordBot

                bot = DiscordBot(brain=discord_brain)
    assert bot._allowed_users == {111, 222, 333}


def test_discord_bot_is_allowed_empty_allows_all(discord_brain):
    """When _allowed_users is empty, every user ID is allowed."""
    bot = make_discord_bot(discord_brain)
    bot._allowed_users = set()
    assert bot._is_allowed(99999) is True


def test_discord_bot_is_allowed_enforces_list(discord_brain):
    """When _allowed_users is set, only listed IDs pass."""
    bot = make_discord_bot(discord_brain)
    bot._allowed_users = {42}
    assert bot._is_allowed(42) is True
    assert bot._is_allowed(99) is False


@pytest.mark.asyncio
async def test_discord_push_message_no_users_logs(discord_brain):
    """push_message with empty allowed_users logs a warning and does nothing."""
    bot = make_discord_bot(discord_brain)
    bot._allowed_users = set()
    # Should not raise
    await bot.push_message("Hello!", [])


@pytest.mark.asyncio
async def test_discord_send_long_splits_at_2000():
    """_send_long calls send_fn multiple times for text > 2000 chars."""
    from channels.discord.bot import _send_long

    calls = []

    async def mock_send(text):
        calls.append(text)

    long_text = "x" * 4500
    await _send_long(mock_send, long_text)

    assert len(calls) == 3
    assert all(len(c) <= 2000 for c in calls)
    assert "".join(calls) == long_text


@pytest.mark.asyncio
async def test_discord_send_file_missing_replies_error(tmp_path):
    """_send_file sends an error message if the file doesn't exist."""
    from channels.discord.bot import _send_file

    replies = []

    async def mock_reply(*args, **kwargs):
        replies.append(args[0] if args else kwargs.get("content", ""))

    await _send_file(mock_reply, tmp_path / "nonexistent.png", "")
    assert any("not found" in r for r in replies)


# ── Web UI ─────────────────────────────────────────────────────────────────────


def make_web_bot(brain):
    """Create a WebBot without starting uvicorn."""
    from channels.web.bot import WebBot

    return WebBot(brain=brain, host="127.0.0.1", port=9999)


@pytest.mark.asyncio
async def test_web_bot_sends_text_response(tmp_memory, empty_skills):
    """WebSocket handler returns a JSON text message after brain.think()."""
    brain, _ = make_brain(
        [LLMResponse(content="Web says hi!", tool_calls=[])],
        tmp_memory,
        empty_skills,
    )
    bot = make_web_bot(brain)

    sent = []

    mock_ws = AsyncMock()
    mock_ws.receive_json = AsyncMock(
        side_effect=[
            {"message": "Hello"},
            Exception("stop"),  # ends the loop
        ]
    )

    async def capture_send(data):
        sent.append(data)

    mock_ws.send_json = AsyncMock(side_effect=capture_send)

    # Grab the ws handler from the FastAPI routes
    ws_route = next(
        r for r in bot.app.routes if getattr(r, "path", "") == "/ws/{session_id}"
    )
    try:
        await ws_route.endpoint(mock_ws, session_id="test-session")
    except Exception:
        pass

    assert any(
        m.get("type") == "text" and "Web says hi!" in m.get("content", "") for m in sent
    )


@pytest.mark.asyncio
async def test_web_bot_clear_command(tmp_memory, empty_skills):
    """/clear command clears history and returns confirmation."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = make_web_bot(brain)

    sent = []
    mock_ws = AsyncMock()
    mock_ws.receive_json = AsyncMock(
        side_effect=[
            {"message": "/clear"},
            Exception("stop"),
        ]
    )
    mock_ws.send_json = AsyncMock(side_effect=lambda d: sent.append(d))

    ws_route = next(
        r for r in bot.app.routes if getattr(r, "path", "") == "/ws/{session_id}"
    )
    try:
        await ws_route.endpoint(mock_ws, session_id="clear-test")
    except Exception:
        pass

    assert any("cleared" in m.get("content", "").lower() for m in sent)


@pytest.mark.asyncio
async def test_web_bot_error_returned_as_json(tmp_memory, empty_skills):
    """If brain.think() raises, an error JSON is sent to the WebSocket."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    brain.think = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    bot = make_web_bot(brain)

    sent = []
    mock_ws = AsyncMock()
    mock_ws.receive_json = AsyncMock(
        side_effect=[
            {"message": "Hello"},
            Exception("stop"),
        ]
    )
    mock_ws.send_json = AsyncMock(side_effect=lambda d: sent.append(d))

    ws_route = next(
        r for r in bot.app.routes if getattr(r, "path", "") == "/ws/{session_id}"
    )
    try:
        await ws_route.endpoint(mock_ws, session_id="err-session")
    except Exception:
        pass

    assert any(m.get("type") == "error" for m in sent)


def test_web_ui_html_is_served(tmp_memory, empty_skills):
    """GET / returns the HTML file."""
    from fastapi.testclient import TestClient

    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = make_web_bot(brain)

    client = TestClient(bot.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Atlas" in response.text


@pytest.mark.asyncio
async def test_web_push_message_logs(tmp_memory, empty_skills, caplog):
    """push_message on WebBot logs but does not raise."""
    import logging

    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = make_web_bot(brain)

    with caplog.at_level(logging.INFO, logger="channels.web.bot"):
        await bot.push_message("Proactive!", [])

    assert any("push" in r.message.lower() for r in caplog.records)


# ── Telegram reply context ──────────────────────────────────────────────────────


def _make_telegram_bot(brain):
    """Create a TelegramBot with a fake token (no real connection)."""
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake-token-for-tests"}):
        with patch("channels.telegram.bot.Application"):
            from channels.telegram.bot import TelegramBot

            bot = TelegramBot(brain=brain)
    return bot


def _make_update(
    replied_text=None, replied_caption=None, replied_media=None, sender_name=None
):
    """Build a minimal mock Update with an optional reply_to_message."""
    from unittest.mock import MagicMock

    update = MagicMock()

    if replied_text is None and replied_caption is None and replied_media is None:
        update.message.reply_to_message = None
        return update

    replied = MagicMock()
    replied.text = replied_text
    replied.caption = replied_caption

    # Set all media attributes to None by default
    replied.photo = None
    replied.document = None
    replied.voice = None
    replied.audio = None
    replied.sticker = None

    if replied_media == "photo":
        replied.photo = MagicMock()
    elif replied_media == "document":
        doc = MagicMock()
        doc.file_name = "report.pdf"
        replied.document = doc
    elif replied_media == "voice":
        replied.voice = MagicMock()

    if sender_name:
        replied.from_user = MagicMock()
        replied.from_user.first_name = sender_name
        replied.from_user.username = sender_name
    else:
        replied.from_user = None

    update.message.reply_to_message = replied
    return update


def test_telegram_build_reply_context_no_reply(tmp_memory, empty_skills):
    """Returns empty string when the message is not a reply."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = _make_telegram_bot(brain)

    update = _make_update()
    assert bot._build_reply_context(update) == ""


def test_telegram_build_reply_context_text_reply(tmp_memory, empty_skills):
    """Quoted text appears in the context string."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = _make_telegram_bot(brain)

    update = _make_update(replied_text="What's the weather today?", sender_name="Alice")
    ctx = bot._build_reply_context(update)
    assert "Alice" in ctx
    assert "What's the weather today?" in ctx


def test_telegram_build_reply_context_no_sender(tmp_memory, empty_skills):
    """Works correctly when the sender name is unavailable."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = _make_telegram_bot(brain)

    update = _make_update(replied_text="Hello world")
    ctx = bot._build_reply_context(update)
    assert "Hello world" in ctx
    assert ctx.startswith("[Replying to:")


def test_telegram_build_reply_context_photo_reply(tmp_memory, empty_skills):
    """Returns [photo] label when replying to a photo with no caption."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = _make_telegram_bot(brain)

    update = _make_update(replied_media="photo")
    ctx = bot._build_reply_context(update)
    assert "[photo]" in ctx


def test_telegram_build_reply_context_document_reply(tmp_memory, empty_skills):
    """Returns file name label when replying to a document."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = _make_telegram_bot(brain)

    update = _make_update(replied_media="document")
    ctx = bot._build_reply_context(update)
    assert "report.pdf" in ctx


def test_telegram_build_reply_context_truncates_long_text(tmp_memory, empty_skills):
    """Very long quoted messages are truncated at 300 characters."""
    brain, _ = make_brain([], tmp_memory, empty_skills)
    bot = _make_telegram_bot(brain)

    long_text = "a" * 500
    update = _make_update(replied_text=long_text)
    ctx = bot._build_reply_context(update)
    # The context string itself will be longer than 300 due to prefix, but quoted portion ≤ 300
    assert "…" in ctx
    assert long_text[:50] in ctx
