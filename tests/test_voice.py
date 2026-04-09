"""
tests/test_voice.py

Tests for voice message transcription (HTTP client + Docker auto-start)
and Telegram voice handler.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── _to_wav ────────────────────────────────────────────────────────────────────


def test_to_wav_returns_same_path_for_wav(tmp_path):
    """WAV files are returned unchanged."""
    from channels.transcribe import _to_wav

    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)
    assert _to_wav(wav) == wav


def test_to_wav_no_pydub_raises(tmp_path):
    """RuntimeError with install instructions when pydub is missing."""
    import sys

    ogg = tmp_path / "voice.ogg"
    ogg.write_bytes(b"OggS" + b"\x00" * 50)

    with patch.dict(sys.modules, {"pydub": None}):
        import channels.transcribe as t_mod

        with pytest.raises(RuntimeError, match="pydub"):
            t_mod._to_wav(ogg)


def test_to_wav_converts_ogg(tmp_path):
    """OGG file is converted to WAV via pydub."""
    import channels.transcribe as t_mod

    ogg = tmp_path / "voice.ogg"
    ogg.write_bytes(b"fake ogg data")

    mock_audio = MagicMock()
    mock_audio.set_frame_rate.return_value = mock_audio
    mock_audio.set_channels.return_value = mock_audio
    mock_audio.set_sample_width.return_value = mock_audio

    MockSeg = MagicMock()
    MockSeg.from_file.return_value = mock_audio

    with patch("pydub.AudioSegment", MockSeg):
        result = t_mod._to_wav(ogg)

    assert result.suffix == ".wav"
    mock_audio.export.assert_called_once()


# ── _check_server ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_server_true_when_healthy():
    import channels.transcribe as t_mod

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        assert await t_mod._check_server() is True


@pytest.mark.asyncio
async def test_check_server_false_when_unreachable():
    import httpx
    import channels.transcribe as t_mod

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        assert await t_mod._check_server() is False


# ── _start_container ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_container_success():
    """docker run exit 0 → logs container ID, no exception."""
    import channels.transcribe as t_mod

    with patch.object(
        t_mod, "_run_docker", AsyncMock(return_value=(0, "abc123def456", ""))
    ):
        await t_mod._start_container()  # should not raise


@pytest.mark.asyncio
async def test_start_container_name_conflict_is_ok():
    """docker run exit 125 (name conflict) → no exception, just a log."""
    import channels.transcribe as t_mod

    with patch.object(
        t_mod, "_run_docker", AsyncMock(return_value=(125, "", "already in use"))
    ):
        await t_mod._start_container()  # should not raise


@pytest.mark.asyncio
async def test_start_container_docker_not_found():
    """FileNotFoundError from _run_docker → RuntimeError with clear message."""
    import channels.transcribe as t_mod

    async def raise_fnf(_cmd):
        raise RuntimeError("Docker is not installed or not in PATH.")

    with patch.object(t_mod, "_run_docker", raise_fnf):
        with pytest.raises(RuntimeError, match="Docker is not installed"):
            await t_mod._start_container()


@pytest.mark.asyncio
async def test_start_container_docker_run_fails():
    """Non-GPU, non-125 exit code → RuntimeError with stderr."""
    import channels.transcribe as t_mod

    with patch.object(
        t_mod, "_run_docker", AsyncMock(return_value=(1, "", "image not found"))
    ):
        with pytest.raises(RuntimeError, match="docker run failed"):
            await t_mod._start_container()


@pytest.mark.asyncio
async def test_start_container_gpu_error_retries_cpu():
    """GPU runtime error → retries without --gpus, succeeds on CPU."""
    import channels.transcribe as t_mod

    calls = []

    async def mock_run_docker(cmd):
        calls.append(cmd)
        if "--gpus" in cmd:
            return (
                1,
                "",
                "OCI runtime create failed: nvidia-persistenced: no such file or directory",
            )
        return (0, "cpu_container_id", "")

    with patch.object(t_mod, "_run_docker", mock_run_docker):
        await t_mod._start_container()  # should not raise

    assert any("--gpus" in c for c in calls), "should have tried with GPU first"
    assert any("--gpus" not in c for c in calls), "should have retried without GPU"


# ── _ensure_server_running ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_server_skips_start_if_already_healthy():
    """If /health returns 200, docker is never called."""
    import channels.transcribe as t_mod

    with (
        patch.object(t_mod, "_check_server", AsyncMock(return_value=True)),
        patch.object(t_mod, "_start_container", AsyncMock()) as mock_start,
    ):
        await t_mod._ensure_server_running()
        mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_server_starts_container_then_polls():
    """Starts container, polls health, returns once healthy."""
    import channels.transcribe as t_mod

    health_calls = [False, False, True]  # fails twice, then ready

    with (
        patch.object(t_mod, "_check_server", AsyncMock(side_effect=health_calls)),
        patch.object(t_mod, "_start_container", AsyncMock()),
        patch("asyncio.sleep", AsyncMock()),
    ):
        await t_mod._ensure_server_running()  # should not raise


@pytest.mark.asyncio
async def test_ensure_server_raises_on_timeout():
    """RuntimeError if server never becomes healthy within timeout."""
    import channels.transcribe as t_mod

    with (
        patch.object(t_mod, "_check_server", AsyncMock(return_value=False)),
        patch.object(t_mod, "_start_container", AsyncMock()),
        patch("asyncio.sleep", AsyncMock()),
        patch("channels.transcribe._STARTUP_TIMEOUT", 0),
    ):
        with pytest.raises(RuntimeError, match="did not become healthy"):
            await t_mod._ensure_server_running()


# ── transcribe (full flow) ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transcribe_starts_server_and_returns_text(tmp_path):
    """transcribe() ensures server is running, posts WAV, returns transcript."""
    import channels.transcribe as t_mod

    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"transcript": "hello world"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch.object(t_mod, "_to_wav", return_value=wav),
        patch.object(t_mod, "_ensure_server_running", AsyncMock()),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await t_mod.transcribe(wav)

    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_server_error_in_json(tmp_path):
    """RuntimeError raised when server returns {"error": ...}."""
    import channels.transcribe as t_mod

    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"error": "model crashed"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch.object(t_mod, "_to_wav", return_value=wav),
        patch.object(t_mod, "_ensure_server_running", AsyncMock()),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        with pytest.raises(RuntimeError, match="Parakeet server error"):
            await t_mod.transcribe(wav)


@pytest.mark.asyncio
async def test_transcribe_cleans_up_converted_wav(tmp_path):
    """Converted WAV is deleted after transcription."""
    import channels.transcribe as t_mod

    ogg = tmp_path / "voice.ogg"
    ogg.write_bytes(b"OggS")
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"transcript": "ok"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch.object(t_mod, "_to_wav", return_value=wav),
        patch.object(t_mod, "_ensure_server_running", AsyncMock()),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        await t_mod.transcribe(ogg)

    assert not wav.exists()


# ── Telegram bot voice handler ─────────────────────────────────────────────────


def make_tg_bot():
    with patch("channels.telegram.bot.Application") as mock_app_cls:
        mock_app = MagicMock()
        mock_app_cls.builder.return_value.token.return_value.build.return_value = (
            mock_app
        )

        from channels.telegram.bot import TelegramBot

        bot = TelegramBot.__new__(TelegramBot)
        bot.brain = MagicMock()
        bot.brain.think = AsyncMock(return_value=("Got it!", []))
        bot.brain.take_files = MagicMock(return_value=[])
        bot.app = mock_app
        bot._allowed_users = set()
        bot._history = MagicMock()
        bot._history.load.return_value = []
        return bot


def make_voice_update(file_unique_id="abc123", caption=""):
    update = MagicMock()
    update.effective_user.id = 1
    update.effective_user.username = "testuser"
    update.message.voice = MagicMock()
    update.message.voice.file_unique_id = file_unique_id
    update.message.audio = None
    update.message.caption = caption
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    return update


@pytest.mark.asyncio
async def test_handle_voice_transcribes_and_replies(tmp_path):
    bot = make_tg_bot()
    update = make_voice_update()

    tg_file = AsyncMock()
    tg_file.download_to_drive = AsyncMock()
    update.message.voice.get_file = AsyncMock(return_value=tg_file)

    with (
        patch("channels.telegram.bot._UPLOAD_DIR", tmp_path),
        patch(
            "channels.transcribe.transcribe",
            new=AsyncMock(return_value="remind me tomorrow"),
        ),
        patch("channels.telegram.bot.TelegramBot._send_file", new=AsyncMock()),
    ):
        await bot._handle_voice(update, MagicMock())

    bot.brain.think.assert_called_once()
    assert "remind me tomorrow" in bot.brain.think.call_args.kwargs["user_message"]
    update.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_handle_voice_fallback_when_docker_unavailable(tmp_path):
    bot = make_tg_bot()
    update = make_voice_update()

    tg_file = AsyncMock()
    tg_file.download_to_drive = AsyncMock()
    update.message.voice.get_file = AsyncMock(return_value=tg_file)

    async def fail_transcribe(_):
        raise RuntimeError("Docker is not installed")

    with (
        patch("channels.telegram.bot._UPLOAD_DIR", tmp_path),
        patch("channels.transcribe.transcribe", new=fail_transcribe),
        patch("channels.telegram.bot.TelegramBot._send_file", new=AsyncMock()),
    ):
        await bot._handle_voice(update, MagicMock())

    bot.brain.think.assert_called_once()
    msg = bot.brain.think.call_args.kwargs["user_message"]
    assert "voice" in msg.lower() or "audio" in msg.lower()


@pytest.mark.asyncio
async def test_handle_voice_caption_appended(tmp_path):
    bot = make_tg_bot()
    update = make_voice_update(caption="this is urgent")

    tg_file = AsyncMock()
    tg_file.download_to_drive = AsyncMock()
    update.message.voice.get_file = AsyncMock(return_value=tg_file)

    with (
        patch("channels.telegram.bot._UPLOAD_DIR", tmp_path),
        patch(
            "channels.transcribe.transcribe", new=AsyncMock(return_value="call John")
        ),
        patch("channels.telegram.bot.TelegramBot._send_file", new=AsyncMock()),
    ):
        await bot._handle_voice(update, MagicMock())

    msg = bot.brain.think.call_args.kwargs["user_message"]
    assert "call John" in msg
    assert "this is urgent" in msg


@pytest.mark.asyncio
async def test_handle_voice_unauthorized():
    bot = make_tg_bot()
    bot._allowed_users = {999}
    update = make_voice_update()
    await bot._handle_voice(update, MagicMock())
    update.message.reply_text.assert_called_once_with("⛔ Unauthorized.")
    bot.brain.think.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Telegram bot photo handler
# ══════════════════════════════════════════════════════════════════════════════


def make_photo_update(caption=""):
    """Build a minimal mock Update with a photo message."""
    update = MagicMock()
    update.effective_user.id = 1
    update.effective_user.username = "testuser"

    photo = MagicMock()
    photo.file_unique_id = "photo123"
    update.message.photo = [photo]  # handler uses photo[-1]
    update.message.caption = caption
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    update.message.reply_to_message = None
    return update


@pytest.mark.asyncio
async def test_handle_photo_sends_multimodal_content_to_brain():
    """Photo is downloaded, base64-encoded, and passed as an image block to brain."""
    bot = make_tg_bot()
    update = make_photo_update()

    tg_file = AsyncMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake jpeg data"))
    update.message.photo[-1].get_file = AsyncMock(return_value=tg_file)

    bot._stream_think = AsyncMock(return_value=("Here's the image!", []))

    await bot._handle_photo(update, MagicMock())

    bot._stream_think.assert_called_once()
    _, content, _ = bot._stream_think.call_args.args
    image_blocks = [b for b in content if b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["media_type"] == "image/jpeg"
    assert image_blocks[0]["data"]  # base64 data is non-empty


@pytest.mark.asyncio
async def test_handle_photo_includes_caption():
    """Caption text is included as a text block before the image."""
    bot = make_tg_bot()
    update = make_photo_update(caption="What is in this image?")

    tg_file = AsyncMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake jpeg"))
    update.message.photo[-1].get_file = AsyncMock(return_value=tg_file)

    bot._stream_think = AsyncMock(return_value=("It shows a cat!", []))

    await bot._handle_photo(update, MagicMock())

    _, content, _ = bot._stream_think.call_args.args
    text_blocks = [b for b in content if b.get("type") == "text"]
    assert any("What is in this image?" in b["text"] for b in text_blocks)


@pytest.mark.asyncio
async def test_handle_photo_no_caption_still_works():
    """Photo without caption still calls brain with just the image block."""
    bot = make_tg_bot()
    update = make_photo_update(caption="")

    tg_file = AsyncMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake jpeg"))
    update.message.photo[-1].get_file = AsyncMock(return_value=tg_file)

    bot._stream_think = AsyncMock(return_value=("I see an image.", []))

    await bot._handle_photo(update, MagicMock())

    bot._stream_think.assert_called_once()
    _, content, _ = bot._stream_think.call_args.args
    assert any(b.get("type") == "image" for b in content)


@pytest.mark.asyncio
async def test_handle_photo_unauthorized():
    """Unauthorized users are rejected and brain is never called."""
    bot = make_tg_bot()
    bot._allowed_users = {999}
    update = make_photo_update()

    await bot._handle_photo(update, MagicMock())

    update.message.reply_text.assert_called_once_with("⛔ Unauthorized.")
    bot.brain.think.assert_not_called()


@pytest.mark.asyncio
async def test_handle_photo_download_error():
    """Error during photo download sends error message to user without calling brain."""
    bot = make_tg_bot()
    update = make_photo_update()

    tg_file = AsyncMock()
    tg_file.download_as_bytearray = AsyncMock(side_effect=Exception("network error"))
    update.message.photo[-1].get_file = AsyncMock(return_value=tg_file)

    bot._stream_think = AsyncMock(return_value=("", []))

    await bot._handle_photo(update, MagicMock())

    bot._stream_think.assert_not_called()
    update.message.reply_text.assert_called_once()
    assert "⚠️" in update.message.reply_text.call_args.args[0]


# ══════════════════════════════════════════════════════════════════════════════
# Discord audio handling
# ══════════════════════════════════════════════════════════════════════════════


def make_discord_bot():
    """Create a DiscordBot with mocked internals."""
    with patch("channels.discord.bot.discord.Client"):
        from channels.discord.bot import DiscordBot

        bot = DiscordBot.__new__(DiscordBot)
        bot.brain = MagicMock()
        bot.brain.think = AsyncMock(return_value=("Got it!", []))
        bot.brain.take_files = MagicMock(return_value=[])
        bot._allowed_users = set()
        bot._history = MagicMock()
        bot._history.load.return_value = []
        bot._client = MagicMock()
        return bot


def make_discord_audio_message(filename="voice.ogg", content_type="audio/ogg", text=""):
    """Create a mock Discord message with an audio attachment."""
    message = MagicMock()
    message.author.id = 1
    message.author.__str__ = lambda s: "testuser"
    message.content = text
    message.reply = AsyncMock()
    message.channel.typing = MagicMock(return_value=AsyncMock())
    message.channel.typing.return_value.__aenter__ = AsyncMock()
    message.channel.typing.return_value.__aexit__ = AsyncMock()

    att = MagicMock()
    att.filename = filename
    att.id = 12345
    att.url = "https://cdn.discord.test/voice.ogg"
    att.content_type = content_type
    return message, [att]


def _mock_aiohttp_session(audio_bytes=b"fake audio data"):
    """Build a properly nested aiohttp.ClientSession mock for async-with usage."""
    mock_resp = MagicMock()
    mock_resp.read = AsyncMock(return_value=audio_bytes)

    mock_get_ctx = MagicMock()
    mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_get_ctx
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_session)
    return mock_cls


@pytest.mark.asyncio
async def test_discord_handle_audio_transcribes(tmp_path):
    """Discord audio attachment is downloaded, transcribed, and sent to brain."""
    bot = make_discord_bot()
    message, attachments = make_discord_audio_message()

    with (
        patch("channels.discord.bot._UPLOAD_DIR", tmp_path),
        patch("channels.discord.bot.aiohttp.ClientSession", _mock_aiohttp_session()),
        patch(
            "channels.transcribe.transcribe",
            new=AsyncMock(return_value="schedule meeting"),
        ),
    ):
        await bot._handle_audio(message, attachments, "")

    bot.brain.think.assert_called_once()
    assert "schedule meeting" in bot.brain.think.call_args.kwargs["user_message"]


@pytest.mark.asyncio
async def test_discord_handle_audio_fallback(tmp_path):
    """Discord audio falls back gracefully when transcription unavailable."""
    bot = make_discord_bot()
    message, attachments = make_discord_audio_message()

    async def fail_transcribe(_):
        raise RuntimeError("Docker is not installed")

    with (
        patch("channels.discord.bot._UPLOAD_DIR", tmp_path),
        patch("channels.discord.bot.aiohttp.ClientSession", _mock_aiohttp_session()),
        patch("channels.transcribe.transcribe", new=fail_transcribe),
    ):
        await bot._handle_audio(message, attachments, "")

    bot.brain.think.assert_called_once()
    msg = bot.brain.think.call_args.kwargs["user_message"]
    assert "voice" in msg.lower() or "audio" in msg.lower()


@pytest.mark.asyncio
async def test_discord_handle_audio_with_text(tmp_path):
    """Discord audio with accompanying text includes both in user_message."""
    bot = make_discord_bot()
    message, attachments = make_discord_audio_message(text="check this")

    with (
        patch("channels.discord.bot._UPLOAD_DIR", tmp_path),
        patch("channels.discord.bot.aiohttp.ClientSession", _mock_aiohttp_session()),
        patch(
            "channels.transcribe.transcribe", new=AsyncMock(return_value="hello world")
        ),
    ):
        await bot._handle_audio(message, attachments, "check this")

    msg = bot.brain.think.call_args.kwargs["user_message"]
    assert "hello world" in msg
    assert "check this" in msg


# ══════════════════════════════════════════════════════════════════════════════
# Web audio handling
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_web_handle_audio_transcribes(tmp_path):
    """Web audio message is decoded, transcribed, and response sent via WS."""
    import base64
    from channels.web.bot import WebBot

    bot = WebBot.__new__(WebBot)
    bot.brain = MagicMock()
    bot.brain.think = AsyncMock(return_value=("Got it!", []))
    bot.brain.take_files = MagicMock(return_value=[])
    bot._history = MagicMock()
    bot._history.load.return_value = []

    ws = AsyncMock()
    audio_b64 = base64.b64encode(b"fake audio").decode()
    data = {"type": "audio", "data": audio_b64, "filename": "recording.webm"}

    with (
        patch("channels.web.bot._UPLOAD_DIR", tmp_path),
        patch(
            "channels.transcribe.transcribe",
            new=AsyncMock(return_value="buy groceries"),
        ),
    ):
        await bot._handle_audio_ws(ws, "session123", [], data)

    bot.brain.think.assert_called_once()
    assert "buy groceries" in bot.brain.think.call_args.kwargs["user_message"]
    # Should send transcript + response
    calls = [c.args[0] for c in ws.send_json.call_args_list]
    assert any(c.get("type") == "transcript" for c in calls)
    assert any(c.get("type") == "text" for c in calls)


@pytest.mark.asyncio
async def test_web_handle_audio_fallback(tmp_path):
    """Web audio falls back gracefully when transcription unavailable."""
    import base64
    from channels.web.bot import WebBot

    bot = WebBot.__new__(WebBot)
    bot.brain = MagicMock()
    bot.brain.think = AsyncMock(return_value=("Got it!", []))
    bot.brain.take_files = MagicMock(return_value=[])
    bot._history = MagicMock()
    bot._history.load.return_value = []

    ws = AsyncMock()
    audio_b64 = base64.b64encode(b"fake audio").decode()
    data = {"type": "audio", "data": audio_b64, "filename": "recording.webm"}

    async def fail_transcribe(_):
        raise RuntimeError("Docker is not installed")

    with (
        patch("channels.web.bot._UPLOAD_DIR", tmp_path),
        patch("channels.transcribe.transcribe", new=fail_transcribe),
    ):
        await bot._handle_audio_ws(ws, "session123", [], data)

    bot.brain.think.assert_called_once()
    msg = bot.brain.think.call_args.kwargs["user_message"]
    assert "voice" in msg.lower() or "audio" in msg.lower()


@pytest.mark.asyncio
async def test_web_handle_audio_empty_data(tmp_path):
    """Web audio with empty data returns error."""
    from channels.web.bot import WebBot

    bot = WebBot.__new__(WebBot)
    bot.brain = MagicMock()
    bot._history = MagicMock()

    ws = AsyncMock()
    data = {"type": "audio", "data": "", "filename": "recording.webm"}

    await bot._handle_audio_ws(ws, "session123", [], data)

    bot.brain.think.assert_not_called()
    ws.send_json.assert_called_once()
    assert ws.send_json.call_args.args[0]["type"] == "error"


# ══════════════════════════════════════════════════════════════════════════════
# CLI /voice command
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cli_voice_command_transcribes(tmp_path):
    """CLI /voice <path> transcribes and sends transcript to brain."""
    from channels.cli.bot import CLIBot

    bot = CLIBot.__new__(CLIBot)
    bot.brain = MagicMock()
    bot.brain.think = AsyncMock(return_value=("Sure!", []))
    bot.brain.take_files = MagicMock(return_value=[])
    bot._history_store = MagicMock()
    bot._history_store.load.return_value = []
    bot._running = True

    audio_file = tmp_path / "test.ogg"
    audio_file.write_bytes(b"fake audio")

    inputs = iter([f"/voice {audio_file}", "/quit"])

    with (
        patch(
            "channels.transcribe.transcribe", new=AsyncMock(return_value="hello there")
        ),
        patch.object(bot, "_read_input", side_effect=inputs),
        patch("builtins.print"),
    ):
        await bot.start()

    bot.brain.think.assert_called_once()
    assert "hello there" in bot.brain.think.call_args.kwargs["user_message"]


@pytest.mark.asyncio
async def test_cli_voice_command_file_not_found(tmp_path):
    """CLI /voice with nonexistent file prints error and continues."""
    from channels.cli.bot import CLIBot

    bot = CLIBot.__new__(CLIBot)
    bot.brain = MagicMock()
    bot._history_store = MagicMock()
    bot._running = True

    inputs = iter(["/voice /nonexistent/audio.ogg", "/quit"])
    printed = []

    with (
        patch.object(bot, "_read_input", side_effect=inputs),
        patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a))),
    ):
        await bot.start()

    bot.brain.think.assert_not_called()
    assert any("not found" in p.lower() or "File not found" in p for p in printed)
