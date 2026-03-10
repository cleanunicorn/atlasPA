"""
tests/test_voice.py

Tests for voice message transcription (HTTP client + Docker auto-start)
and Telegram voice handler.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── _to_wav ────────────────────────────────────────────────────────────────────

def test_to_wav_returns_same_path_for_wav(tmp_path):
    """WAV files are returned unchanged."""
    from channels.telegram.transcribe import _to_wav
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)
    assert _to_wav(wav) == wav


def test_to_wav_no_pydub_raises(tmp_path):
    """RuntimeError with install instructions when pydub is missing."""
    import sys
    ogg = tmp_path / "voice.ogg"
    ogg.write_bytes(b"OggS" + b"\x00" * 50)

    with patch.dict(sys.modules, {"pydub": None}):
        import channels.telegram.transcribe as t_mod
        with pytest.raises(RuntimeError, match="pydub"):
            t_mod._to_wav(ogg)


def test_to_wav_converts_ogg(tmp_path):
    """OGG file is converted to WAV via pydub."""
    import channels.telegram.transcribe as t_mod

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
    import channels.telegram.transcribe as t_mod

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
    import channels.telegram.transcribe as t_mod

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
    import channels.telegram.transcribe as t_mod

    with patch.object(t_mod, "_run_docker", AsyncMock(return_value=(0, "abc123def456", ""))):
        await t_mod._start_container()  # should not raise


@pytest.mark.asyncio
async def test_start_container_name_conflict_is_ok():
    """docker run exit 125 (name conflict) → no exception, just a log."""
    import channels.telegram.transcribe as t_mod

    with patch.object(t_mod, "_run_docker", AsyncMock(return_value=(125, "", "already in use"))):
        await t_mod._start_container()  # should not raise


@pytest.mark.asyncio
async def test_start_container_docker_not_found():
    """FileNotFoundError from _run_docker → RuntimeError with clear message."""
    import channels.telegram.transcribe as t_mod

    async def raise_fnf(_cmd):
        raise RuntimeError("Docker is not installed or not in PATH.")

    with patch.object(t_mod, "_run_docker", raise_fnf):
        with pytest.raises(RuntimeError, match="Docker is not installed"):
            await t_mod._start_container()


@pytest.mark.asyncio
async def test_start_container_docker_run_fails():
    """Non-GPU, non-125 exit code → RuntimeError with stderr."""
    import channels.telegram.transcribe as t_mod

    with patch.object(t_mod, "_run_docker", AsyncMock(return_value=(1, "", "image not found"))):
        with pytest.raises(RuntimeError, match="docker run failed"):
            await t_mod._start_container()


@pytest.mark.asyncio
async def test_start_container_gpu_error_retries_cpu():
    """GPU runtime error → retries without --gpus, succeeds on CPU."""
    import channels.telegram.transcribe as t_mod

    calls = []

    async def mock_run_docker(cmd):
        calls.append(cmd)
        if "--gpus" in cmd:
            return (1, "", "OCI runtime create failed: nvidia-persistenced: no such file or directory")
        return (0, "cpu_container_id", "")

    with patch.object(t_mod, "_run_docker", mock_run_docker):
        await t_mod._start_container()  # should not raise

    assert any("--gpus" in c for c in calls), "should have tried with GPU first"
    assert any("--gpus" not in c for c in calls), "should have retried without GPU"


# ── _ensure_server_running ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ensure_server_skips_start_if_already_healthy():
    """If /health returns 200, docker is never called."""
    import channels.telegram.transcribe as t_mod

    with (
        patch.object(t_mod, "_check_server", AsyncMock(return_value=True)),
        patch.object(t_mod, "_start_container", AsyncMock()) as mock_start,
    ):
        await t_mod._ensure_server_running()
        mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_server_starts_container_then_polls():
    """Starts container, polls health, returns once healthy."""
    import channels.telegram.transcribe as t_mod

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
    import channels.telegram.transcribe as t_mod

    with (
        patch.object(t_mod, "_check_server", AsyncMock(return_value=False)),
        patch.object(t_mod, "_start_container", AsyncMock()),
        patch("asyncio.sleep", AsyncMock()),
        patch("channels.telegram.transcribe._STARTUP_TIMEOUT", 0),
    ):
        with pytest.raises(RuntimeError, match="did not become healthy"):
            await t_mod._ensure_server_running()


# ── transcribe (full flow) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transcribe_starts_server_and_returns_text(tmp_path):
    """transcribe() ensures server is running, posts WAV, returns transcript."""
    import channels.telegram.transcribe as t_mod

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
    import channels.telegram.transcribe as t_mod

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
    import channels.telegram.transcribe as t_mod

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
        mock_app_cls.builder.return_value.token.return_value.build.return_value = mock_app

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
        patch("channels.telegram.transcribe.transcribe", new=AsyncMock(return_value="remind me tomorrow")),
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
        patch("channels.telegram.transcribe.transcribe", new=fail_transcribe),
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
        patch("channels.telegram.transcribe.transcribe", new=AsyncMock(return_value="call John")),
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
