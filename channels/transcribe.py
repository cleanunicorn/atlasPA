"""
channels/transcribe.py

Voice transcription via the Parakeet HTTP microservice.
Shared across all channel adapters (Telegram, Discord, Web, CLI).

Atlas starts the Docker container automatically on the first voice message and
waits for the model to load. The container stays running for subsequent messages.

── Config (config/.env) ────────────────────────────────────────────────────────

    PARAKEET_URL=http://localhost:8765      # default
    PARAKEET_IMAGE=nvcr.io/nvidia/nemo:25.11.01  # default

── Audio conversion ─────────────────────────────────────────────────────────────

Telegram voice notes arrive as OGG/Opus. The server expects 16 kHz mono WAV.
Conversion uses pydub + system ffmpeg.

    sudo apt install ffmpeg   # or: brew install ffmpeg
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_PARAKEET_URL   = os.getenv("PARAKEET_URL",   "http://localhost:8765")
_PARAKEET_IMAGE = os.getenv("PARAKEET_IMAGE", "nvcr.io/nvidia/nemo:25.11.01")
_CONTAINER_NAME = "atlas_parakeet"
_TRANSCRIBE_SERVER_DIR = (Path(__file__).parent.parent.parent / "transcribe_server").resolve()

_parsed_url = urlparse(_PARAKEET_URL)
_PARAKEET_PORT = _parsed_url.port or 8765

_HTTP_TIMEOUT    = 120   # seconds per transcription request
_STARTUP_TIMEOUT = 180   # seconds to wait for model to load on first start


# ── Audio conversion ──────────────────────────────────────────────────────────

def _to_wav(src: Path) -> Path:
    """
    Convert any audio file to 16 kHz mono WAV required by Parakeet.
    Returns the WAV path (equals src if already .wav).
    Raises RuntimeError if pydub/ffmpeg are unavailable.
    """
    if src.suffix.lower() == ".wav":
        return src

    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "pydub is required for audio conversion.\n"
            "Install with:  uv add pydub\n"
            "You also need ffmpeg:  sudo apt install ffmpeg"
        )

    wav_path = src.with_suffix(".wav")
    audio = AudioSegment.from_file(str(src))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(str(wav_path), format="wav")
    return wav_path


# ── Docker management ─────────────────────────────────────────────────────────

async def _check_server() -> bool:
    """Return True if the Parakeet HTTP server is reachable and healthy."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{_PARAKEET_URL}/health")
            return r.status_code == 200
    except Exception:
        return False


_GPU_ERROR_HINTS = ("nvidia", "runc", "OCI runtime", "nvidia-persistenced", "no such file or directory")


async def _run_docker(cmd: list[str]) -> tuple[int, str, str]:
    """Run a docker command and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        raise RuntimeError(
            "Docker is not installed or not in PATH.\n"
            "Install Docker to enable voice transcription."
        )
    return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


async def _start_container() -> None:
    """
    Launch the Parakeet Docker container in detached mode.
    Tries with --gpus all first; if the GPU runtime is unavailable, retries
    without GPU (CPU-only, slower but functional).
    Silently continues if the container is already starting (name conflict).
    Raises RuntimeError if Docker is not available or the run command fails.
    """
    base_cmd = [
        "docker", "run",
        "--rm", "--detach",
        "--shm-size=8g",
        f"--volume={_TRANSCRIBE_SERVER_DIR}:/transcribe_server:ro",
        f"--publish={_PARAKEET_PORT}:{_PARAKEET_PORT}",
        f"--name={_CONTAINER_NAME}",
        _PARAKEET_IMAGE,
        "python", "/transcribe_server/server.py",
    ]

    # Try with GPU first
    gpu_cmd = ["docker", "run", "--gpus", "all"] + base_cmd[2:]
    returncode, stdout, stderr = await _run_docker(gpu_cmd)

    if returncode != 0 and returncode != 125 and "already in use" not in stderr:
        # Check if the failure is GPU-related → retry without GPU
        if any(hint in stderr for hint in _GPU_ERROR_HINTS):
            logger.warning(
                "GPU runtime unavailable (%s) — retrying without --gpus (CPU mode, slower).",
                stderr.split("\n")[0],
            )
            returncode, stdout, stderr = await _run_docker(base_cmd)

    if returncode == 0:
        container_id = stdout[:12]
        logger.info(f"Started Parakeet container {container_id}")
    elif returncode == 125 or "already in use" in stderr:
        logger.info("Parakeet container already exists — waiting for health check…")
    else:
        raise RuntimeError(
            f"docker run failed (exit {returncode}):\n{stderr}\n\n"
            f"Make sure the image is pulled:  docker pull {_PARAKEET_IMAGE}"
        )


async def _ensure_server_running() -> None:
    """
    Ensure the Parakeet server is up and healthy.
    Starts the Docker container if needed and waits for the model to load.
    Raises RuntimeError if the server does not become healthy within the timeout.
    """
    if await _check_server():
        return

    logger.info(
        f"Parakeet server not running — starting Docker container "
        f"({_PARAKEET_IMAGE})…"
    )
    await _start_container()

    # Poll health — model loading takes ~30–90 s on first start
    logger.info(f"Waiting up to {_STARTUP_TIMEOUT}s for model to load…")
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        await asyncio.sleep(3)
        if await _check_server():
            logger.info("Parakeet server is ready.")
            return

    raise RuntimeError(
        f"Parakeet container started but did not become healthy within "
        f"{_STARTUP_TIMEOUT}s.\n"
        f"Check logs with:  docker logs {_CONTAINER_NAME}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def transcribe(audio_path: Path) -> str:
    """
    Transcribe an audio file using the Parakeet microservice.

    Starts the Docker container automatically on first call.
    Raises RuntimeError if Docker is unavailable or the server fails to start.
    """
    wav_path = await asyncio.to_thread(_to_wav, audio_path)
    try:
        await _ensure_server_running()

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            with open(wav_path, "rb") as f:
                response = await client.post(
                    f"{_PARAKEET_URL}/transcribe",
                    files={"file": (wav_path.name, f, "audio/wav")},
                )
            response.raise_for_status()
            data = response.json()

        if "error" in data:
            raise RuntimeError(f"Parakeet server error: {data['error']}")

        return data.get("transcript", "")

    finally:
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)
