"""
skills/shell_exec/tool.py

Run shell commands in a thread-pool executor to avoid asyncio event loop
conflicts when called alongside other async tools (browser, HTTP, etc.).
Output (stdout + stderr combined) is returned as a string.
"""

import asyncio
import os
import subprocess
from pathlib import Path

PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to run",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default: 30, max: 120)",
            "default": 30,
        },
        "working_dir": {
            "type": "string",
            "description": "Directory to run the command in (default: user home directory)",
        },
    },
    "required": ["command"],
}

MAX_TIMEOUT = 120
MAX_OUTPUT_CHARS = 8000  # Truncate very long output so it fits in context


async def run(command: str, timeout: int = 30, working_dir: str = "", **kwargs) -> str:
    """
    Execute a shell command and return combined stdout + stderr.

    The command runs inside a thread-pool executor so it never competes with
    the running event loop (avoiding "Cannot run the event loop while another
    loop is running" errors when other async tools are active).

    Args:
        command:     Shell command string.
        timeout:     Max seconds to wait (capped at MAX_TIMEOUT).
        working_dir: Directory to run in. Defaults to user's home directory.

    Returns:
        Command output as a string. Never raises.
    """
    timeout = min(int(timeout), MAX_TIMEOUT)

    cwd = Path(working_dir).expanduser() if working_dir else Path.home()
    if not cwd.exists():
        return f"Error: working directory does not exist: {cwd}"

    def _run_blocking() -> tuple[bytes, int | None, bool]:
        """Run the subprocess synchronously; returns (output, returncode, timed_out)."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                cwd=cwd,
                env={**os.environ},
                timeout=timeout,
            )
            return proc.stdout, proc.returncode, False
        except subprocess.TimeoutExpired as exc:
            # exc.stdout may contain partial output captured before the kill
            return exc.stdout or b"", None, True

    try:
        loop = asyncio.get_event_loop()
        stdout, returncode, timed_out = await loop.run_in_executor(None, _run_blocking)

        if timed_out:
            return f"Error: command timed out after {timeout}s\n$ {command}"

        output = stdout.decode(errors="replace").rstrip()

        if not output:
            output = f"(command exited with code {returncode}, no output)"
        elif returncode != 0:
            output = f"[exit code {returncode}]\n{output}"

        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = (
                output[:half]
                + f"\n\n... [{len(output) - MAX_OUTPUT_CHARS} chars truncated] ...\n\n"
                + output[-half:]
            )

        return output

    except Exception as e:
        return f"Error running command: {e}"
