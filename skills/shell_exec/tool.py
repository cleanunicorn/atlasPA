"""
skills/shell_exec/tool.py

Run shell commands with a timeout.
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

    try:

        def _run_sync():
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env={**os.environ},
                    timeout=timeout,
                )
                return result.stdout, result.returncode, False
            except subprocess.TimeoutExpired as exc:
                return (exc.output or b""), None, True

        raw_stdout, returncode, timed_out = await asyncio.to_thread(_run_sync)

        if timed_out:
            return f"Error: command timed out after {timeout}s\n$ {command}"

        output = raw_stdout.decode(errors="replace").rstrip()

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
