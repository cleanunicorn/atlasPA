"""
skills/code_runner/tool.py

Execute Python code snippets in an isolated subprocess.
stdout + stderr are captured and returned as a string.
"""

import asyncio
import subprocess
import sys

PARAMETERS = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "The Python code to execute",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default: 15, max: 60)",
            "default": 15,
        },
    },
    "required": ["code"],
}

MAX_TIMEOUT = 60
MAX_OUTPUT_CHARS = 8000


async def run(code: str, timeout: int = 15, **kwargs) -> str:
    """
    Run Python code in a subprocess and return stdout + stderr.

    Args:
        code:    Python source code to execute.
        timeout: Max seconds before killing the process.

    Returns:
        Combined stdout/stderr output. Never raises.
    """
    timeout = min(int(timeout), MAX_TIMEOUT)

    try:

        def _run_sync():
            try:
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    timeout=timeout,
                )
                return result.stdout, result.returncode, False
            except subprocess.TimeoutExpired as exc:
                return (exc.output or b""), None, True

        raw_stdout, returncode, timed_out = await asyncio.to_thread(_run_sync)

        if timed_out:
            return f"Error: code timed out after {timeout}s"

        output = raw_stdout.decode(errors="replace").rstrip()

        if not output:
            output = f"(no output, exit code {returncode})"
        elif returncode != 0:
            # Errors are useful — don't suppress them
            pass  # Output already contains the traceback from stderr

        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = (
                output[:half]
                + f"\n\n... [{len(output) - MAX_OUTPUT_CHARS} chars truncated] ...\n\n"
                + output[-half:]
            )

        return output

    except Exception as e:
        return f"Error running code: {e}"
