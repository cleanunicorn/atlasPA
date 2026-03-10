"""
skills/code_runner/tool.py

Execute Python code snippets in an isolated subprocess.
stdout + stderr are captured and returned as a string.
"""

import asyncio
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
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,  # No interactive prompts
        )

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"Error: code timed out after {timeout}s"

        output = stdout.decode(errors="replace").rstrip()

        if not output:
            output = f"(no output, exit code {proc.returncode})"
        elif proc.returncode != 0:
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
