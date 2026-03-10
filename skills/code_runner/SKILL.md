# code_runner

Execute a Python code snippet in an isolated subprocess and return the output.

## When to use
- When the user asks you to run, test, or evaluate Python code
- When you need to compute something non-trivial (math, data processing, string manipulation)
- When you want to verify that a code snippet works before sharing it

## Input
- `code` (string, required): The Python code to execute
- `timeout` (integer, optional): Timeout in seconds (default: 15, max: 60)

## Output
Returns stdout + stderr. If the code raises an exception, the traceback is returned.
Returns an error message on timeout.

## Safety
- Code runs in a subprocess — it cannot crash the agent process
- Has a configurable timeout to prevent infinite loops
- Inherits the current environment (no additional sandboxing in Phase 3; Phase 6 adds Docker)
- stdin is closed to prevent interactive prompts from hanging
