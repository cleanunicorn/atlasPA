# shell_exec

Run a shell command and return its output. Commands run with a timeout (default 30s).

## When to use
- When the user asks you to run a command, script, or system task
- When you need to check system state (disk usage, processes, git status, etc.)
- When you need to manipulate files or directories beyond what file_ops supports

## Input
- `command` (string, required): The shell command to run
- `timeout` (integer, optional): Timeout in seconds (default: 30, max: 120)
- `working_dir` (string, optional): Directory to run in (default: current directory)

## Output
Returns stdout + stderr combined. On timeout or error, returns a descriptive error message.

## Safety
- Has a configurable timeout to prevent runaway processes
- Working directory defaults to the project root, not system root
- Phase 6 will add an explicit approval workflow for destructive commands
