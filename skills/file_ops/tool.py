"""
skills/file_ops/tool.py

Read, write, and list files. Sandboxed to ~/agent-files/.
"""

from pathlib import Path

from paths import DATA_DIR

PARAMETERS = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["read", "write", "list", "delete"],
            "description": "The file operation to perform",
        },
        "path": {
            "type": "string",
            "description": "File path relative to ~/agent-files/ (e.g. 'notes/todo.md')",
        },
        "content": {
            "type": "string",
            "description": "Content to write (only for 'write' operation)",
        },
    },
    "required": ["operation"],
}

# All file access is sandboxed to this directory
SANDBOX_DIR = DATA_DIR


def _safe_path(relative_path: str) -> Path | None:
    """
    Resolve a relative path inside the sandbox. Returns None if path escapes sandbox.
    """
    sandbox = SANDBOX_DIR.resolve()
    target = (sandbox / relative_path).resolve()
    # Ensure the resolved path is inside the sandbox
    try:
        target.relative_to(sandbox)
        return target
    except ValueError:
        return None


def run(operation: str, path: str = "", content: str = "", **kwargs) -> str:
    """
    Perform a file operation inside ~/agent-files/.

    Args:
        operation: "read", "write", "list", or "delete"
        path:      Relative path inside sandbox (required for read/write/delete)
        content:   Content to write (required for write)

    Returns:
        Result string. Never raises — errors returned as strings.
    """
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    if operation == "list":
        list_dir = SANDBOX_DIR
        if path:
            resolved = _safe_path(path)
            if resolved is None:
                return "Error: path escapes sandbox"
            list_dir = resolved

        if not list_dir.exists():
            return f"Directory does not exist: {list_dir}"

        entries = sorted(list_dir.iterdir())
        if not entries:
            return f"Directory is empty: {list_dir.relative_to(SANDBOX_DIR.parent)}"

        lines = [f"Contents of {list_dir.relative_to(SANDBOX_DIR.parent)}:"]
        for entry in entries:
            prefix = "📁 " if entry.is_dir() else "📄 "
            lines.append(f"  {prefix}{entry.name}")
        return "\n".join(lines)

    if not path:
        return f"Error: 'path' is required for operation '{operation}'"

    target = _safe_path(path)
    if target is None:
        return "Error: path escapes the ~/agent-files/ sandbox"

    if operation == "read":
        if not target.exists():
            return f"File not found: {path}"
        try:
            return target.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    elif operation == "write":
        if not content and content != "":
            return "Error: 'content' is required for write operation"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"✅ Written to {path} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing file: {e}"

    elif operation == "delete":
        if not target.exists():
            return f"File not found: {path}"
        try:
            target.unlink()
            return f"✅ Deleted: {path}"
        except Exception as e:
            return f"Error deleting file: {e}"

    else:
        return f"Unknown operation: '{operation}'. Use: read, write, list, delete"
