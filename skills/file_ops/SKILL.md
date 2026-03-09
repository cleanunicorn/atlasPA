# file_ops

Read, write, and list files. Sandboxed to ~/agent-files/ for safety.

## When to use
- When the user asks you to save, read, or list files
- When you need to persist information to disk
- When the user wants to create a note, document, or code file

## Input
- `operation` (string, required): One of: "read", "write", "list", "delete"
- `path` (string, required for read/write/delete): File path relative to ~/agent-files/
- `content` (string, required for write): Content to write to the file

## Output
Returns the file content (for read), a confirmation message (for write/delete),
or a directory listing (for list).

## Security
All file access is sandboxed to ~/agent-files/. Path traversal attempts are blocked.
