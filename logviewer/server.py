"""
logviewer/server.py

Web UI for browsing LLM observability logs (logs/llm.jsonl).
Run: uv run python -m logviewer.server
Opens at http://localhost:7331
"""

import json
import os
import threading
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Config ─────────────────────────────────────────────────────────────────────

LOG_DIR = Path(os.getenv("LLM_LOG_DIR", "logs")).resolve()
PORT = int(os.getenv("LOG_VIEWER_PORT", "7331"))


def _safe_log_path(filename: str) -> Path:
    """Resolve filename to an absolute path and reject anything outside LOG_DIR."""
    # Reject names that aren't a plain *.jsonl filename (no slashes, no dots leading)
    if "/" in filename or "\\" in filename or not filename.endswith(".jsonl"):
        raise HTTPException(400, "Invalid log file name")
    path = (LOG_DIR / filename).resolve()
    if not path.is_relative_to(LOG_DIR):
        raise HTTPException(400, "Invalid log file name")
    return path


# ── File index ─────────────────────────────────────────────────────────────────
# For each log file we build a byte-offset index so we can seek to any line
# in O(1) without loading the whole file into memory.

_index_cache: dict[str, list[int]] = {}  # path → list of byte offsets
_index_lock = threading.Lock()


def _build_index(path: Path) -> list[int]:
    """Return a list of byte offsets, one per line."""
    offsets: list[int] = []
    with path.open("rb") as f:
        offset = 0
        for line in f:
            offsets.append(offset)
            offset += len(line)
    return offsets


def _get_index(path: Path) -> list[int]:
    key = str(path)
    mtime = path.stat().st_mtime
    with _index_lock:
        cached = _index_cache.get(key)
        if cached is None or cached[0] != mtime:  # type: ignore[index]
            idx = _build_index(path)
            _index_cache[key] = [mtime, *idx]  # type: ignore[list-item]
        return _index_cache[key][1:]  # type: ignore[return-value]


def _read_entry(path: Path, line_no: int) -> dict:
    offsets = _get_index(path)
    if line_no < 0 or line_no >= len(offsets):
        raise IndexError(f"line {line_no} out of range (0–{len(offsets) - 1})")
    with path.open("rb") as f:
        f.seek(offsets[line_no])
        return json.loads(f.readline())


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="LLM Log Viewer", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path(__file__).parent / "index.html"
    return HTMLResponse(html.read_text())


# ── API ────────────────────────────────────────────────────────────────────────


@app.get("/api/files")
async def list_files():
    if not LOG_DIR.exists():
        return {"files": []}
    files = sorted(
        LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    result = []
    for f in files:
        try:
            idx = _get_index(f)
            result.append({"name": f.name, "count": len(idx), "size": f.stat().st_size})
        except Exception:
            pass
    return {"files": result}


@app.get("/api/entries")
async def list_entries(
    file: str = "llm.jsonl",
    page: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    search: str = "",
    order: str = "desc",
):
    path = _safe_log_path(file)
    if not path.exists():
        raise HTTPException(404, f"Log file not found: {file}")

    offsets = _get_index(path)
    total = len(offsets)

    # Build ordered indices
    indices = list(range(total))
    if order == "desc":
        indices = list(reversed(indices))

    # Search filter (simple substring match on raw JSON line)
    if search:
        s = search.lower()
        filtered = []
        with path.open("rb") as f:
            for i in indices:
                f.seek(offsets[i])
                raw = f.readline().decode("utf-8", errors="replace")
                if s in raw.lower():
                    filtered.append(i)
        indices = filtered

    filtered_total = len(indices)
    start = (page - 1) * limit
    page_indices = indices[start : start + limit]

    entries = []
    with path.open("rb") as f:
        for i in page_indices:
            f.seek(offsets[i])
            try:
                entry = json.loads(f.readline())
            except json.JSONDecodeError:
                continue
            # Return only summary fields for the list view
            entries.append(
                {
                    "line": i,
                    "ts": entry.get("ts"),
                    "provider": entry.get("provider"),
                    "model": entry.get("model"),
                    "stop_reason": entry.get("response", {}).get("stop_reason"),
                    "usage": entry.get("response", {}).get("usage", {}),
                    "n_messages": len(entry.get("request", {}).get("messages", [])),
                    "n_tool_calls": len(
                        entry.get("response", {}).get("tool_calls", [])
                    ),
                    "has_content": bool(entry.get("response", {}).get("content")),
                }
            )

    return {
        "total": filtered_total,
        "page": page,
        "limit": limit,
        "pages": max(1, (filtered_total + limit - 1) // limit),
        "entries": entries,
    }


@app.get("/api/entry/{line}")
async def get_entry(line: int, file: str = "llm.jsonl"):
    path = _safe_log_path(file)
    if not path.exists():
        raise HTTPException(404, f"Log file not found: {file}")
    try:
        return _read_entry(path, line)
    except IndexError as e:
        raise HTTPException(404, str(e))


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("logviewer.server:app", host="0.0.0.0", port=PORT, reload=False)
