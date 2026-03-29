"""
skills/browser/tool.py

Session-based browser automation via Playwright (Chromium, headless).

Supports persistent sessions across multiple tool calls — the LLM can log in,
navigate, fill forms, scroll, screenshot, and extract content across an entire
ReAct loop (and across conversation turns).

The run() function is synchronous. All Playwright work is dispatched to a
dedicated background thread managed by sessions.py.
"""

PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "goto",
                "read",
                "screenshot",
                "click",
                "fill",
                "extract",
                "scroll",
                "back",
                "forward",
                "current_url",
                "tabs",
                "new_tab",
                "switch_tab",
                "download",
                "close",
            ],
            "description": (
                "Action to perform. "
                "'goto' navigates to a URL. "
                "'read' extracts visible page text. "
                "'screenshot' saves a PNG. "
                "'click' clicks an element by CSS selector. "
                "'fill' types text into an input. "
                "'extract' gets content (text, images, media) from matched elements. "
                "'scroll' scrolls the page (value: 'down 500', 'up 300', 'top', 'bottom') or an element into view (selector). "
                "'back'/'forward' navigate browser history. "
                "'current_url' returns the current URL and title. "
                "'tabs' lists open tabs. "
                "'new_tab' opens a new tab. "
                "'switch_tab' switches to a tab by tab_id. "
                "'download' downloads a resource (image, file) to disk. "
                "'close' closes the browser session."
            ),
            "default": "read",
        },
        "url": {
            "type": "string",
            "description": (
                "URL to navigate to. Required for 'goto'. "
                "When provided on the first call (no session_id), auto-navigates before the action."
            ),
        },
        "session_id": {
            "type": "string",
            "description": (
                "Reuse an existing browser session. "
                "Omit to create a new session. "
                "The session_id is returned in every response."
            ),
        },
        "selector": {
            "type": "string",
            "description": "CSS selector for click/fill/extract/scroll/download actions.",
        },
        "value": {
            "type": "string",
            "description": (
                "Text to type (fill), scroll direction (scroll: 'down 500', 'top'), "
                "URL to open (new_tab), or direct download URL (download)."
            ),
        },
        "tab_id": {
            "type": "string",
            "description": "Tab ID for switch_tab action. Use 'tabs' to list available IDs.",
        },
        "wait_for": {
            "type": "string",
            "description": "CSS selector to wait for before acting (for dynamic/JS pages).",
        },
        "timeout": {
            "type": "integer",
            "description": "Action timeout in seconds (default: 30, max: 120).",
            "default": 30,
        },
        "wait_seconds": {
            "type": "number",
            "description": "Extra seconds to wait after navigation before acting (default: 0).",
            "default": 0,
        },
    },
    "required": ["action"],
}


def run(
    action: str = "read",
    url: str = "",
    session_id: str = "",
    selector: str = "",
    value: str = "",
    tab_id: str = "",
    wait_for: str = "",
    timeout: int = 30,
    wait_seconds: float = 0,
    **kwargs,
) -> str:
    """
    Browser automation with persistent sessions.

    Dispatches to a dedicated Playwright thread. Returns a result string
    (never raises). Every response includes the session_id for reuse.
    """
    try:
        from skills.browser.sessions import run_on_browser_thread, execute_action
    except ImportError as e:
        return (
            f"Error: {e}\n"
            "Run: uv add playwright && uv run playwright install chromium"
        )

    try:
        return run_on_browser_thread(
            execute_action(
                action=action,
                url=url,
                session_id=session_id or None,
                selector=selector,
                value=value,
                tab_id=tab_id,
                wait_for=wait_for,
                timeout=timeout,
                wait_seconds=wait_seconds,
            )
        )
    except Exception as e:
        return f"Browser error: {e}"
