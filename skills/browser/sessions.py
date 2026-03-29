"""
skills/browser/sessions.py

Persistent browser session management for Atlas.

All Playwright state lives on a dedicated daemon thread with its own event loop.
The public API is `run_on_browser_thread(coro)` which submits work from any
thread and blocks until the result is ready.

Session lifecycle:
  - First call without session_id → creates a new session (browser + context)
  - Subsequent calls with session_id → reuses existing session
  - Sessions auto-close after IDLE_TIMEOUT seconds of inactivity
  - Max MAX_SESSIONS concurrent sessions; oldest evicted if limit reached
  - Explicit close via action="close"
  - atexit hook closes everything on process shutdown
"""

import asyncio
import atexit
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from paths import DOWNLOADS_DIR, SCREENSHOTS_DIR

logger = logging.getLogger(__name__)

MAX_SESSIONS = 10
IDLE_TIMEOUT = 600  # seconds
MAX_TEXT_CHARS = 12_000
_CLEANUP_INTERVAL = 60  # seconds between idle-session sweeps
_HARD_TIMEOUT = 300  # seconds — run_on_browser_thread gives up

# ── Dedicated browser thread ────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_started = threading.Event()
_lock = threading.Lock()


def _run_loop(loop: asyncio.AbstractEventLoop):
    """Target for the browser daemon thread."""
    asyncio.set_event_loop(loop)
    _started.set()
    loop.run_forever()


def _ensure_thread() -> asyncio.AbstractEventLoop:
    """Start the browser daemon thread if not already running."""
    global _loop, _thread
    with _lock:
        if _loop is not None and _thread is not None and _thread.is_alive():
            return _loop
        _loop = asyncio.new_event_loop()
        _started.clear()
        _thread = threading.Thread(
            target=_run_loop,
            args=(_loop,),
            daemon=True,
            name="atlas-browser-loop",
        )
        _thread.start()
        _started.wait()  # block until run_forever() is active
        return _loop


def run_on_browser_thread(coro) -> Any:
    """Submit a coroutine to the browser thread and block until complete."""
    loop = _ensure_thread()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=_HARD_TIMEOUT)


# ── Session data ────────────────────────────────────────────────────────────


@dataclass
class BrowserSession:
    session_id: str
    playwright: Any  # async_playwright context manager result
    browser: Any  # Browser instance
    context: Any  # BrowserContext (cookies, storage)
    pages: dict[str, Any] = field(default_factory=dict)  # tab_id → Page
    active_tab: str = ""
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def page(self) -> Any:
        """The currently active page."""
        return self.pages[self.active_tab]

    def touch(self):
        """Update last-used timestamp."""
        self.last_used = time.time()


# ── Session manager ─────────────────────────────────────────────────────────

_manager: "SessionManager | None" = None


def _get_manager() -> "SessionManager":
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, BrowserSession] = {}
        self._cleanup_scheduled = False

    async def get_or_create(self, session_id: str | None = None) -> BrowserSession:
        """Return an existing session or create a new one."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.touch()
            return session

        if session_id and session_id not in self._sessions:
            raise ValueError(
                f"Session '{session_id}' not found. "
                "Omit session_id to create a new session, or use 'tabs' on an active session."
            )

        # Evict oldest if at limit
        if len(self._sessions) >= MAX_SESSIONS:
            oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_used)
            logger.info(f"Evicting oldest session {oldest_id} (limit {MAX_SESSIONS})")
            await self._close_one(oldest_id)

        return await self._create_session()

    async def _create_session(self) -> BrowserSession:
        from playwright.async_api import async_playwright

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        session_id = uuid.uuid4().hex[:8]
        tab_id = uuid.uuid4().hex[:8]

        session = BrowserSession(
            session_id=session_id,
            playwright=pw,
            browser=browser,
            context=context,
            pages={tab_id: page},
            active_tab=tab_id,
        )
        self._sessions[session_id] = session
        self._schedule_cleanup()
        logger.info(f"Created browser session {session_id}")
        return session

    async def close_session(self, session_id: str) -> str:
        if session_id not in self._sessions:
            return f"Session '{session_id}' not found."
        await self._close_one(session_id)
        return f"Session '{session_id}' closed."

    async def _close_one(self, session_id: str):
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            await session.context.close()
        except Exception:
            pass
        try:
            await session.browser.close()
        except Exception:
            pass
        try:
            await session.playwright.stop()
        except Exception:
            pass
        logger.info(f"Closed browser session {session_id}")

    async def close_all(self):
        for sid in list(self._sessions):
            await self._close_one(sid)

    async def _cleanup_idle(self):
        now = time.time()
        to_close = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_used > IDLE_TIMEOUT
        ]
        for sid in to_close:
            logger.info(f"Closing idle session {sid}")
            await self._close_one(sid)
        if self._sessions:
            self._schedule_cleanup()
        else:
            self._cleanup_scheduled = False

    def _schedule_cleanup(self):
        if self._cleanup_scheduled:
            return
        self._cleanup_scheduled = True
        loop = _ensure_thread()
        loop.call_later(_CLEANUP_INTERVAL, self._run_cleanup)

    def _run_cleanup(self):
        """Called by loop.call_later on the browser thread — schedule the async cleanup."""
        self._cleanup_scheduled = False
        # We're already on the browser thread's event loop, so get_event_loop works.
        asyncio.ensure_future(self._cleanup_idle())


# ── atexit shutdown ─────────────────────────────────────────────────────────


def _shutdown():
    mgr = _manager
    if mgr is None:
        return
    try:
        run_on_browser_thread(mgr.close_all())
    except Exception:
        pass


atexit.register(_shutdown)


# ── Action execution ────────────────────────────────────────────────────────


async def execute_action(
    action: str,
    url: str = "",
    session_id: str | None = None,
    selector: str = "",
    value: str = "",
    tab_id: str = "",
    wait_for: str = "",
    timeout: int = 30,
    wait_seconds: float = 0,
) -> str:
    """Main dispatch — runs on the browser event loop thread."""
    from playwright.async_api import TimeoutError as PWTimeout

    manager = _get_manager()
    timeout_ms = min(int(timeout), 120) * 1000

    logger.info(f"Browser action={action} session={session_id or '(new)'} url={url[:80] if url else ''}")

    # Close doesn't need a live page
    if action == "close":
        if not session_id:
            return "Error: 'session_id' is required for close action."
        return await manager.close_session(session_id)

    # Get or create session
    try:
        session = await manager.get_or_create(session_id)
    except ValueError as e:
        return str(e)

    is_new_session = session_id is None or session_id == ""

    # Navigate if url provided and (goto action or new session)
    if url and (action == "goto" or is_new_session):
        logger.info(f"Navigating to {url[:120]} (session {session.session_id})")
        try:
            await session.page.goto(
                url, timeout=timeout_ms, wait_until="domcontentloaded"
            )
            logger.info(f"Navigation complete (session {session.session_id})")
        except PWTimeout:
            logger.warning(f"Navigation timed out: {url[:120]}")
            return (
                f"Error: page load timed out after {timeout}s — {url}\n"
                f"session_id: {session.session_id}"
            )

    if wait_for:
        try:
            await session.page.wait_for_selector(wait_for, timeout=timeout_ms)
        except PWTimeout:
            pass  # proceed anyway

    if wait_seconds and wait_seconds > 0:
        await asyncio.sleep(min(float(wait_seconds), 30))

    # Dispatch
    logger.info(f"Dispatching action={action} (session {session.session_id})")
    result = await _dispatch(session, action, selector, value, tab_id, url, timeout_ms)
    logger.info(f"Action {action} complete (session {session.session_id}, {len(result)} chars)")
    return f"{result}\nsession_id: {session.session_id}"


async def _dispatch(
    session: BrowserSession,
    action: str,
    selector: str,
    value: str,
    tab_id: str,
    url: str,
    timeout_ms: int,
) -> str:
    from playwright.async_api import TimeoutError as PWTimeout

    page = session.page

    match action:
        case "goto":
            title = await page.title()
            return f"Navigated to: {title}\nURL: {page.url}"

        case "read":
            return await _action_read(page)

        case "screenshot":
            return await _action_screenshot(page)

        case "click":
            if not selector:
                return "Error: 'selector' is required for click action."
            try:
                await page.click(selector, timeout=timeout_ms)
                return f"Clicked: {selector}\nURL: {page.url}"
            except PWTimeout:
                return f"Error: element not found or not clickable: {selector}"

        case "fill":
            if not selector:
                return "Error: 'selector' is required for fill action."
            if not value:
                return "Error: 'value' is required for fill action."
            try:
                await page.fill(selector, value, timeout=timeout_ms)
                return f"Filled '{selector}' with text ({len(value)} chars)"
            except PWTimeout:
                return f"Error: input element not found: {selector}"

        case "extract":
            if not selector:
                return "Error: 'selector' is required for extract action."
            return await _action_extract(page, selector)

        case "scroll":
            return await _action_scroll(page, value, selector)

        case "back":
            try:
                await page.go_back(timeout=timeout_ms)
                return f"Navigated back.\nURL: {page.url}"
            except PWTimeout:
                return "Error: navigation back timed out."

        case "forward":
            try:
                await page.go_forward(timeout=timeout_ms)
                return f"Navigated forward.\nURL: {page.url}"
            except PWTimeout:
                return "Error: navigation forward timed out."

        case "current_url":
            title = await page.title()
            return f"URL: {page.url}\nTitle: {title}"

        case "tabs":
            return _format_tabs(session)

        case "new_tab":
            return await _action_new_tab(session, url=value or url)

        case "switch_tab":
            if not tab_id:
                return "Error: 'tab_id' is required for switch_tab action."
            return _action_switch_tab(session, tab_id)

        case "download":
            return await _action_download(page, session, selector, value)

        case _:
            valid = (
                "goto, read, screenshot, click, fill, extract, scroll, "
                "back, forward, current_url, tabs, new_tab, switch_tab, "
                "download, close"
            )
            return f"Unknown action: '{action}'. Valid: {valid}"


# ── Action implementations ──────────────────────────────────────────────────


async def _action_read(page) -> str:
    text = await page.evaluate(
        """() => {
        const scripts = document.querySelectorAll('script, style, noscript');
        scripts.forEach(el => el.remove());
        return document.body
            ? document.body.innerText
            : document.documentElement.innerText;
    }"""
    )
    text = _clean_text(text or "")
    title = await page.title()
    header = f"# {title}\nURL: {page.url}\n\n"
    full = header + text
    if len(full) > MAX_TEXT_CHARS:
        full = full[:MAX_TEXT_CHARS] + f"\n\n... [truncated at {MAX_TEXT_CHARS} chars]"
    return full


async def _action_screenshot(page) -> str:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = SCREENSHOTS_DIR / filename
    await page.screenshot(path=str(path), full_page=False)
    return f"Screenshot saved: {path}"


async def _action_extract(page, selector: str) -> str:
    """Extract content from matched elements — text, images, audio, video, links."""
    try:
        elements = await page.query_selector_all(selector)
    except Exception as e:
        return f"Error querying selector: {e}"

    if not elements:
        return f"No elements matched: {selector}"

    results = []
    for el in elements[:20]:
        tag = (await el.evaluate("el => el.tagName")).lower()

        if tag == "img":
            src = await el.get_attribute("src") or ""
            alt = await el.get_attribute("alt") or ""
            if src:
                local = await _download_resource(page, src)
                results.append(f"[image] {alt}\n  src: {src}\n  saved: {local}")
            else:
                results.append(f"[image] {alt} (no src)")

        elif tag in ("audio", "video"):
            src = await el.get_attribute("src") or ""
            # Check for <source> children
            if not src:
                source_el = await el.query_selector("source")
                if source_el:
                    src = await source_el.get_attribute("src") or ""
            if src:
                local = await _download_resource(page, src)
                results.append(f"[{tag}] src: {src}\n  saved: {local}")
            else:
                results.append(f"[{tag}] (no src found)")

        elif tag == "a":
            href = await el.get_attribute("href") or ""
            text = (await el.inner_text()).strip()
            results.append(f"[link] {text}\n  href: {href}")

        else:
            text = (await el.inner_text()).strip()
            if text:
                results.append(text)

    result = "\n---\n".join(results)
    if len(result) > MAX_TEXT_CHARS:
        result = result[:MAX_TEXT_CHARS] + "\n... [truncated]"
    return result or "(elements found but no extractable content)"


async def _action_scroll(page, value: str, selector: str) -> str:
    if selector:
        try:
            await page.locator(selector).scroll_into_view_if_needed()
            return f"Scrolled element into view: {selector}"
        except Exception as e:
            return f"Error scrolling to element: {e}"

    value = (value or "down 500").strip().lower()

    if value == "bottom":
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    elif value == "top":
        await page.evaluate("window.scrollTo(0, 0)")
    elif value.startswith("up"):
        amount = _parse_scroll_amount(value, 500)
        await page.evaluate(f"window.scrollBy(0, -{amount})")
    else:
        amount = _parse_scroll_amount(value, 500)
        await page.evaluate(f"window.scrollBy(0, {amount})")

    scroll_pos = await page.evaluate("window.scrollY")
    scroll_max = await page.evaluate(
        "document.body.scrollHeight - window.innerHeight"
    )
    return f"Scrolled. Position: {int(scroll_pos)}/{int(scroll_max)}px"


async def _action_new_tab(session: BrowserSession, url: str = "") -> str:
    page = await session.context.new_page()
    tab_id = uuid.uuid4().hex[:8]
    session.pages[tab_id] = page
    session.active_tab = tab_id
    if url:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception:
            pass
    return f"New tab opened: {tab_id}\nURL: {page.url}"


def _action_switch_tab(session: BrowserSession, tab_id: str) -> str:
    if tab_id not in session.pages:
        available = ", ".join(session.pages.keys())
        return f"Error: no tab '{tab_id}'. Available: {available}"
    session.active_tab = tab_id
    return f"Switched to tab {tab_id}"


def _format_tabs(session: BrowserSession) -> str:
    lines = ["Open tabs:"]
    for tid, pg in session.pages.items():
        marker = " (active)" if tid == session.active_tab else ""
        try:
            url = pg.url
        except Exception:
            url = "(closed)"
        lines.append(f"  {tid}: {url}{marker}")
    return "\n".join(lines)


async def _action_download(page, session: BrowserSession, selector: str, value: str) -> str:
    """Download a resource by selector (src/href) or direct URL (via value)."""
    url = ""
    if value and (value.startswith("http://") or value.startswith("https://")):
        url = value
    elif selector:
        el = await page.query_selector(selector)
        if not el:
            return f"Error: no element matched: {selector}"
        url = (
            await el.get_attribute("src")
            or await el.get_attribute("href")
            or ""
        )
    if not url:
        return "Error: could not determine download URL. Provide a selector with src/href or a direct URL in 'value'."
    local = await _download_resource(page, url)
    return f"Downloaded: {local}\nSource: {url}"


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _download_resource(page, url: str) -> str:
    """Download a URL to ~/agent-files/downloads/ and return the local path."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve relative URLs
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        parsed = urlparse(page.url)
        url = f"{parsed.scheme}://{parsed.netloc}{url}"
    elif not url.startswith("http"):
        # relative path
        base = page.url.rsplit("/", 1)[0]
        url = f"{base}/{url}"

    # Derive filename
    path_part = urlparse(url).path
    filename = Path(path_part).name or "download"
    # Deduplicate with timestamp
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = DOWNLOADS_DIR / f"{stem}_{ts}{suffix}"

    try:
        response = await page.context.request.get(url)
        body = await response.body()
        local_path.write_bytes(body)
        return str(local_path)
    except Exception as e:
        return f"(download failed: {e})"


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _parse_scroll_amount(value: str, default: int) -> int:
    """Extract a numeric pixel amount from a scroll value string."""
    parts = value.split()
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return default
