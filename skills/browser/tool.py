"""
skills/browser/tool.py

Full browser automation via Playwright (Chromium, headless).
Supports reading page content, clicking, filling forms, extracting elements,
and taking screenshots.
"""

from pathlib import Path

PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to navigate to",
        },
        "action": {
            "type": "string",
            "enum": ["read", "screenshot", "click", "fill", "extract"],
            "description": "Action to perform (default: read)",
            "default": "read",
        },
        "selector": {
            "type": "string",
            "description": "CSS selector for click/fill/extract actions",
        },
        "value": {
            "type": "string",
            "description": "Text to type into the selected element (for fill action)",
        },
        "wait_for": {
            "type": "string",
            "description": "CSS selector to wait for before reading (for dynamic/JS pages)",
        },
        "timeout": {
            "type": "integer",
            "description": "Page load timeout in seconds (default: 30)",
            "default": 30,
        },
        "wait_seconds": {
            "type": "number",
            "description": "Seconds to wait after page load before acting (useful for animations or lazy-loaded content, default: 0)",
            "default": 0,
        },
    },
    "required": ["url"],
}

SCREENSHOT_DIR = Path.home() / "agent-files" / "screenshots"
MAX_TEXT_CHARS = 12_000


async def run(
    url: str,
    action: str = "read",
    selector: str = "",
    value: str = "",
    wait_for: str = "",
    timeout: int = 30,
    wait_seconds: float = 0,
    **kwargs,
) -> str:
    """
    Control a headless Chromium browser via Playwright.

    Args:
        url:          URL to navigate to.
        action:       "read" | "screenshot" | "click" | "fill" | "extract"
        selector:     CSS selector (required for click/fill/extract).
        value:        Text to type (required for fill).
        wait_for:     CSS selector to wait for after navigation.
        timeout:      Page load timeout in seconds.
        wait_seconds: Extra seconds to sleep after load/wait_for, before acting.

    Returns:
        Result string. Never raises.
    """
    try:
        import asyncio
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        return (
            "Error: Playwright not installed.\n"
            "Run: uv add playwright && uv run playwright install chromium"
        )

    timeout_ms = min(int(timeout), 120) * 1000

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            except PWTimeout:
                await browser.close()
                return f"Error: page load timed out after {timeout}s — {url}"

            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=timeout_ms)
                except PWTimeout:
                    pass  # Proceed anyway — may still have useful content

            if wait_seconds and wait_seconds > 0:
                await asyncio.sleep(min(float(wait_seconds), 30))

            result = await _perform_action(page, action, selector, value, url, timeout_ms)
            await browser.close()
            return result

    except Exception as e:
        return f"Browser error: {e}"


async def _perform_action(page, action: str, selector: str, value: str, url: str, timeout_ms: int) -> str:
    from playwright.async_api import TimeoutError as PWTimeout

    if action == "read":
        # Extract visible text, stripping scripts/styles
        text = await page.evaluate("""() => {
            const scripts = document.querySelectorAll('script, style, noscript');
            scripts.forEach(el => el.remove());
            return document.body ? document.body.innerText : document.documentElement.innerText;
        }""")
        text = _clean_text(text or "")
        title = await page.title()
        header = f"# {title}\nURL: {url}\n\n"
        full = header + text
        if len(full) > MAX_TEXT_CHARS:
            full = full[:MAX_TEXT_CHARS] + f"\n\n... [truncated at {MAX_TEXT_CHARS} chars]"
        return full

    elif action == "screenshot":
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = SCREENSHOT_DIR / filename
        await page.screenshot(path=str(path), full_page=False)
        return f"Screenshot saved: {path}"

    elif action == "click":
        if not selector:
            return "Error: 'selector' is required for click action"
        try:
            await page.click(selector, timeout=timeout_ms)
            return f"✅ Clicked: {selector}"
        except PWTimeout:
            return f"Error: element not found or not clickable: {selector}"

    elif action == "fill":
        if not selector:
            return "Error: 'selector' is required for fill action"
        if value is None:
            return "Error: 'value' is required for fill action"
        try:
            await page.fill(selector, value, timeout=timeout_ms)
            return f"✅ Filled '{selector}' with text ({len(value)} chars)"
        except PWTimeout:
            return f"Error: input element not found: {selector}"

    elif action == "extract":
        if not selector:
            return "Error: 'selector' is required for extract action"
        try:
            elements = await page.query_selector_all(selector)
            if not elements:
                return f"No elements matched: {selector}"
            texts = []
            for el in elements[:20]:  # Cap at 20 elements
                t = await el.inner_text()
                if t.strip():
                    texts.append(t.strip())
            result = "\n---\n".join(texts)
            if len(result) > MAX_TEXT_CHARS:
                result = result[:MAX_TEXT_CHARS] + "\n... [truncated]"
            return result or "(elements found but no text content)"
        except Exception as e:
            return f"Error extracting elements: {e}"

    else:
        return f"Unknown action: '{action}'. Use: read, screenshot, click, fill, extract"


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    import re
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
