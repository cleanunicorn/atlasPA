# browser

Control a real Chromium browser via Playwright. Navigate pages, extract content, click elements, fill forms, and take screenshots.

## When to use
- When you need the full rendered content of a page (JavaScript-heavy sites, SPAs)
- When http_request is not enough (login-gated pages, dynamic content, forms)
- When the user asks you to interact with a website
- When you need to extract structured content from a page

## Input
- `url` (string, required): The URL to navigate to
- `action` (string, optional): What to do — "read" (default), "screenshot", "click", "fill", "extract"
- `selector` (string, optional): CSS selector for click/fill/extract actions
- `value` (string, optional): Text to fill into the selected input (for "fill" action)
- `wait_for` (string, optional): CSS selector to wait for before reading (for dynamic pages)
- `timeout` (integer, optional): Page load timeout in seconds (default: 30)

## Output
- "read": Full visible text content of the page
- "screenshot": Path to a saved screenshot file
- "click": Confirmation that the element was clicked
- "fill": Confirmation that text was typed into the element
- "extract": Inner text of the matched element(s)

## Requirements
Playwright must be installed and browsers downloaded:
    uv add playwright
    uv run playwright install chromium
