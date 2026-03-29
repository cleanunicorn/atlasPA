# browser

Control a real Chromium browser with persistent sessions via Playwright. Navigate pages, extract content, click elements, fill forms, scroll, take screenshots, and download media.

## Session workflow

1. First call (omit `session_id`) creates a new browser session and returns a `session_id`
2. Pass `session_id` on subsequent calls to reuse the same browser (cookies, auth, history preserved)
3. Sessions auto-close after 10 minutes of inactivity
4. Use `action: "close"` to explicitly end a session

## Actions

| Action | Required params | Description |
|--------|----------------|-------------|
| `goto` | `url` | Navigate to a URL |
| `read` | — | Extract visible page text |
| `screenshot` | — | Save viewport as PNG |
| `click` | `selector` | Click an element |
| `fill` | `selector`, `value` | Type text into an input |
| `extract` | `selector` | Get content from matched elements (text, images, media, links) |
| `scroll` | `value` or `selector` | Scroll page or element into view |
| `back` | — | Browser history back |
| `forward` | — | Browser history forward |
| `current_url` | — | Return current URL and title |
| `tabs` | — | List open tabs with IDs |
| `new_tab` | `value` (URL, optional) | Open a new tab |
| `switch_tab` | `tab_id` | Switch to a different tab |
| `download` | `selector` or `value` (URL) | Download a resource to disk |
| `close` | `session_id` | Close the browser session |

## Parameters

- `action` (string, required): The action to perform
- `url` (string): URL to navigate to. Required for `goto`. Auto-navigates on first call if provided
- `session_id` (string): Reuse an existing session. Omit to create new
- `selector` (string): CSS selector for click/fill/extract/scroll/download
- `value` (string): Text for fill, scroll direction (`"down 500"`, `"top"`, `"bottom"`), URL for new_tab/download
- `tab_id` (string): Tab ID for switch_tab (use `tabs` to list)
- `wait_for` (string): CSS selector to wait for before acting
- `timeout` (integer): Timeout in seconds (default: 30, max: 120)
- `wait_seconds` (number): Extra wait after navigation (default: 0)

## Extract behavior

The `extract` action handles different element types:
- **Text elements** (div, p, span, li, etc.) — returns inner text
- **Images** (`img`) — downloads to `~/agent-files/downloads/`, returns path + alt text
- **Audio/Video** — downloads source, returns path
- **Links** (`a`) — returns href + link text

## Scroll syntax

- `"down 500"` — scroll down 500px
- `"up 300"` — scroll up 300px
- `"top"` — scroll to page top
- `"bottom"` — scroll to page bottom
- With `selector` — scrolls that element into view

## Example: multi-step login flow

```
1. action=goto, url=https://example.com/login          → session_id: a3f1b2c4
2. action=fill, session_id=a3f1b2c4, selector=#email, value=user@example.com
3. action=fill, session_id=a3f1b2c4, selector=#password, value=secret
4. action=click, session_id=a3f1b2c4, selector=button[type=submit]
5. action=read, session_id=a3f1b2c4                     → dashboard content
6. action=close, session_id=a3f1b2c4
```

## Requirements

Playwright must be installed:
    uv add playwright
    uv run playwright install chromium
