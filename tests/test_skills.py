"""
tests/test_skills.py

Unit tests for Phase 3 skills: shell_exec, http_request, code_runner, browser.
Uses mocks where external dependencies (network, browser) are involved.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── shell_exec ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shell_exec_basic():
    """shell_exec returns stdout of a simple command."""
    from skills.shell_exec.tool import run

    result = await run("echo hello")
    assert "hello" in result


@pytest.mark.asyncio
async def test_shell_exec_nonzero_exit():
    """shell_exec includes the exit code when it's non-zero."""
    from skills.shell_exec.tool import run

    result = await run("exit 1", timeout=5)
    assert "1" in result  # exit code mentioned


@pytest.mark.asyncio
async def test_shell_exec_timeout():
    """shell_exec returns a timeout error for commands that run too long."""
    from skills.shell_exec.tool import run

    result = await run("sleep 10", timeout=1)
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_shell_exec_invalid_working_dir():
    """shell_exec returns an error for a non-existent working directory."""
    from skills.shell_exec.tool import run

    result = await run("echo hi", working_dir="/this/does/not/exist/ever")
    assert "does not exist" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_shell_exec_stderr_captured():
    """shell_exec captures stderr as well as stdout."""
    from skills.shell_exec.tool import run

    result = await run("python3 -c \"import sys; sys.stderr.write('err msg')\"")
    assert "err msg" in result


# ── code_runner ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_code_runner_basic():
    """code_runner executes Python and returns stdout."""
    from skills.code_runner.tool import run

    result = await run("print('hello from python')")
    assert "hello from python" in result


@pytest.mark.asyncio
async def test_code_runner_math():
    """code_runner can do arithmetic."""
    from skills.code_runner.tool import run

    result = await run("print(2 ** 10)")
    assert "1024" in result


@pytest.mark.asyncio
async def test_code_runner_exception():
    """code_runner returns the traceback on exception, not raises."""
    from skills.code_runner.tool import run

    result = await run("raise ValueError('oops')")
    assert "ValueError" in result
    assert "oops" in result


@pytest.mark.asyncio
async def test_code_runner_timeout():
    """code_runner kills infinite loops after the timeout."""
    from skills.code_runner.tool import run

    result = await run("while True: pass", timeout=1)
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_code_runner_no_stdin():
    """code_runner doesn't hang waiting for stdin."""
    from skills.code_runner.tool import run

    # input() with closed stdin should raise EOFError immediately
    result = await run("x = input('enter: ')", timeout=5)
    assert "EOF" in result or "timed out" not in result.lower()


# ── http_request ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_request_get(httpx_mock=None):
    """http_request returns status and body for a successful GET."""
    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.reason_phrase = "OK"
    mock_response.url = "https://example.com"
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = "<html><body>Hello World</body></html>"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=mock_response)

    from skills.http_request.tool import run

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await run("https://example.com")

    assert "200" in result
    assert "Hello World" in result


@pytest.mark.asyncio
async def test_http_request_timeout():
    """http_request returns an error message on timeout."""
    import httpx

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

    from skills.http_request.tool import run

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await run("https://example.com", timeout=1)

    assert "timed out" in result.lower() or "timeout" in result.lower()


@pytest.mark.asyncio
async def test_http_request_post_with_body():
    """http_request sends the body for POST requests."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.reason_phrase = "Created"
    mock_response.url = "https://api.example.com/items"
    mock_response.headers = {"content-type": "application/json"}
    mock_response.text = '{"id": 42}'

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=mock_response)

    from skills.http_request.tool import run

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await run(
            "https://api.example.com/items",
            method="POST",
            body='{"name": "test"}',
        )

    assert "201" in result
    assert "42" in result
    # Verify body was passed
    call_kwargs = mock_client.request.call_args
    assert b'{"name": "test"}' == call_kwargs.kwargs.get("content")


# ── browser ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_browser_read_page():
    """browser skill reads page text content via Playwright."""
    mock_page = AsyncMock()
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.goto = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="This is the page content.")
    mock_page.query_selector_all = AsyncMock(return_value=[])

    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.close = AsyncMock()

    mock_chromium = AsyncMock()
    mock_chromium.launch = AsyncMock(return_value=mock_browser)

    mock_pw = AsyncMock()
    mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_pw.__aexit__ = AsyncMock(return_value=False)
    mock_pw.chromium = mock_chromium

    from skills.browser.tool import run

    with patch("playwright.async_api.async_playwright", return_value=mock_pw):
        result = await run("https://example.com", action="read")

    assert "Test Page" in result
    assert "page content" in result


@pytest.mark.asyncio
async def test_browser_missing_playwright():
    """browser skill returns a helpful error if playwright is not installed."""
    from skills.browser.tool import run

    with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'playwright'"),
        ):
            result = await run("https://example.com")
    assert "playwright" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_browser_extract_selector():
    """browser extract action returns inner text of matching elements."""
    mock_el1 = AsyncMock()
    mock_el1.inner_text = AsyncMock(return_value="Item One")
    mock_el2 = AsyncMock()
    mock_el2.inner_text = AsyncMock(return_value="Item Two")

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.query_selector_all = AsyncMock(return_value=[mock_el1, mock_el2])

    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.close = AsyncMock()

    mock_chromium = AsyncMock()
    mock_chromium.launch = AsyncMock(return_value=mock_browser)

    mock_pw = AsyncMock()
    mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_pw.__aexit__ = AsyncMock(return_value=False)
    mock_pw.chromium = mock_chromium

    from skills.browser.tool import run

    with patch("playwright.async_api.async_playwright", return_value=mock_pw):
        result = await run("https://example.com", action="extract", selector="li")

    assert "Item One" in result
    assert "Item Two" in result
