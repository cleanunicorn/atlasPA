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


def test_browser_read_page():
    """browser skill reads page text via Playwright (session-based)."""
    from skills.browser.tool import run

    fake_response = "# Test Page\nURL: https://example.com\n\nThis is the page content.\nsession_id: abc12345"
    with patch("skills.browser.sessions.run_on_browser_thread", return_value=fake_response):
        result = run(action="read", url="https://example.com")

    assert "Test Page" in result
    assert "page content" in result
    assert "session_id" in result


def test_browser_missing_playwright():
    """browser skill returns a helpful error if sessions can't be imported."""
    from skills.browser.tool import run

    # Simulate import failure of the sessions module
    with patch.dict("sys.modules", {"skills.browser.sessions": None}):
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'playwright'"),
        ):
            result = run(action="read", url="https://example.com")
    assert "error" in result.lower()


def test_browser_extract_selector():
    """browser extract action returns text of matching elements."""
    from skills.browser.tool import run

    fake_response = "Item One\n---\nItem Two\nsession_id: abc12345"
    with patch("skills.browser.sessions.run_on_browser_thread", return_value=fake_response):
        result = run(action="extract", url="https://example.com", selector="li")

    assert "Item One" in result
    assert "Item Two" in result


def test_browser_session_reuse():
    """Passing session_id reuses an existing session."""
    from skills.browser.tool import run

    with patch("skills.browser.sessions.run_on_browser_thread") as mock_dispatch:
        mock_dispatch.return_value = "Clicked: #btn\nsession_id: abc12345"
        result = run(action="click", session_id="abc12345", selector="#btn")

    assert "abc12345" in result


def test_browser_close_session():
    """action=close delegates to session manager."""
    from skills.browser.tool import run

    with patch("skills.browser.sessions.run_on_browser_thread") as mock_dispatch:
        mock_dispatch.return_value = "Session 'abc12345' closed.\nsession_id: abc12345"
        result = run(action="close", session_id="abc12345")

    assert "closed" in result.lower()
