"""
skills/http_request/tool.py

Generic HTTP requests via httpx (async, follows redirects).
"""

PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The full URL to request (include https://)",
        },
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
            "description": "HTTP method (default: GET)",
            "default": "GET",
        },
        "headers": {
            "type": "object",
            "description": "HTTP headers as key-value pairs",
            "additionalProperties": {"type": "string"},
        },
        "body": {
            "type": "string",
            "description": "Request body string (for POST/PUT/PATCH). Pass JSON as a string.",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default: 15)",
            "default": 15,
        },
    },
    "required": ["url"],
}

MAX_BODY_CHARS = 10_000


async def run(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: str = "",
    timeout: int = 15,
    **kwargs,
) -> str:
    """
    Make an HTTP request and return a formatted response.

    Args:
        url:     Target URL.
        method:  HTTP method string (case-insensitive).
        headers: Optional request headers.
        body:    Optional request body string.
        timeout: Seconds before giving up.

    Returns:
        Formatted string with status, selected headers, and body. Never raises.
    """
    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed. Run: uv add httpx"

    method = method.upper()
    headers = headers or {}
    timeout = min(int(timeout), 60)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            request_kwargs = {"headers": headers}
            if body:
                request_kwargs["content"] = body.encode()

            response = await client.request(method, url, **request_kwargs)

        # Format output
        content_type = response.headers.get("content-type", "")
        body_text = response.text

        if len(body_text) > MAX_BODY_CHARS:
            body_text = (
                body_text[:MAX_BODY_CHARS]
                + f"\n\n... [truncated at {MAX_BODY_CHARS} chars]"
            )

        lines = [
            f"HTTP {response.status_code} {response.reason_phrase}",
            f"URL: {response.url}",
            f"Content-Type: {content_type}",
            "",
            body_text,
        ]
        return "\n".join(lines)

    except httpx.TimeoutException:
        return f"Error: request timed out after {timeout}s — {url}"
    except httpx.RequestError as e:
        return f"Error: request failed — {e}"
    except Exception as e:
        return f"Error: {e}"
