"""
skills/web_search/tool.py

DuckDuckGo instant answers — no API key required.
Uses the duckduckgo-search library for text search.
"""

PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to look up on the web",
        }
    },
    "required": ["query"],
}


def run(query: str, **kwargs) -> str:
    """
    Search DuckDuckGo and return a text summary of the top results.

    Args:
        query: The search query.

    Returns:
        Formatted string with titles, snippets, and URLs of top results.
        Returns an error message string on failure (never raises).
    """
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r)

        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No snippet")
            href = r.get("href", "")
            lines.append(f"{i}. **{title}**")
            lines.append(f"   {body}")
            if href:
                lines.append(f"   URL: {href}")
            lines.append("")

        return "\n".join(lines).strip()

    except ImportError:
        return "Error: duckduckgo-search package not installed. Run: uv add duckduckgo-search"
    except Exception as e:
        return f"Web search failed: {e}"
