"""
channels/telegram/formatting.py

Convert standard Markdown to Telegram HTML.

Telegram HTML supports: <b>, <i>, <u>, <s>, <code>, <pre>, <a>
Special chars &, <, > must be escaped as HTML entities.

Usage:
    from channels.telegram.formatting import md_to_html
    await update.message.reply_text(md_to_html(text), parse_mode="HTML")
"""

import html
import re


def md_to_html(text: str) -> str:
    """
    Convert Markdown-formatted text to Telegram HTML.

    Handles: bold, italic, strikethrough, inline code, fenced code blocks,
    and headings. Everything else is passed through with HTML chars escaped.
    """
    # ── Step 1: Extract fenced code blocks before any other processing ──────
    code_blocks: list[str] = []

    def _save_code_block(m: re.Match) -> str:
        lang = m.group(1).strip()
        code = html.escape(m.group(2))
        if lang:
            rendered = (
                f'<pre><code class="language-{html.escape(lang)}">{code}</code></pre>'
            )
        else:
            rendered = f"<pre>{code}</pre>"
        code_blocks.append(rendered)
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```(\w*)\n?(.*?)```", _save_code_block, text, flags=re.DOTALL)

    # ── Step 2: Extract inline code ──────────────────────────────────────────
    inline_codes: list[str] = []

    def _save_inline_code(m: re.Match) -> str:
        inline_codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`\n]+)`", _save_inline_code, text)

    # ── Step 3: Escape remaining HTML special characters ────────────────────
    text = html.escape(text)

    # ── Step 4: Apply Markdown → HTML conversions ────────────────────────────
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text, flags=re.DOTALL)

    # Italic: *text* (single asterisk, not bold)
    text = re.sub(r"\*([^\*\n]+)\*", r"<i>\1</i>", text)

    # Italic: _text_ (underscores, word-boundary aware to avoid snake_case)
    text = re.sub(r"(?<![_\w])_([^_\n]+)_(?![_\w])", r"<i>\1</i>", text)

    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text, flags=re.DOTALL)

    # Headings: # / ## / ### → bold (Telegram has no heading element)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # ── Step 5: Restore extracted blocks ─────────────────────────────────────
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CB{i}\x00", block)
    for i, code in enumerate(inline_codes):
        text = text.replace(f"\x00IC{i}\x00", code)

    return text
