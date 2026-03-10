"""
tests/test_formatting.py

Tests for Telegram HTML formatter.
"""

from channels.telegram.formatting import md_to_html


def test_bold_asterisks():
    assert md_to_html("**hello**") == "<b>hello</b>"


def test_bold_underscores():
    assert md_to_html("__hello__") == "<b>hello</b>"


def test_italic_asterisk():
    assert md_to_html("*hello*") == "<i>hello</i>"


def test_italic_underscore():
    assert md_to_html("_hello_") == "<i>hello</i>"


def test_strikethrough():
    assert md_to_html("~~gone~~") == "<s>gone</s>"


def test_inline_code():
    assert md_to_html("`code`") == "<code>code</code>"


def test_fenced_code_block():
    result = md_to_html("```\nprint('hi')\n```")
    assert "<pre>" in result
    assert "print(&#x27;hi&#x27;)" in result  # html.escape applied inside


def test_fenced_code_block_with_lang():
    result = md_to_html("```python\nx = 1\n```")
    assert 'class="language-python"' in result
    assert "x = 1" in result


def test_heading():
    assert md_to_html("# Title") == "<b>Title</b>"
    assert md_to_html("## Sub") == "<b>Sub</b>"


def test_html_special_chars_escaped():
    result = md_to_html("a < b & c > d")
    assert "&lt;" in result
    assert "&amp;" in result
    assert "&gt;" in result


def test_html_not_escaped_inside_code():
    result = md_to_html("`a < b`")
    assert "<code>a &lt; b</code>" == result


def test_snake_case_not_italicised():
    # Underscores in identifiers should NOT become italic
    result = md_to_html("use the_function_name here")
    assert "<i>" not in result


def test_plain_text_unchanged():
    assert md_to_html("Hello world") == "Hello world"


def test_mixed():
    result = md_to_html("**Bold** and *italic* and `code`")
    assert "<b>Bold</b>" in result
    assert "<i>italic</i>" in result
    assert "<code>code</code>" in result
