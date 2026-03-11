"""
skills/self_inspect/tool.py

Self-inspection skill — lets the agent read and understand its own source code,
active configuration, built-in tools, and operational limits.

Operations:
    overview      — Architecture map (key files, skill dirs, execution flow)
    source        — Read a project source file; list all .py files if no target given
    builtin_tools — Parse brain/engine.py via AST to list built-in tools + descriptions
    config        — Show active env config with secrets redacted
    skill_detail  — Show a skill's SKILL.md + tool.py
    limits        — Show loop cap, memory thresholds, history cap, sandbox rules
"""

import ast
import os
import re
from pathlib import Path

# Project root: skills/self_inspect/ -> skills/ -> root
_ROOT = Path(__file__).parent.parent.parent.resolve()

_ALLOWED_EXTS = {".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml"}

# Relative paths that must never be read (contain secrets)
_BLOCKED = {"config/.env", "config/credentials.json", ".env"}

# Skip these during directory listings
_SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".git", ".mypy_cache"}

PARAMETERS = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["overview", "source", "builtin_tools", "config", "skill_detail", "limits"],
            "description": (
                "overview: Architecture map and file index. "
                "source: Read a source file (or list all .py files if no target). "
                "builtin_tools: List all built-in agent tools with full descriptions. "
                "config: Show active configuration with secrets redacted. "
                "skill_detail: Show a skill's SKILL.md and tool.py source. "
                "limits: Show operational limits and thresholds."
            ),
        },
        "target": {
            "type": "string",
            "description": (
                "For 'source': relative path from project root (e.g. 'brain/engine.py'). "
                "For 'skill_detail': skill name (e.g. 'web_search'). "
                "Omit for other operations."
            ),
        },
    },
    "required": ["operation"],
}


def run(operation: str, target: str = "", **kwargs) -> str:
    handlers = {
        "overview": _overview,
        "source": _source,
        "builtin_tools": _builtin_tools,
        "config": _config,
        "skill_detail": _skill_detail,
        "limits": _limits,
    }
    handler = handlers.get(operation)
    if not handler:
        return f"Unknown operation '{operation}'. Valid: {', '.join(handlers)}"
    try:
        return handler(target.strip() if target else "")
    except Exception as exc:
        return f"Error in self_inspect.{operation}: {exc}"


# ── Operations ────────────────────────────────────────────────────────────────


def _overview(_: str) -> str:
    lines = ["# Atlas — Architecture Overview", f"Root: {_ROOT}", ""]

    key_files = [
        ("gateway.py",              "Legacy entry point — orchestrates all components"),
        ("main.py",                 "Unified CLI (run / logs / setup commands)"),
        ("brain/engine.py",         "ReAct reasoning loop (Brain class, MAX_ITERATIONS=10)"),
        ("providers/",              "LLM abstraction (anthropic / openai / ollama / openrouter)"),
        ("providers/base.py",       "Shared types: Message, ToolDefinition, ToolCall, LLMResponse"),
        ("memory/store.py",         "Persistent markdown memory (soul.md, context.md)"),
        ("memory/history.py",       "Per-user conversation history (JSON, capped at 200 msgs)"),
        ("memory/retriever.py",     "Keyword-based context relevance filtering"),
        ("memory/summariser.py",    "LLM-assisted context compression (triggered at >20 entries)"),
        ("skills/registry.py",      "Auto-discovery and management of core + addon skills"),
        ("heartbeat/scheduler.py",  "Cron-based background job scheduler"),
        ("channels/cli/bot.py",     "Terminal interface"),
        ("channels/telegram/bot.py","Telegram bot"),
        ("channels/discord/bot.py", "Discord bot"),
        ("channels/web/bot.py",     "Gradio web UI"),
    ]

    lines.append("## Key Files")
    for rel, desc in key_files:
        exists = "✅" if (_ROOT / rel).exists() else "❌"
        lines.append(f"  {exists} {rel} — {desc}")
    lines.append("")

    # Skill directories
    core_dir = _ROOT / "skills"
    addon_dir = Path.home() / "agent-files" / "skills"
    lines.append("## Skill Directories")
    lines.append(f"  Core:  {core_dir}")
    lines.append(f"  Addon: {addon_dir} ({'exists' if addon_dir.exists() else 'not created yet'})")

    core_skills = _list_skill_names(core_dir)
    addon_skills = _list_skill_names(addon_dir)
    if core_skills:
        lines.append(f"  Core  ({len(core_skills)}): {', '.join(core_skills)}")
    if addon_skills:
        lines.append(f"  Addon ({len(addon_skills)}): {', '.join(addon_skills)}")
    lines.append("")

    lines.append("## Execution Flow")
    lines.append("  User message → Channel → brain.think()")
    lines.append("    → build system prompt (soul + relevant memory + skills index)")
    lines.append("    → LLM call with all tools")
    lines.append("    → if tool calls: execute concurrently → loop (max 10 times)")
    lines.append("    → if final text: return to channel → user")
    lines.append("")
    lines.append("## Memory Pipeline")
    lines.append("  soul.md (identity) + context.md (user facts, auto-summarised at 20 entries)")
    lines.append("  Relevance filtering: top-15 context entries injected per query")
    lines.append("  Conversation history: per-user JSON, capped at 200 messages")

    return "\n".join(lines)


def _source(target: str) -> str:
    if not target:
        # List all Python source files in the project
        lines = [f"Project source files under {_ROOT}:", "(provide 'target' to read a file)", ""]
        for p in sorted(_ROOT.rglob("*.py")):
            parts = p.relative_to(_ROOT).parts
            if any(part in _SKIP_DIRS or part.startswith(".") for part in parts):
                continue
            lines.append(f"  {p.relative_to(_ROOT)}")
        return "\n".join(lines)

    # Resolve and security-check the path
    try:
        resolved = (_ROOT / target).resolve()
    except Exception as exc:
        return f"Invalid path: {exc}"

    try:
        rel = resolved.relative_to(_ROOT)
    except ValueError:
        return "Access denied: path escapes the project root."

    rel_str = str(rel).replace("\\", "/")
    if rel_str in _BLOCKED or any(rel_str.startswith(b) for b in _BLOCKED):
        return f"Access denied: '{rel_str}' may contain secrets."

    if resolved.suffix not in _ALLOWED_EXTS:
        return f"Cannot read '{resolved.suffix}' files. Allowed: {', '.join(sorted(_ALLOWED_EXTS))}"

    if not resolved.exists():
        return f"File not found: {target}"
    if not resolved.is_file():
        return f"Not a file: {target}"

    content = resolved.read_text(encoding="utf-8", errors="replace")
    line_list = content.splitlines()
    size_kb = len(content) / 1024
    header = f"# {rel} ({len(line_list)} lines, {size_kb:.1f} KB)\n\n"

    MAX_LINES = 400
    if len(line_list) > MAX_LINES:
        content = "\n".join(line_list[:MAX_LINES])
        content += f"\n\n[... {len(line_list) - MAX_LINES} more lines — use a narrower target or request a specific section]"

    return header + content


def _builtin_tools(_: str) -> str:
    engine_path = _ROOT / "brain" / "engine.py"
    if not engine_path.exists():
        return "Error: brain/engine.py not found."

    source = engine_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"Failed to parse brain/engine.py: {exc}"

    tools: list[tuple[str, str, dict]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ToolDefinition"):
            continue
        name = desc = None
        params: dict = {}
        for kw in node.keywords:
            if kw.arg == "name":
                try:
                    name = ast.literal_eval(kw.value)
                except Exception:
                    pass
            elif kw.arg == "description":
                try:
                    desc = ast.literal_eval(kw.value)
                except Exception:
                    pass
            elif kw.arg == "parameters":
                try:
                    params = ast.literal_eval(kw.value)
                except Exception:
                    pass
        if name:
            tools.append((name, desc or "", params))

    if not tools:
        return "Could not extract ToolDefinition entries from brain/engine.py."

    lines = [f"# Built-in Agent Tools ({len(tools)} total)", ""]
    lines.append("These are always available, regardless of installed skills:")
    lines.append("")

    for name, desc, params in tools:
        lines.append(f"**{name}**")
        if desc:
            lines.append(f"  {desc}")
        # Show required parameters
        required = params.get("required", [])
        props = params.get("properties", {})
        if props:
            param_parts = []
            for pname, pdef in props.items():
                req = "*" if pname in required else ""
                ptype = pdef.get("type", "any")
                param_parts.append(f"{pname}{req}: {ptype}")
            lines.append(f"  Parameters: {', '.join(param_parts)}  (* = required)")
        lines.append("")

    return "\n".join(lines)


def _config(_: str) -> str:
    SECRET_PATTERNS = {"api_key", "token", "secret", "password", "credential", "webhook"}

    atlas_keys = [
        ("LLM_PROVIDER",               "anthropic"),
        ("ANTHROPIC_MODEL",            "(provider default)"),
        ("OPENAI_MODEL",               "(provider default)"),
        ("OLLAMA_BASE_URL",            "http://localhost:11434/v1"),
        ("OLLAMA_MODEL",               "(not set)"),
        ("AGENT_NAME",                 "Atlas"),
        ("AGENT_TIMEZONE",             "UTC"),
        ("WEB_HOST",                   "0.0.0.0"),
        ("WEB_PORT",                   "7860"),
        ("LOG_LEVEL",                  "INFO"),
        ("LLM_LOG_FILE",               "(auto: logs/llm.jsonl)"),
        ("CONTEXT_MAX_INJECTED",       "15"),
        ("CONTEXT_SUMMARY_THRESHOLD",  "20"),
        ("HISTORY_MAX_MESSAGES",       "200"),
    ]

    lines = ["# Active Configuration", ""]
    lines.append("## Environment Variables (secrets redacted)")
    for key, default in atlas_keys:
        val = os.environ.get(key)
        if val is None:
            lines.append(f"  {key}: (not set, default: {default})")
        else:
            lines.append(f"  {key}: {val}")

    # Report which secret keys are set, without values
    secret_set = sorted(
        k for k in os.environ
        if any(s in k.lower() for s in SECRET_PATTERNS) and os.environ[k]
    )
    if secret_set:
        lines.append("")
        lines.append("## Secrets present (values hidden)")
        for key in secret_set:
            lines.append(f"  {key}: [SET]")

    lines.append("")
    lines.append("## Memory Files")
    memory_dir = _ROOT / "memory"
    for fname in ("soul.md", "context.md", "location.md"):
        fpath = memory_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            content_lines = fpath.read_text(encoding="utf-8", errors="replace").count("\n")
            lines.append(f"  {fname}: {size} bytes, {content_lines} lines")
        else:
            lines.append(f"  {fname}: not found")

    lines.append("")
    lines.append("## LLM Log")
    log_path = _ROOT / "logs" / "llm.jsonl"
    if log_path.exists():
        size_mb = log_path.stat().st_size / 1024 / 1024
        lines.append(f"  logs/llm.jsonl: {size_mb:.2f} MB")
    else:
        lines.append("  logs/llm.jsonl: not found (or LLM_LOG_FILE=off)")

    return "\n".join(lines)


def _skill_detail(target: str) -> str:
    if not target:
        return "Provide a skill name as 'target' (e.g. target='web_search')."

    search_dirs = [
        _ROOT / "skills",
        Path.home() / "agent-files" / "skills",
    ]

    skill_dir = None
    for base in search_dirs:
        candidate = base / target
        if candidate.is_dir() and (candidate / "tool.py").exists():
            skill_dir = candidate
            break

    if not skill_dir:
        available = _list_skill_names(_ROOT / "skills")
        return (
            f"Skill '{target}' not found.\n"
            f"Available core skills: {', '.join(available)}"
        )

    lines = [f"# Skill: {target}", f"Location: {skill_dir}", ""]

    skill_md = skill_dir / "SKILL.md"
    tool_py = skill_dir / "tool.py"

    if skill_md.exists():
        lines.append("## SKILL.md")
        lines.append(skill_md.read_text(encoding="utf-8", errors="replace"))
        lines.append("")

    if tool_py.exists():
        lines.append("## tool.py")
        lines.append(tool_py.read_text(encoding="utf-8", errors="replace"))

    return "\n".join(lines)


def _limits(_: str) -> str:
    lines = ["# Operational Limits", ""]

    lines.append("## ReAct Loop")
    max_iter = _extract_constant(_ROOT / "brain" / "engine.py", "MAX_ITERATIONS")
    lines.append(f"  Max iterations per turn: {max_iter or '10 (default)'}")
    lines.append("  Each iteration: one LLM call + all resulting tool calls (parallel)")
    lines.append("  Fallback if limit hit: fixed error message returned to user")
    lines.append("")

    lines.append("## Memory")
    ctx_injected = _extract_constant(_ROOT / "memory" / "store.py", "CONTEXT_MAX_INJECTED")
    ctx_threshold = _extract_constant(_ROOT / "memory" / "summariser.py", "CONTEXT_SUMMARY_THRESHOLD")
    hist_max = _extract_constant(_ROOT / "memory" / "history.py", "HISTORY_MAX_MESSAGES")
    lines.append(f"  Max context entries injected per query: {ctx_injected or '15 (default)'}")
    lines.append(f"  Context summarisation threshold: {ctx_threshold or '20 (default)'} entries")
    lines.append(f"  Conversation history cap: {hist_max or '200 (default)'} messages per user")
    lines.append("  History images: base64 data stripped on save (replaced with '[image attached]')")
    lines.append("")

    lines.append("## Source Reading (this skill)")
    lines.append("  Scope: project root only — no path traversal")
    lines.append(f"  Blocked: {', '.join(sorted(_BLOCKED))}")
    lines.append(f"  Allowed extensions: {', '.join(sorted(_ALLOWED_EXTS))}")
    lines.append("  Max lines shown: 400 (truncated beyond that)")
    lines.append("")

    lines.append("## file_ops Skill Sandbox")
    lines.append("  Read/write scoped to: ~/agent-files/")
    lines.append("  Path traversal (../) blocked")
    lines.append("")

    lines.append("## Skill System")
    core_skills = _list_skill_names(_ROOT / "skills")
    lines.append(f"  Core skills (bundled, read-only): {len(core_skills)} — {', '.join(core_skills)}")
    lines.append("  Addon skills: ~/agent-files/skills/ (install via manage_skills)")
    lines.append("  Skill tool names: prefixed with 'skill_' (e.g. skill_web_search)")
    lines.append("  New skills callable immediately after install (no restart needed)")
    lines.append("")

    lines.append("## Provider")
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    lines.append(f"  Active: {provider}")
    if provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        lines.append(f"  Model: {model}")
    elif provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        lines.append(f"  Model: {model}")
    elif provider == "ollama":
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = os.environ.get("OLLAMA_MODEL", "(not set)")
        lines.append(f"  Base URL: {base}  Model: {model}")
    elif provider == "openrouter":
        lines.append("  Routed via OpenRouter (see OPENAI_MODEL for model)")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _list_skill_names(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(
        d.name for d in directory.iterdir()
        if d.is_dir() and (d / "tool.py").exists()
    )


def _extract_constant(path: Path, name: str) -> str | None:
    """Extract a top-level integer/string constant from a Python source file."""
    if not path.exists():
        return None
    source = path.read_text(encoding="utf-8", errors="replace")
    # Match: NAME = value (possibly from os.getenv with default)
    m = re.search(
        rf'{re.escape(name)}\s*=\s*(?:int\s*\(\s*os\.getenv\s*\([^,]+,\s*"?(\d+)"?\s*\)\s*\)|(\d+))',
        source,
    )
    if m:
        return m.group(1) or m.group(2)
    return None
