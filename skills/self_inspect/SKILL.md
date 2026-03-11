# self_inspect

Read and understand Atlas's own source code, built-in tools, active configuration, and operational limits. Use this for self-awareness: understanding what you can do, how you work, and what constraints you operate under.

## Operations

- **overview** — Architecture map: key files, skill directories, execution flow
- **source** — Read any project source file by relative path. Omit `target` to list all Python files.
- **builtin_tools** — List all built-in agent tools with their full descriptions (parsed live from brain/engine.py)
- **config** — Show active environment configuration; secrets are redacted
- **skill_detail** — Show a skill's SKILL.md documentation and tool.py source
- **limits** — Show operational limits: loop cap, memory thresholds, history cap, sandbox rules

## When to use

- User asks "what can you do?", "how do you work?", "what are your limits?"
- Debugging unexpected behaviour (check source or config)
- Before installing a new skill, review existing ones for patterns
- Understanding which tools are available and what they accept

## Input

- `operation` (string, required): one of the operations listed above
- `target` (string, optional): file path for `source`, skill name for `skill_detail`

## Output

Markdown-formatted text describing the requested aspect of the agent.
