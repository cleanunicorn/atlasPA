"""
skills/registry.py

Discovers and manages skills from two locations:

  Core skills   — skills/ directory (bundled with Atlas source, read-only)
  Addon skills  — ~/agent-files/skills/ (user-installed via the gateway)

Each skill is a folder containing:
    SKILL.md   — Description and usage (read by the LLM on demand)
    tool.py    — Python implementation with a `run(**kwargs) -> str` function
                 and an optional PARAMETERS dict (JSON Schema)

Design:
    - The registry builds a compact index (name + description) injected into
      the system prompt. The LLM sees only the index, NOT the full SKILL.md.
    - When the LLM calls a skill tool, the registry executes tool.py.
    - Addon skills can be installed / uninstalled at runtime via manage_skills.
    - reload() re-discovers both directories without restarting Atlas.
"""

import ast
import asyncio
import importlib.util
import logging
import shutil
from pathlib import Path
from providers.base import ToolDefinition
from paths import ADDON_SKILLS_DIR

logger = logging.getLogger(__name__)

CORE_SKILLS_DIR = Path(__file__).parent


class Skill:
    """A single skill loaded from a skills/<name>/ directory."""

    def __init__(self, name: str, description: str, path: Path, source: str = "core"):
        self.name = name
        self.description = description
        self.path = path
        self.source = source  # "core" or "addon"
        self._module = None
        self._load_module()

    def _load_module(self) -> None:
        """Load tool.py and cache the module. Safe to call at init."""
        tool_path = self.path / "tool.py"
        if not tool_path.exists():
            return
        try:
            spec = importlib.util.spec_from_file_location(
                f"skills.{self.name}", tool_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._module = module
        except Exception as e:
            logger.warning(f"Could not load {self.name}/tool.py: {e}")

    def load_skill_md(self) -> str:
        """Return full SKILL.md content (loaded on demand for context injection)."""
        skill_md = self.path / "SKILL.md"
        if skill_md.exists():
            return skill_md.read_text()
        return f"No SKILL.md found for {self.name}"

    async def run(self, **kwargs) -> str:
        """Execute this skill's run() function. Supports sync and async."""
        if self._module is None:
            return f"Error: {self.name}/tool.py not found or failed to load"
        module_run = getattr(self._module, "run", None)
        if module_run is None:
            return f"Error: skill '{self.name}' has no run() function"
        try:
            if asyncio.iscoroutinefunction(module_run):
                result = await module_run(**kwargs)
            else:
                result = await asyncio.to_thread(module_run, **kwargs)
            return str(result)
        except Exception as e:
            logger.exception(f"Skill '{self.name}' raised an error")
            return f"Error running skill '{self.name}': {e}"

    def to_tool_definition(self) -> ToolDefinition:
        """Convert this skill to a ToolDefinition the LLM can call."""
        default_schema = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input for this skill"}
            },
            "required": ["input"],
        }
        schema = default_schema
        if self._module and hasattr(self._module, "PARAMETERS"):
            schema = self._module.PARAMETERS

        return ToolDefinition(
            name=f"skill_{self.name}",
            description=self.description,
            parameters=schema,
        )


class SkillRegistry:
    """Auto-discovers and manages all skills in both the core and addon directories."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._discover()

    # ── Discovery ──────────────────────────────────────────────────────────────

    def _discover(self) -> None:
        """Scan core and addon directories and register all valid skills."""
        self._skills.clear()
        self._scan_dir(CORE_SKILLS_DIR, source="core")
        if ADDON_SKILLS_DIR.exists():
            self._scan_dir(ADDON_SKILLS_DIR, source="addon")

    def _scan_dir(self, directory: Path, source: str) -> None:
        for skill_dir in sorted(directory.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue

            skill_md = skill_dir / "SKILL.md"
            tool_py = skill_dir / "tool.py"

            if not skill_md.exists() and not tool_py.exists():
                continue

            description = f"Skill: {skill_dir.name}"
            if skill_md.exists():
                description = skill_md.read_text().strip()

            # Addon skills with the same name as a core skill are skipped
            name = skill_dir.name
            if name in self._skills:
                logger.warning(
                    f"Addon skill '{name}' conflicts with a core skill — skipped."
                )
                continue

            skill = Skill(
                name=name, description=description, path=skill_dir, source=source
            )
            self._skills[name] = skill
            logger.info(f"Registered {source} skill: {name}")

    def reload(self) -> None:
        """Re-discover both directories. Picks up newly installed addon skills."""
        logger.info("Reloading skill registry…")
        self._discover()
        logger.info(f"Skill registry reloaded — {len(self._skills)} skill(s) active.")

    # ── Install / uninstall ────────────────────────────────────────────────────

    def install(self, name: str, skill_md: str, tool_py: str) -> str:
        """
        Write an addon skill to ~/agent-files/skills/<name>/ and reload.

        Validates:
          - name is a Python identifier
          - tool_py parses without syntax errors
          - tool_py defines a top-level `run` function (sync or async)
        """
        name = name.strip().lower().replace(" ", "_").replace("-", "_")
        if not name or not name.isidentifier():
            return (
                f"Error: '{name}' is not a valid skill name. "
                "Use lowercase letters, digits, and underscores only."
            )

        if name in self._skills and self._skills[name].source == "core":
            return f"Error: '{name}' is a core skill and cannot be overwritten."

        if not tool_py.strip():
            return "Error: tool_py (the Python implementation) is required."

        # Syntax check
        try:
            tree = ast.parse(tool_py)
        except SyntaxError as e:
            return f"Error: tool.py has a syntax error: {e}"

        # Must have a run() function
        has_run = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
            for node in ast.walk(tree)
        )
        if not has_run:
            return (
                "Error: tool.py must define a `run` function "
                "(e.g. `async def run(**kwargs) -> str:`)."
            )

        skill_dir = ADDON_SKILLS_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill_md or f"# {name}\n\nAddon skill.")
        (skill_dir / "tool.py").write_text(tool_py)

        self.reload()
        return (
            f"✅ Addon skill '{name}' installed successfully.\n"
            f"   Saved to: {skill_dir}\n"
            f"   Call it as: skill_{name}"
        )

    def uninstall(self, name: str) -> str:
        """Delete an addon skill directory and reload the registry."""
        skill = self._skills.get(name)
        if skill is None:
            return f"Error: no skill named '{name}' is installed."
        if skill.source == "core":
            return f"Error: '{name}' is a core skill and cannot be uninstalled via manage_skills."

        shutil.rmtree(skill.path)
        self.reload()
        return f"✅ Addon skill '{name}' uninstalled and removed."

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_skills_summary(self, only: list[str] | None = None) -> str:
        """
        Compact index injected into the system prompt.
        Groups core and addon skills separately so the LLM knows what's user-installed.

        If *only* is provided, include only skills whose name is in the list.
        """
        skills = self._skills
        if only is not None:
            skills = {n: s for n, s in skills.items() if n in only}
        if not skills:
            return "_No skills loaded._"

        core = [(n, s) for n, s in skills.items() if s.source == "core"]
        addon = [(n, s) for n, s in skills.items() if s.source == "addon"]

        lines = []
        if core:
            lines.append("**Core skills:**")
            for name, skill in core:
                lines.append(f"  - **{name}**: {skill.description}")
        if addon:
            lines.append("**Addon skills** (user-installed, can be uninstalled):")
            for name, skill in addon:
                lines.append(f"  - **{name}**: {skill.description}")
        return "\n".join(lines)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return ToolDefinitions for all registered skills."""
        return [s.to_tool_definition() for s in self._skills.values()]

    def get_skill(self, name: str) -> Skill | None:
        """Look up a skill by name. Returns None if not found."""
        return self._skills.get(name)

    def all_skill_names(self) -> list[str]:
        """Return a sorted list of all registered skill names."""
        return sorted(self._skills.keys())
