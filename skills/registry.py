"""
skills/registry.py

Discovers skills from the skills/ directory.
Each skill is a folder containing:
    SKILL.md   — Description and usage (read by the LLM on demand)
    tool.py    — Python implementation with a `run(**kwargs) -> str` function
                 and an optional PARAMETERS dict (JSON Schema)

Design (mirrors OpenClaw):
    - The registry builds a compact index (name + description) injected into
      the system prompt. The LLM sees only the index, NOT the full SKILL.md.
    - When the LLM calls a skill tool, the registry executes tool.py.
    - This keeps the context window lean. Skill details are read on demand.
"""

import asyncio
import importlib.util
import logging
from pathlib import Path
from providers.base import ToolDefinition

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent


class Skill:
    """A single skill loaded from a skills/<name>/ directory."""

    def __init__(self, name: str, description: str, path: Path):
        self.name = name
        self.description = description
        self.path = path
        self._module = None
        # Eagerly load the module to extract PARAMETERS schema
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
        try:
            result = self._module.run(**kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return str(result)
        except Exception as e:
            logger.exception(f"Skill '{self.name}' raised an error")
            return f"Error running skill '{self.name}': {e}"

    def to_tool_definition(self) -> ToolDefinition:
        """Convert this skill to a ToolDefinition the LLM can call."""
        # Use PARAMETERS from tool.py if defined, otherwise use a generic schema
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
    """Auto-discovers and manages all skills in the skills/ directory."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._discover()

    def _discover(self) -> None:
        """Scan the skills/ directory and register all valid skills."""
        for skill_dir in sorted(SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            if skill_dir.name.startswith("_"):
                continue

            skill_md = skill_dir / "SKILL.md"
            tool_py = skill_dir / "tool.py"

            if not skill_md.exists() and not tool_py.exists():
                continue

            # Extract description from first non-header line of SKILL.md
            description = f"Skill: {skill_dir.name}"
            if skill_md.exists():
                for line in skill_md.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        description = line[:200]
                        break

            skill = Skill(name=skill_dir.name, description=description, path=skill_dir)
            self._skills[skill_dir.name] = skill
            logger.info(f"Registered skill: {skill_dir.name}")

    def get_skills_summary(self) -> str:
        """
        Compact index of all skills injected into the system prompt.
        The LLM sees name + one-line description, not the full SKILL.md.
        """
        if not self._skills:
            return "_No skills loaded._"
        lines = [f"- **{name}**: {skill.description}" for name, skill in self._skills.items()]
        return "\n".join(lines)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return ToolDefinitions for all registered skills."""
        return [s.to_tool_definition() for s in self._skills.values()]

    def get_skill(self, name: str) -> Skill | None:
        """Look up a skill by name. Returns None if not found."""
        return self._skills.get(name)

    def all_skill_names(self) -> list[str]:
        """Return a sorted list of all registered skill names."""
        return list(self._skills.keys())
