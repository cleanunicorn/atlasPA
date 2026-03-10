"""
tests/test_skill_manager.py

Tests for the addon skill management system:
  - SkillRegistry.install() / uninstall() / reload()
  - manage_skills built-in tool in Brain
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── SkillRegistry fixtures ─────────────────────────────────────────────────────

MINIMAL_TOOL_PY = """\
PARAMETERS = {"type": "object", "properties": {"msg": {"type": "string"}}, "required": []}

async def run(msg: str = "hello", **_) -> str:
    return f"echo: {msg}"
"""

MINIMAL_SKILL_MD = "# echo\n\nEchoes back the input message."


def make_registry(tmp_path: Path):
    """Return a SkillRegistry that only scans tmp_path as both core and addon dir."""
    from skills.registry import SkillRegistry, CORE_SKILLS_DIR, ADDON_SKILLS_DIR
    registry = SkillRegistry.__new__(SkillRegistry)
    registry._skills = {}

    addon_dir = tmp_path / "addon_skills"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core_skills"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core_skills").mkdir()
        registry._discover()

    return registry, addon_dir


# ── install ────────────────────────────────────────────────────────────────────

def test_install_writes_files(tmp_path):
    from skills.registry import SkillRegistry

    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core").mkdir()
        registry = SkillRegistry()
        result = registry.install("echo", MINIMAL_SKILL_MD, MINIMAL_TOOL_PY)

    assert "✅" in result
    assert "echo" in result
    assert (addon_dir / "echo" / "tool.py").exists()
    assert (addon_dir / "echo" / "SKILL.md").exists()
    assert "echo" in registry.all_skill_names()


def test_install_skill_immediately_available(tmp_path):
    from skills.registry import SkillRegistry

    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core").mkdir()
        registry = SkillRegistry()
        registry.install("echo", MINIMAL_SKILL_MD, MINIMAL_TOOL_PY)
        skill = registry.get_skill("echo")

    assert skill is not None
    assert skill.source == "addon"


def test_install_invalid_name(tmp_path):
    from skills.registry import SkillRegistry

    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core").mkdir()
        registry = SkillRegistry()
        result = registry.install("my-skill!", MINIMAL_SKILL_MD, MINIMAL_TOOL_PY)

    assert "Error" in result


def test_install_syntax_error(tmp_path):
    from skills.registry import SkillRegistry

    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core").mkdir()
        registry = SkillRegistry()
        result = registry.install("bad_skill", MINIMAL_SKILL_MD, "def run(: pass")

    assert "syntax error" in result.lower()


def test_install_missing_run_function(tmp_path):
    from skills.registry import SkillRegistry

    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core").mkdir()
        registry = SkillRegistry()
        result = registry.install("no_run", MINIMAL_SKILL_MD, "x = 1 + 1")

    assert "Error" in result
    assert "run" in result


def test_install_cannot_overwrite_core(tmp_path):
    from skills.registry import SkillRegistry

    core_dir = tmp_path / "core"
    core_dir.mkdir()
    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    # Create a core skill named "echo"
    core_echo = core_dir / "echo"
    core_echo.mkdir()
    (core_echo / "SKILL.md").write_text("# echo\n\nCore skill.")
    (core_echo / "tool.py").write_text("async def run(**_): return 'core'")

    with (
        patch("skills.registry.CORE_SKILLS_DIR", core_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        registry = SkillRegistry()
        result = registry.install("echo", MINIMAL_SKILL_MD, MINIMAL_TOOL_PY)

    assert "Error" in result
    assert "core" in result.lower()


# ── uninstall ──────────────────────────────────────────────────────────────────

def test_uninstall_removes_skill(tmp_path):
    from skills.registry import SkillRegistry

    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core").mkdir()
        registry = SkillRegistry()
        registry.install("echo", MINIMAL_SKILL_MD, MINIMAL_TOOL_PY)
        assert "echo" in registry.all_skill_names()

        result = registry.uninstall("echo")

    assert "✅" in result
    assert "echo" not in registry.all_skill_names()
    assert not (addon_dir / "echo").exists()


def test_uninstall_unknown_skill(tmp_path):
    from skills.registry import SkillRegistry

    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    with (
        patch("skills.registry.CORE_SKILLS_DIR", tmp_path / "core"),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        (tmp_path / "core").mkdir()
        registry = SkillRegistry()
        result = registry.uninstall("nonexistent")

    assert "Error" in result


def test_uninstall_core_skill_blocked(tmp_path):
    from skills.registry import SkillRegistry

    core_dir = tmp_path / "core"
    core_dir.mkdir()
    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    core_skill = core_dir / "mysearch"
    core_skill.mkdir()
    (core_skill / "SKILL.md").write_text("# mysearch\n\nCore skill.")
    (core_skill / "tool.py").write_text("async def run(**_): return 'results'")

    with (
        patch("skills.registry.CORE_SKILLS_DIR", core_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        registry = SkillRegistry()
        result = registry.uninstall("mysearch")

    assert "Error" in result
    assert "core" in result.lower()


# ── get_skills_summary ─────────────────────────────────────────────────────────

def test_skills_summary_groups_by_source(tmp_path):
    from skills.registry import SkillRegistry

    core_dir = tmp_path / "core"
    core_dir.mkdir()
    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()

    # Core skill
    core_s = core_dir / "search"
    core_s.mkdir()
    (core_s / "SKILL.md").write_text("# search\n\nSearch the web.")
    (core_s / "tool.py").write_text("async def run(**_): return ''")

    with (
        patch("skills.registry.CORE_SKILLS_DIR", core_dir),
        patch("skills.registry.ADDON_SKILLS_DIR", addon_dir),
    ):
        registry = SkillRegistry()
        registry.install("echo", MINIMAL_SKILL_MD, MINIMAL_TOOL_PY)
        summary = registry.get_skills_summary()

    assert "Core skills" in summary
    assert "Addon skills" in summary
    assert "search" in summary
    assert "echo" in summary


# ── manage_skills built-in in Brain ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_brain_manage_skills_install():
    from brain.engine import Brain
    from providers.base import ToolCall
    from unittest.mock import MagicMock

    mock_skills = MagicMock()
    mock_skills.install.return_value = "✅ Addon skill 'echo' installed successfully."

    brain = Brain(provider=MagicMock(), memory=MagicMock(), skills=mock_skills)
    tc = ToolCall(id="1", name="manage_skills", arguments={
        "action": "install",
        "name": "echo",
        "skill_md": MINIMAL_SKILL_MD,
        "tool_py": MINIMAL_TOOL_PY,
    })
    result = await brain._execute_tool(tc)

    mock_skills.install.assert_called_once_with("echo", MINIMAL_SKILL_MD, MINIMAL_TOOL_PY)
    assert "✅" in result


@pytest.mark.asyncio
async def test_brain_manage_skills_uninstall():
    from brain.engine import Brain
    from providers.base import ToolCall

    mock_skills = MagicMock()
    mock_skills.uninstall.return_value = "✅ Addon skill 'echo' uninstalled."

    brain = Brain(provider=MagicMock(), memory=MagicMock(), skills=mock_skills)
    tc = ToolCall(id="2", name="manage_skills", arguments={"action": "uninstall", "name": "echo"})
    result = await brain._execute_tool(tc)

    mock_skills.uninstall.assert_called_once_with("echo")
    assert "✅" in result


@pytest.mark.asyncio
async def test_brain_manage_skills_list():
    from brain.engine import Brain
    from providers.base import ToolCall

    mock_skills = MagicMock()
    mock_skills.get_skills_summary.return_value = "Core skills:\n  - search"

    brain = Brain(provider=MagicMock(), memory=MagicMock(), skills=mock_skills)
    tc = ToolCall(id="3", name="manage_skills", arguments={"action": "list"})
    result = await brain._execute_tool(tc)

    mock_skills.get_skills_summary.assert_called_once()
    assert "search" in result


@pytest.mark.asyncio
async def test_brain_manage_skills_missing_name():
    from brain.engine import Brain
    from providers.base import ToolCall

    brain = Brain(provider=MagicMock(), memory=MagicMock(), skills=MagicMock())
    tc = ToolCall(id="4", name="manage_skills", arguments={"action": "install"})
    result = await brain._execute_tool(tc)
    assert "Error" in result


@pytest.mark.asyncio
async def test_brain_manage_skills_unknown_action():
    from brain.engine import Brain
    from providers.base import ToolCall

    brain = Brain(provider=MagicMock(), memory=MagicMock(), skills=MagicMock())
    tc = ToolCall(id="5", name="manage_skills", arguments={"action": "explode"})
    result = await brain._execute_tool(tc)
    assert "Unknown" in result
