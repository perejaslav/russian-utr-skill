# -*- coding: utf-8 -*-
"""Tests for scripts/install.py and for Agent Skills format compliance.

Run:
    python -m pytest tests/test_install.py -q
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "scripts" / "install.py"


def _load_installer():
    spec = importlib.util.spec_from_file_location("utr_install", INSTALLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["utr_install"] = module
    spec.loader.exec_module(module)
    return module


inst = _load_installer()


# ---------------------------------------------------------------------------
# Agent Skills specification (https://agentskills.io/specification)
# ---------------------------------------------------------------------------

ALLOWED_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _frontmatter() -> dict[str, object]:
    """Parse the flat YAML subset used by SKILL.md without a YAML library."""
    text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md must start with YAML frontmatter"
    data: dict[str, object] = {}
    current = None
    for line in match.group(1).splitlines():
        if line.startswith("  ") and current is not None:
            key, _, value = line.strip().partition(":")
            data.setdefault(current, {})[key] = value.strip().strip('"')  # type: ignore[union-attr]
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        current = key if not value else None
        data[key] = value.strip('"') if value else {}
    return data


def test_frontmatter_uses_only_spec_keys():
    assert set(_frontmatter()) <= ALLOWED_KEYS


def test_name_follows_spec():
    name = _frontmatter()["name"]
    assert isinstance(name, str)
    assert 1 <= len(name) <= 64
    assert NAME_PATTERN.match(name)
    assert name == inst.SKILL_NAME


def test_description_fits_limit():
    description = _frontmatter()["description"]
    assert isinstance(description, str)
    assert 1 <= len(description) <= 1024


def test_compatibility_fits_limit():
    compatibility = _frontmatter().get("compatibility", "x")
    assert isinstance(compatibility, str)
    assert 1 <= len(compatibility) <= 500


def test_metadata_is_string_map():
    metadata = _frontmatter().get("metadata", {})
    assert isinstance(metadata, dict)
    assert all(isinstance(value, str) for value in metadata.values())


def test_skill_body_is_harness_neutral():
    text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    for marker in ("Claude", ".claude", "~/.codex", "Bash tool", "Read tool"):
        assert marker not in text, marker


def test_skill_references_exist():
    text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    for path in re.findall(r"`((?:references|scripts|examples)/[^`\s]+)`", text):
        assert (ROOT / path).exists(), path


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


def test_default_installs_shared_and_claude_dirs(tmp_path, capsys):
    assert inst.main(["--home", str(tmp_path)]) == 0
    for root in (".agents/skills", ".claude/skills"):
        target = tmp_path / root / "russian-utr-skill"
        assert (target / "SKILL.md").is_file()
        assert (target / "scripts" / "utr-lint.py").is_file()
        assert (target / "references" / "writing-rules.md").is_file()
        assert not (target / "tests").exists()
        assert not (target / ".git").exists()


def test_all_deduplicates_targets(tmp_path):
    found = inst.targets(list(inst.HARNESSES), tmp_path, None)
    assert len(found) == len(set(found))
    assert tmp_path / ".agents/skills/russian-utr-skill" in found


def test_project_scope(tmp_path, capsys):
    assert inst.main(["--harness", "opencode", "--project", str(tmp_path), "--home", "/nonexistent"]) == 0
    assert (tmp_path / ".opencode/skills/russian-utr-skill/SKILL.md").is_file()


def test_existing_target_needs_force(tmp_path, capsys):
    args = ["--harness", "pi", "--home", str(tmp_path)]
    assert inst.main(args) == 0
    assert inst.main(args) == 1
    assert inst.main([*args, "--force"]) == 0
    assert (tmp_path / ".pi/agent/skills/russian-utr-skill/SKILL.md").is_file()


def test_dry_run_changes_nothing(tmp_path, capsys):
    assert inst.main(["--harness", "all", "--dry-run", "--home", str(tmp_path)]) == 0
    assert not any(tmp_path.iterdir())


def test_link_creates_symlink(tmp_path, capsys):
    try:
        code = inst.main(["--harness", "codex", "--link", "--home", str(tmp_path)])
    except OSError:
        pytest.skip("symlinks are not available")
    if code != 0:
        pytest.skip("symlinks are not available")
    target = tmp_path / ".agents/skills/russian-utr-skill"
    assert target.is_symlink()
    assert (target / "SKILL.md").is_file()


def test_unknown_harness_is_usage_error(capsys):
    with pytest.raises(SystemExit) as error:
        inst.main(["--harness", "nosuchagent"])
    assert error.value.code == 2


def test_installed_linter_runs(tmp_path, capsys):
    assert inst.main(["--harness", "agents", "--home", str(tmp_path)]) == 0
    import subprocess

    linter = tmp_path / ".agents/skills/russian-utr-skill/scripts/utr-lint.py"
    result = subprocess.run([sys.executable, str(linter), "--selftest"], capture_output=True)
    assert result.returncode == 0, result.stderr
