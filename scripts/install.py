#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the russian-utr skill into the skill directory of an agent harness.

The skill follows the open Agent Skills format (https://agentskills.io).
Most harnesses read the shared directory ~/.agents/skills. Claude Code reads
only ~/.claude/skills. This script copies the runtime files of the skill
(no tests, no git metadata) into one or more of these directories.

Usage:
    install.py                         # ~/.agents/skills and ~/.claude/skills
    install.py --harness codex         # one harness
    install.py --harness opencode,pi   # several harnesses
    install.py --harness all           # every known directory
    install.py --project PATH          # project scope instead of user scope
    install.py --link                  # symlink instead of copy
    install.py --force                 # replace an existing installation
    install.py --dry-run               # print the plan, change nothing
    install.py --list                  # print known harnesses and paths

Exit codes:
    0  done
    1  target exists and --force is not given, or copy failed
    2  bad command line
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

SKILL_NAME = "russian-utr"
SOURCE = Path(__file__).resolve().parent.parent

#: Files and directories the skill needs at run time.
RUNTIME_ITEMS = (
    "SKILL.md",
    "LICENSE",
    "requirements-optional.txt",
    "references",
    "examples",
    "scripts",
)

#: harness -> (user-scope skills root, project-scope skills root).
#: Paths are relative to the home directory or to the project root.
HARNESSES: dict[str, tuple[str, str]] = {
    "agents": (".agents/skills", ".agents/skills"),
    "claude": (".claude/skills", ".claude/skills"),
    "opencode": (".config/opencode/skills", ".opencode/skills"),
    "codex": (".agents/skills", ".agents/skills"),
    "gemini": (".gemini/skills", ".gemini/skills"),
    "cursor": (".cursor/skills", ".cursor/skills"),
    "pi": (".pi/agent/skills", ".pi/skills"),
    "omp": (".omp/agent/skills", ".omp/skills"),
    "goose": (".agents/skills", ".agents/skills"),
}

#: The default covers every harness listed above: the shared directory plus
#: the one harness that does not read it.
DEFAULT_HARNESSES = ("agents", "claude")


def parse_harnesses(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if names == ["all"]:
        return list(HARNESSES)
    unknown = [name for name in names if name not in HARNESSES]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            "неизвестный харнес: " + ", ".join(unknown or [value])
            + ". Допустимо: " + ", ".join([*HARNESSES, "all"])
        )
    return names


def targets(harnesses: list[str], home: Path, project: Path | None) -> list[Path]:
    """Return unique target directories in a stable order."""
    seen: list[Path] = []
    for name in harnesses:
        user_root, project_root = HARNESSES[name]
        root = project / project_root if project else home / user_root
        target = root / SKILL_NAME
        if target not in seen:
            seen.append(target)
    return seen


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))}


def install(target: Path, link: bool, force: bool) -> None:
    if target.is_symlink() or target.exists():
        if not force:
            raise FileExistsError(f"{target} уже существует. Добавьте --force, чтобы заменить.")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.symlink(SOURCE, target, target_is_directory=True)
        return
    target.mkdir()
    for item in RUNTIME_ITEMS:
        source = SOURCE / item
        if source.is_dir():
            shutil.copytree(source, target / item, ignore=_ignore)
        elif source.is_file():
            shutil.copy2(source, target / item)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Устанавливает скилл russian-utr в каталог скиллов агента."
    )
    parser.add_argument(
        "--harness",
        type=parse_harnesses,
        default=list(DEFAULT_HARNESSES),
        help="список харнесов через запятую или all (по умолчанию: agents,claude)",
    )
    parser.add_argument("--project", type=Path, help="каталог проекта для установки в проект")
    parser.add_argument("--link", action="store_true", help="создать символическую ссылку вместо копии")
    parser.add_argument("--force", action="store_true", help="заменить существующую установку")
    parser.add_argument("--dry-run", action="store_true", help="показать план и ничего не менять")
    parser.add_argument("--list", action="store_true", help="показать харнесы и пути")
    parser.add_argument("--home", type=Path, default=Path.home(), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        for name, (user_root, project_root) in HARNESSES.items():
            print(f"{name:9} ~/{user_root}/{SKILL_NAME}  |  <проект>/{project_root}/{SKILL_NAME}")
        return 0

    project = args.project.resolve() if args.project else None
    status = 0
    for target in targets(args.harness, args.home, project):
        if args.dry_run:
            print(f"план: {target}")
            continue
        try:
            install(target, args.link, args.force)
        except (OSError, FileExistsError) as error:
            print(f"ошибка: {error}", file=sys.stderr)
            status = 1
            continue
        print(f"установлено: {target}")
    return status


if __name__ == "__main__":
    sys.exit(main())
