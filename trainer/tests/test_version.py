"""The release version the trainer shares with the npm packages, and its commit."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

import semantscript_trainer
from semantscript_trainer import cli
from semantscript_trainer._version import __version__

ROOT = Path(__file__).resolve().parents[2]


def test_the_trainer_version_is_the_root_package_version() -> None:
    root_version = json.loads((ROOT / "package.json").read_text())["version"]
    # scripts/version.mjs spells pre-releases the PEP 440 way (1.2.3-rc.1 -> 1.2.3rc1).
    python_spelling = re.sub(r"-(alpha|beta|rc)\.(\d+)$", r"\1\2", root_version)
    python_spelling = python_spelling.replace("alpha", "a").replace("beta", "b")
    assert __version__ == python_spelling
    assert semantscript_trainer.__version__ == __version__
    assert cli.TRAINER_VERSION == __version__


def test_pyproject_reads_the_version_from_the_package() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "semantscript-trainer"
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "semantscript_trainer._version.__version__"
    }


def test_an_installed_trainer_records_no_commit_of_the_surrounding_repository(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    # A project repository with the trainer installed in its .venv.
    project = tmp_path / "project"
    installed = project / ".venv" / "lib" / "python3.12" / "site-packages" / "semantscript_trainer"
    installed.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    assert cli._git_commit(installed) == "0000000"
    # The same layout as the source tree, but the git top level is elsewhere.
    lookalike = project / "vendor" / "trainer" / "src" / "semantscript_trainer"
    lookalike.mkdir(parents=True)
    assert cli._git_commit(lookalike) == "0000000"


def test_a_source_checkout_records_its_commit() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    package = ROOT / "trainer" / "src" / "semantscript_trainer"
    assert cli._git_commit(package) == head.stdout.strip()
