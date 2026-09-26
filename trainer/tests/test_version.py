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
from semantscript_trainer.artifact import ArtifactProvenance

ROOT = Path(__file__).resolve().parents[2]


def test_the_trainer_version_is_the_root_package_version() -> None:
    root_version = json.loads((ROOT / "package.json").read_text())["version"]
    # scripts/version.mjs spells pre-releases the PEP 440 way (1.2.3-rc.1 -> 1.2.3rc1).
    python_spelling = re.sub(r"-(alpha|beta|rc)\.(\d+)$", r"\1\2", root_version)
    python_spelling = python_spelling.replace("alpha", "a").replace("beta", "b")
    assert __version__ == python_spelling
    assert semantscript_trainer.__version__ == __version__
    # The manifest records the semver spelling, the same string as the npm packages.
    assert cli.TRAINER_VERSION == root_version


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


def test_cli_version_flag_prints_the_shared_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"semantscript-trainer {__version__}"


def test_compiler_version_defaults_to_the_shared_version() -> None:
    arguments = cli._build_parser().parse_args(
        ["train", "--bundle", "b.json", "--artifact", "a", "--teacher", "constraints"]
    )
    assert arguments.compiler_version == cli.TRAINER_VERSION


@pytest.mark.parametrize(
    ("python_spelling", "semver"),
    [
        ("0.1.0", "0.1.0"),
        ("0.2.0a1", "0.2.0-alpha.1"),
        ("0.2.0b2", "0.2.0-beta.2"),
        ("0.2.0rc1", "0.2.0-rc.1"),
        ("10.20.30rc12", "10.20.30-rc.12"),
    ],
)
def test_a_pre_release_is_recorded_in_the_semver_spelling(
    python_spelling: str, semver: str
) -> None:
    assert cli.release_semver(python_spelling) == semver
    # The artifact provenance accepts what a pre-release trainer records.
    provenance = ArtifactProvenance(
        application_id="app1",
        application_version="0.0.0",
        compiler_version=semver,
        trainer_version=cli.release_semver(python_spelling),
        created_at="2026-09-26T00:00:00Z",
        training_key_sha256="a" * 64,
    )
    assert provenance.trainer_version == semver


def test_git_commit_handles_a_checkout_path_with_spaces(tmp_path: Path) -> None:
    root = tmp_path / "with space" / "repo"
    package = root / "trainer" / "src" / "semantscript_trainer"
    package.mkdir(parents=True)
    (package / "x.py").write_text("")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*git, "-C", str(root), "add", "."], check=True)
    subprocess.run([*git, "-C", str(root), "commit", "-qm", "x"], check=True)
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    assert cli._git_commit(package) == head.stdout.strip()


def test_the_console_script_names_itself_in_the_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["/venv/bin/semantscript-trainer", "--help"])
    assert cli._build_parser().prog == "semantscript-trainer"
    monkeypatch.setattr("sys.argv", ["/repo/trainer/src/semantscript_trainer/cli.py"])
    assert cli._build_parser().prog == "python -m semantscript_trainer.cli"
