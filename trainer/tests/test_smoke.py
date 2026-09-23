from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from semantscript_trainer import PACKAGE_NAME


def test_trainer_package_imports() -> None:
    assert PACKAGE_NAME == "semantscript_trainer"


def test_trainer_package_imports_without_optional_training_dependencies() -> None:
    source = Path(__file__).parents[1] / "src"
    script = f"""
import builtins
import sys

sys.path.insert(0, {str(source)!r})
real_import = builtins.__import__

def block_optional(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch.") or name == "transformers":
        raise ModuleNotFoundError(f"blocked optional dependency: {{name}}", name=name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_optional
import semantscript_trainer

assert semantscript_trainer.PACKAGE_NAME == "semantscript_trainer"
assert semantscript_trainer.TrainingConfig().encoder_name == "answerdotai/ModernBERT-base"
assert callable(semantscript_trainer.train_classifier)
assert callable(semantscript_trainer.serialize_canonical_inputs)
"""

    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
