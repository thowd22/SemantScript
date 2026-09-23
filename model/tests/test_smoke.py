from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from semantscript_model import PACKAGE_NAME


def test_model_package_imports() -> None:
    assert PACKAGE_NAME == "semantscript_model"


def test_model_package_imports_without_optional_training_dependencies() -> None:
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
import semantscript_model

assert semantscript_model.PACKAGE_NAME == "semantscript_model"
assert semantscript_model.EncoderConfig().model_name == "answerdotai/ModernBERT-base"
assert callable(semantscript_model.classification_loss)
try:
    semantscript_model.classification_loss(None, None)
except ImportError as error:
    assert "optional PyTorch training dependency" in str(error)
else:
    raise AssertionError("loss calculation did not defer the missing-PyTorch failure")
"""

    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
