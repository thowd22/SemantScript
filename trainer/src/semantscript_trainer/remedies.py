"""The fix text the trainer prints, from ``diagnostics/remedies.json``.

``scripts/generate-remedies.mjs`` writes the templates into
``remedies_generated.py``; the same source feeds the runtime's and the CLI's
messages and the fix column of ``docs/diagnostics.md``, so they cannot drift.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, cast

from semantscript_trainer.remedies_generated import REMEDY_TEMPLATES_JSON

REMEDY_TEMPLATES: Mapping[str, Mapping[str, Any]] = cast(
    dict[str, dict[str, Any]], json.loads(REMEDY_TEMPLATES_JSON)
)
_PLACEHOLDER = re.compile(r"\{([a-z][a-zA-Z0-9]*)\}")


def remedy(remedy_id: str, /, **params: object) -> str:
    """Fill one remedy's template; an unknown id or a missing or extra parameter raises."""

    entry = REMEDY_TEMPLATES.get(remedy_id)
    if entry is None:
        raise KeyError(f"unknown remedy {remedy_id!r}")
    names = cast(list[str], entry["params"])
    extra = sorted(set(params) - set(names))
    if extra:
        raise TypeError(f"remedy {remedy_id} has no parameter {', '.join(extra)}")
    missing = [name for name in names if name not in params]
    if missing:
        raise TypeError(f"remedy {remedy_id} needs parameter {', '.join(missing)}")
    return _PLACEHOLDER.sub(lambda match: str(params[match.group(1)]), cast(str, entry["fix"]))


def remedy_family(remedy_id: str, /) -> str:
    """The family one remedy belongs to: verifier, trainer-process, runtime, cli, doctor or editor."""

    entry = REMEDY_TEMPLATES.get(remedy_id)
    if entry is None:
        raise KeyError(f"unknown remedy {remedy_id!r}")
    return cast(str, entry["family"])
