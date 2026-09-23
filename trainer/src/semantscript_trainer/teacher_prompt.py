"""Deterministic provider-neutral prompts for one generated case."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from semantscript_trainer.case_contract import build_case_schema

SYSTEM_PROMPT = """You generate one labeled training case for a typed SemantScript function.
Return only the JSON object required by the supplied response schema. Follow the behavior,
examples, and constraints exactly. Produce realistic, diverse inputs. Do not explain the answer
and do not wrap JSON in Markdown."""


def build_case_messages(
    ir: Mapping[str, Any],
    index: int,
    total: int,
) -> tuple[str, str]:
    """Return stable system/user messages for a zero-based case position."""

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("case index must be a non-negative integer")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1 or index >= total:
        raise ValueError("case total must be positive and greater than the case index")

    try:
        projection = {
            "functionId": ir["id"],
            "casePosition": {"index": index, "total": total},
            "definition": ir["definition"],
            "inputs": ir["inputs"],
            "output": ir["output"],
            "responseSchema": build_case_schema(ir),
        }
    except KeyError as error:
        raise ValueError(f"IR is missing required prompt field {error.args[0]!r}") from error

    payload = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return SYSTEM_PROMPT, f"Generate case {index + 1} of {total} from this contract:\n{payload}"
