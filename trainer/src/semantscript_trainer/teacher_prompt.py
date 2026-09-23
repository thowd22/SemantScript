"""Deterministic provider-neutral prompts for one generated case.

Every teacher shares these messages, so corpus coverage is decided here rather
than by any provider's sampling. The ``coverageBrief`` is a pure function of the
IR and the case position: it cycles the target label through the output
support, cycles a constraint focus through *none*, *satisfy* and *near-miss* for
every input-dependent constraint, and derives per-leaf variation hints from a
hash of the function id, case index and input path. Rejection notes let a
retrying teacher tell the model why the previous inputs were refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from semantscript_trainer.case_contract import _head_support, build_case_schema
from semantscript_trainer.constraints import (
    ConstraintError,
    json_values_equal,
    predicate_depends_on_inputs,
)
from semantscript_trainer.teacher import JsonValue

PROMPT_CONTRACT_VERSION = 2

SYSTEM_PROMPT = """You generate one labeled training case for a typed SemantScript function.
Return only the JSON object required by the supplied response schema. The output must always be
the decision the behavior, examples, and constraints actually imply for the inputs you wrote:
whenever a constraint predicate holds for your inputs, that constraint decides the output, and
otherwise the behavior text decides it. Then honor the coverageBrief in this order of
precedence: constraintFocus, targetOutput, variation. targetOutput is only a preference for
which inputs to write; if no inputs satisfying constraintFocus can correctly produce it, keep
constraintFocus and label truthfully with a different output. Produce realistic, specific,
varied inputs. Do not reuse a typical example, values another case in this corpus would
obviously use, or any inputs listed under rejectedAttempts. Do not explain the answer and do
not wrap JSON in Markdown."""

MAXIMUM_VARIATION_HINTS = 64
MAXIMUM_REJECTION_NOTES = 8
MAXIMUM_REJECTION_REASON_CHARACTERS = 512

_NUMBER_HINTS = ("small", "typical", "large", "unusual but valid")
_STRING_HINTS = ("short", "long", "unusual but realistic")
_BOOLEAN_HINTS = ("false", "true")
_ARRAY_HINTS = ("empty", "one item", "several items")
_OPTIONAL_HINTS = ("omitted", "present")


@dataclass(frozen=True, slots=True)
class RejectionNote:
    """Why a previous attempt at the same case position was refused."""

    reason: str
    inputs: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("rejection reason must be a nonempty string")
        if self.inputs is not None and not isinstance(self.inputs, Mapping):
            raise ValueError("rejection inputs must be a mapping or None")


def build_case_messages(
    ir: Mapping[str, Any],
    index: int,
    total: int,
    *,
    rejected: Sequence[RejectionNote] = (),
) -> tuple[str, str]:
    """Return stable system/user messages for a zero-based case position."""

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("case index must be a non-negative integer")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1 or index >= total:
        raise ValueError("case total must be positive and greater than the case index")
    if isinstance(rejected, (str, bytes)) or not isinstance(rejected, Sequence):
        raise ValueError("rejected attempts must be a sequence of RejectionNote values")
    notes = tuple(rejected)
    if any(not isinstance(note, RejectionNote) for note in notes):
        raise ValueError("rejected attempts must be a sequence of RejectionNote values")

    try:
        projection = {
            "functionId": ir["id"],
            "casePosition": {"index": index, "total": total},
            "coverageBrief": build_coverage_brief(ir, index),
            "definition": ir["definition"],
            "inputs": ir["inputs"],
            "output": ir["output"],
            "responseSchema": build_case_schema(ir),
        }
    except KeyError as error:
        raise ValueError(f"IR is missing required prompt field {error.args[0]!r}") from error

    payload = _dump(projection)
    user = f"Generate case {index + 1} of {total} from this contract:\n{payload}"
    if notes:
        recent = notes[-MAXIMUM_REJECTION_NOTES:]
        rejected_payload = _dump(
            {
                "rejectedAttempts": [
                    {
                        "reason": note.reason[:MAXIMUM_REJECTION_REASON_CHARACTERS],
                        "inputs": note.inputs,
                    }
                    for note in recent
                ]
            }
        )
        user += (
            "\n\nEarlier attempts at this exact case were rejected for the reasons below."
            " Produce materially different inputs that fix every reason:\n"
            f"{rejected_payload}"
        )
    return SYSTEM_PROMPT, user


def build_coverage_brief(ir: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Return the deterministic coverage brief for one case position."""

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("case index must be a non-negative integer")
    support = _scalar_support(ir)
    constraints = _input_dependent_constraints(ir)
    slots = len(support) if support else 1
    target: JsonValue | None = support[index % slots] if support else None
    has_target = support is not None
    region = (index // slots) % (1 + 2 * len(constraints))

    focus: dict[str, Any]
    if region == 0:
        focus = {
            "mode": "none",
            "instruction": "No constraint predicate may hold for these inputs.",
        }
    else:
        position, near_miss = divmod(region - 1, 2)
        constraint_index, constraint = constraints[position]
        common = {
            "constraintIndex": constraint_index,
            "kind": constraint["kind"],
            "source": constraint["source"],
            "constraintOutput": constraint["output"],
        }
        if near_miss:
            focus = {
                "mode": "near-miss",
                **common,
                "instruction": (
                    "Bring the inputs as close as possible to this predicate without "
                    "making it hold; the constraint must stay inactive."
                ),
            }
        else:
            focus = {
                "mode": "satisfy",
                **common,
                "instruction": (
                    "The inputs must make this predicate hold, so the constraint decides "
                    "the output."
                ),
            }
            if support is not None:
                target, has_target = _forced_target(constraint, support, index)

    brief: dict[str, Any] = {
        "constraintFocus": focus,
        "variation": _variation_hints(ir, index),
    }
    if has_target:
        brief["targetOutput"] = {
            "value": target,
            "instruction": (
                "Choose inputs for which this is the correct output whenever the behavior "
                "and constraints allow it; never mislabel a case to reach it."
            ),
        }
    return brief


def _forced_target(
    constraint: Mapping[str, Any],
    support: Sequence[JsonValue],
    index: int,
) -> tuple[JsonValue | None, bool]:
    forced = cast(JsonValue, constraint["output"])
    if constraint["kind"] == "always":
        return forced, True
    allowed = [value for value in support if not json_values_equal(value, forced)]
    if not allowed:
        return None, False
    return allowed[index % len(allowed)], True


def _scalar_support(ir: Mapping[str, Any]) -> list[JsonValue] | None:
    output = ir.get("output")
    if not isinstance(output, Mapping) or output.get("kind") != "scalar":
        return None
    head = output.get("head")
    if not isinstance(head, Mapping):
        return None
    try:
        return _head_support(head)
    except Exception:
        return None


def _input_dependent_constraints(
    ir: Mapping[str, Any],
) -> list[tuple[int, Mapping[str, Any]]]:
    definition = ir.get("definition")
    if not isinstance(definition, Mapping):
        return []
    raw = definition.get("constraints")
    if not isinstance(raw, list):
        return []
    result: list[tuple[int, Mapping[str, Any]]] = []
    for position, constraint in enumerate(raw):
        if not isinstance(constraint, Mapping) or constraint.get("kind") not in ("always", "never"):
            continue
        if not isinstance(constraint.get("source"), str) or "output" not in constraint:
            continue
        try:
            if not predicate_depends_on_inputs(constraint):
                continue
        except (ConstraintError, TypeError, ValueError, KeyError):
            continue
        result.append((position, constraint))
    return result


def _variation_hints(ir: Mapping[str, Any], index: int) -> dict[str, str]:
    hints: dict[str, str] = {}
    seed = f"{ir.get('id')}:{index}"
    inputs = ir.get("inputs")
    if not isinstance(inputs, list):
        return hints
    for entry in inputs:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        spec = entry.get("type")
        if isinstance(name, str) and isinstance(spec, Mapping):
            _walk_variation(spec, name, seed, hints)
    return hints


def _walk_variation(
    spec: Mapping[str, Any],
    path: str,
    seed: str,
    hints: dict[str, str],
) -> None:
    if len(hints) >= MAXIMUM_VARIATION_HINTS:
        return
    kind = spec.get("kind")
    if kind == "number":
        hints[path] = _pick(seed, path, _NUMBER_HINTS)
    elif kind == "string":
        hints[path] = _pick(seed, path, _STRING_HINTS)
    elif kind == "boolean":
        hints[path] = _pick(seed, path, _BOOLEAN_HINTS)
    elif kind == "enum":
        values = spec.get("values")
        if isinstance(values, list) and values:
            hints[path] = "prefer " + _dump_compact(_pick(seed, path, tuple(values)))
    elif kind == "union":
        variants = spec.get("variants")
        if not isinstance(variants, list) or not variants:
            return
        literals = [
            variant.get("value")
            for variant in variants
            if isinstance(variant, Mapping) and variant.get("kind") == "literal"
        ]
        if len(literals) == len(variants):
            hints[path] = "prefer " + _dump_compact(_pick(seed, path, tuple(literals)))
            return
        chosen = _pick(seed, path, tuple(range(len(variants))))
        hints[path] = f"prefer union variant {chosen}"
        variant = variants[chosen]
        if isinstance(variant, Mapping):
            _walk_variation(variant, path, seed, hints)
    elif kind == "array":
        hints[path] = _pick(seed, path, _ARRAY_HINTS)
        items = spec.get("items")
        if isinstance(items, Mapping):
            _walk_variation(items, f"{path}[]", seed, hints)
    elif kind == "tuple":
        items = spec.get("items")
        if isinstance(items, list):
            for position, item in enumerate(items):
                if isinstance(item, Mapping):
                    _walk_variation(item, f"{path}[{position}]", seed, hints)
    elif kind == "object":
        fields = spec.get("fields")
        if not isinstance(fields, list):
            return
        for field in fields:
            if not isinstance(field, Mapping):
                continue
            name = field.get("name")
            field_spec = field.get("type")
            if not isinstance(name, str) or not isinstance(field_spec, Mapping):
                continue
            child = f"{path}.{name}"
            if field.get("optional") is True:
                hints[child] = _pick(seed, child, _OPTIONAL_HINTS)
                if hints[child] == "omitted":
                    continue
                child = f"{child} (when present)"
            _walk_variation(field_spec, child, seed, hints)


def _pick(seed: str, path: str, options: Sequence[Any]) -> Any:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return options[int.from_bytes(digest[:8], "big") % len(options)]


def _dump(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)


def _dump_compact(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True)


__all__ = [
    "MAXIMUM_REJECTION_NOTES",
    "MAXIMUM_VARIATION_HINTS",
    "PROMPT_CONTRACT_VERSION",
    "SYSTEM_PROMPT",
    "RejectionNote",
    "build_case_messages",
    "build_coverage_brief",
]
