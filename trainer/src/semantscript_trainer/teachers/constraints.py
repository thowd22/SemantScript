"""The built-in constraints teacher: labels come from an expression's own constraints.

When an expression's ``always`` and ``never`` constraints admit exactly one
output for an input, they are a labelling function for it. This teacher samples
inputs from the IR's input types, labels each with that one output and builds
the adversarial cases the same way: a boundary pair is two labelled inputs one
field apart on either side of a predicate, a counterfactual twin is a
single-field edit that changes the label. No language model and no API key are
involved, so the whole ``semantscript train`` path runs offline.

Sampling is threshold-aware. Every comparison between an input path and a
literal in a predicate (``order.total > 1000``, ``ticket.category === "outage"``)
records a threshold or a string for that path; a numeric input is sampled over a
range inferred from its thresholds and the gold examples (or the range the
teacher TOML sets for it), with a share of the draws placed on and beside a
threshold, and a string input mostly takes the values the predicates compare it
with.

Two modes:

* **pure** (no fallback): every sampled input must be decided. An input the
  constraints leave open (no admissible output, or several) stops the build with
  that input, the outputs it admits and the fix.
* **mixed** (a ``[teacher.fallback]`` language-model teacher): the constraints
  label the share of inputs they decide, estimated from a seeded pilot sample,
  and the fallback teacher generates the rest; any fallback case whose input the
  constraints decide takes the constraints' label, and a boundary pair or twin
  the constraints cannot build comes from the fallback.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from semantscript_trainer.adversarial import AdversarialGenerationError
from semantscript_trainer.case_contract import _head_support
from semantscript_trainer.constraints import (
    CompiledConstraints,
    ConstraintConfigurationError,
    ConstraintEvaluationError,
    compile_constraints,
    json_values_equal,
)
from semantscript_trainer.teacher import (
    AdversarialTeacher,
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    JsonValue,
    NeuralFunctionIr,
    Teacher,
    TeacherConfigurationError,
    TeacherDescriptor,
)
from semantscript_trainer.teacher_config import (
    CONSTRAINTS_ALGORITHM_VERSION,
    CONSTRAINTS_BACKEND,
    ConstraintsTeacherConfig,
    NumberRange,
)

CONSTRAINTS_TEACHER_MODEL = f"compiled-constraints-v{CONSTRAINTS_ALGORITHM_VERSION}"
# Inputs drawn to estimate the share of inputs the constraints decide (mixed mode).
PILOT_SAMPLES = 512
# A stream that yields no new distinct input for this many draws is exhausted: the
# input space is smaller than the requested count, so samples repeat from there on.
STALL_ATTEMPTS = 5_000
# In mixed mode, how long the constraints search for a two-sided boundary pair or a
# decided twin before the fallback teacher is asked instead.
MIXED_SEARCH_ATTEMPTS = 2_000
# How long a pure-mode boundary search samples before the constraint is reported as
# having no two-sided boundary the constraints decide.
BOUNDARY_SEARCH_ATTEMPTS = 20_000
STRING_POOL = ("alpha", "bravo", "charlie", "delta")
OPTIONAL_PRESENT_SHARE = 0.75
_COMPARISONS = frozenset(("===", "!==", "<", "<=", ">", ">="))


class _Omit:
    """Counterfactual edit that removes an optional field."""

    def __repr__(self) -> str:
        return "<omitted>"


_OMIT = _Omit()


# --- sampler ------------------------------------------------------------------


@dataclass(slots=True)
class _PathFacts:
    """What the predicates and the gold examples say about one input path."""

    thresholds: set[float] = field(default_factory=set)
    strings: set[str] = field(default_factory=set)
    length_thresholds: set[float] = field(default_factory=set)
    examples: list[JsonValue] = field(default_factory=list)
    referenced: bool = False


@dataclass(frozen=True, slots=True)
class _Leaf:
    """One editable position in a concrete input value."""

    segments: tuple[str | int, ...]
    key: str
    spec: Mapping[str, Any]
    optional: bool
    present: bool


class ConstraintSampler:
    """Samples, labels and edits the inputs of one neural-function IR record."""

    def __init__(self, ir: Mapping[str, Any], config: ConstraintsTeacherConfig) -> None:
        self.ir = copy.deepcopy(dict(ir))
        self.config = config
        self.where = describe_expression(self.ir)
        output = self.ir.get("output")
        if not isinstance(output, Mapping) or output.get("kind") != "scalar":
            raise _Unsupported("its output is an object, and the constraints label scalar outputs")
        try:
            self.constraints: CompiledConstraints = compile_constraints(self.ir)
            self.support: tuple[JsonValue, ...] = tuple(_head_support(output["head"]))
        except (ConstraintConfigurationError, TeacherConfigurationError) as error:
            raise TeacherConfigurationError(
                f"{self.where}: its constraints cannot be compiled: {error}"
            ) from error
        if len(self.constraints) == 0:
            raise _Unsupported("it declares no constraints")
        self.rules: tuple[tuple[str, JsonValue], ...] = tuple(
            (cast(str, item["kind"]), cast(JsonValue, item["output"])) for item in self.constraints
        )
        self.inputs: tuple[Mapping[str, Any], ...] = tuple(self.ir["inputs"])
        self.known_keys: set[str] = set()
        for entry in self.inputs:
            self._collect_keys(entry["type"], entry["name"])
        self.facts: dict[str, _PathFacts] = {}
        self.global_thresholds: set[float] = set()
        for item in self.constraints:
            self._collect_predicate(item["predicate"])
        examples = cast(dict[str, Any], self.ir["definition"]).get("examples", [])
        for example in examples if isinstance(examples, list) else []:
            inputs = example.get("inputs") if isinstance(example, Mapping) else None
            if isinstance(inputs, Mapping):
                for entry in self.inputs:
                    if entry["name"] in inputs:
                        self._collect_example(entry["type"], entry["name"], inputs[entry["name"]])
        self._ranges: dict[str, NumberRange] = {}

    # Labels ---------------------------------------------------------------

    def admissible(self, inputs: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
        """Every output that violates no active constraint (the output-contract rule).

        Raises ``ConstraintEvaluationError`` when a predicate cannot evaluate on
        the input (such an input is never used).
        """

        active = [
            index
            for index in range(len(self.constraints))
            if self.constraints.evaluate(index, inputs)
        ]
        allowed: list[JsonValue] = []
        for value in self.support:
            ok = True
            for index in active:
                kind, required = self.rules[index]
                equal = json_values_equal(value, required)
                if (kind == "always" and not equal) or (kind == "never" and equal):
                    ok = False
                    break
            if ok:
                allowed.append(value)
        return tuple(allowed)

    def label(self, inputs: Mapping[str, JsonValue]) -> JsonValue | _Undecided:
        allowed = self.admissible(inputs)
        if len(allowed) == 1:
            return allowed[0]
        return _Undecided(allowed)

    def predicate(self, index: int, inputs: Mapping[str, JsonValue]) -> bool:
        return self.constraints.evaluate(index, inputs)

    # Sampling -------------------------------------------------------------

    def sample(self, rng: random.Random) -> dict[str, JsonValue]:
        return {
            entry["name"]: self.sample_type(entry["type"], entry["name"], rng)
            for entry in self.inputs
        }

    def sample_type(self, spec: Mapping[str, Any], key: str, rng: random.Random) -> JsonValue:
        kind = spec["kind"]
        if kind == "object":
            value: dict[str, JsonValue] = {}
            for item in spec["fields"]:
                if item.get("optional") and rng.random() >= OPTIONAL_PRESENT_SHARE:
                    continue
                value[item["name"]] = self.sample_type(item["type"], f"{key}.{item['name']}", rng)
            return value
        if kind == "boolean":
            return rng.random() < 0.5
        if kind == "number":
            return self.sample_number(key, rng)
        if kind == "string":
            return self.sample_string(key, rng)
        if kind == "literal":
            return cast(JsonValue, spec["value"])
        if kind == "enum":
            return cast(JsonValue, rng.choice(list(spec["values"])))
        if kind == "union":
            return self.sample_type(rng.choice(list(spec["variants"])), key, rng)
        if kind == "null":
            return None
        if kind == "tuple":
            return [
                self.sample_type(item, f"{key}[{index}]", rng)
                for index, item in enumerate(spec["items"])
            ]
        if kind == "array":
            return [
                self.sample_type(spec["items"], f"{key}[]", rng)
                for _ in range(self._array_length(key, rng))
            ]
        raise TeacherConfigurationError(f"{self.where}: unsupported input kind {kind!r} at {key}")

    def range_for(self, key: str) -> NumberRange:
        cached = self._ranges.get(key)
        if cached is None:
            configured = self.config.range_for(key)
            cached = configured if configured is not None else self._infer_range(key)
            self._ranges[key] = cached
        return cached

    def thresholds_for(self, key: str) -> list[float]:
        facts = self.facts.get(key)
        own = facts.thresholds if facts is not None else set()
        if own:
            return sorted(own)
        if facts is not None and facts.referenced:
            return sorted(self.global_thresholds)
        return []

    def sample_number(self, key: str, rng: random.Random) -> int | float:
        numbers = self.range_for(key)
        low, high = numbers.low, numbers.high
        near = [value for value in self.thresholds_for(key) if low <= value <= high]
        if near and rng.random() < self.config.near_threshold_share:
            threshold = rng.choice(near)
            step = _step(threshold, numbers.decimals)
            value: float = threshold + rng.choice((-step, 0, 0, step))
            value = min(high, max(low, value))
            return _number(round(value, max(numbers.decimals, _decimals_of(threshold))))
        decimals = numbers.decimals if numbers.decimals and rng.random() < 0.3 else 0
        if numbers.distribution == "count":
            if rng.random() < 0.5 and low <= 0 <= high:
                return 0
            return rng.randint(max(math.ceil(low), 1), max(math.floor(high), 1))
        if numbers.distribution == "log" and rng.random() < 0.5:
            # Half log-uniform (small values are as common as large ones), half
            # uniform (large values are still seen often enough to learn their
            # thresholds).
            value = 10 ** rng.uniform(0, math.log10(max(high, 1.0)))
        else:
            value = rng.uniform(low, high)
        value = min(high, max(low, value))
        return _number(round(value, decimals))

    def sample_string(self, key: str, rng: random.Random) -> str:
        facts = self.facts.get(key)
        literals = sorted(facts.strings) if facts is not None else []
        if facts is not None and facts.length_thresholds and rng.random() < 0.5:
            target = int(
                max(0, rng.choice(sorted(facts.length_thresholds)) + rng.choice((-1, 0, 1)))
            )
            return (rng.choice(STRING_POOL) * (target // 5 + 1))[:target]
        if literals and rng.random() < 0.8:
            return rng.choice(literals)
        examples = [value for value in (facts.examples if facts else []) if isinstance(value, str)]
        return rng.choice([*STRING_POOL, *examples])

    def _array_length(self, key: str, rng: random.Random) -> int:
        facts = self.facts.get(key)
        if facts is not None and facts.length_thresholds and rng.random() < 0.5:
            return int(max(0, rng.choice(sorted(facts.length_thresholds)) + rng.choice((-1, 0, 1))))
        return rng.randint(0, 3)

    def _infer_range(self, key: str) -> NumberRange:
        facts = self.facts.get(key, _PathFacts())
        thresholds = self.thresholds_for(key)
        examples = [
            float(value)
            for value in facts.examples
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        values = [*thresholds, *examples]
        decimals = min(3, max((_decimals_of(value) for value in values), default=0))
        if thresholds:
            high = max(10.0, 2 * max(thresholds), *examples)
        else:
            high = max(100.0, *examples)
        low = min(0.0, *(2 * value for value in thresholds if value < 0), *examples)
        high = float(math.ceil(high))
        low = float(math.floor(low))
        if decimals == 0 and low == 0 and max(values, default=0) <= 10 and thresholds:
            distribution = "count"
        elif low >= 0 and high >= 1000:
            distribution = "log"
        else:
            distribution = "uniform"
        return NumberRange(low=low, high=high, distribution=distribution, decimals=decimals)

    # Edits ----------------------------------------------------------------

    def leaves(self, inputs: Mapping[str, JsonValue]) -> list[_Leaf]:
        found: list[_Leaf] = []

        def walk(
            spec: Mapping[str, Any],
            value: Any,
            segments: tuple[str | int, ...],
            key: str,
            optional: bool,
        ) -> None:
            kind = spec["kind"]
            if kind == "object" and isinstance(value, dict):
                for item in spec["fields"]:
                    name = item["name"]
                    child_key = f"{key}.{name}"
                    if name in value:
                        walk(
                            item["type"],
                            value[name],
                            (*segments, name),
                            child_key,
                            bool(item.get("optional")),
                        )
                    else:
                        found.append(_Leaf((*segments, name), child_key, item["type"], True, False))
                return
            if kind == "tuple" and isinstance(value, list):
                for index, item in enumerate(spec["items"]):
                    walk(item, value[index], (*segments, index), f"{key}[{index}]", False)
                return
            if kind == "array" and isinstance(value, list):
                # The whole array is one position (its length can change) and so
                # is each item.
                found.append(_Leaf(segments, key, spec, optional, True))
                for index, item in enumerate(value):
                    walk(spec["items"], item, (*segments, index), f"{key}[]", False)
                return
            found.append(_Leaf(segments, key, spec, optional, True))

        for entry in self.inputs:
            name = entry["name"]
            if name in inputs:
                walk(entry["type"], inputs[name], (name,), name, False)
        return found

    def candidate_values(self, leaf: _Leaf, current: Any, rng: random.Random) -> list[Any]:
        if not leaf.present:
            return [self.sample_type(leaf.spec, leaf.key, rng) for _ in range(3)]
        spec = leaf.spec
        kind = spec["kind"]
        values: list[Any] = []
        if kind == "boolean":
            values = [not current]
        elif kind == "number":
            numbers = self.range_for(leaf.key)
            for threshold in self.thresholds_for(leaf.key):
                step = _step(threshold, numbers.decimals)
                for delta in (0, step, -step):
                    candidate = threshold + delta
                    if numbers.low <= candidate <= numbers.high:
                        values.append(
                            _number(
                                round(candidate, max(numbers.decimals, _decimals_of(threshold)))
                            )
                        )
            values.extend(self.sample_number(leaf.key, rng) for _ in range(4))
        elif kind == "string":
            facts = self.facts.get(leaf.key)
            values = [*(sorted(facts.strings) if facts else []), rng.choice(STRING_POOL)]
        elif kind == "enum":
            values = list(spec["values"])
        elif kind == "union":
            for variant in spec["variants"]:
                if variant["kind"] == "literal":
                    values.append(variant["value"])
                else:
                    values.append(self.sample_type(variant, leaf.key, rng))
        elif kind == "array":
            values = [
                [self.sample_type(spec["items"], f"{leaf.key}[]", rng) for _ in range(length)]
                for length in (0, 1, 2)
            ]
        unique: list[Any] = []
        seen: set[str] = set()
        for value in values:
            encoded = _key(value)
            if encoded in seen or _same(value, current):
                continue
            seen.add(encoded)
            unique.append(value)
        if leaf.optional:
            unique.append(_OMIT)
        return unique

    def edits(
        self, inputs: Mapping[str, JsonValue], rng: random.Random
    ) -> Iterator[tuple[_Leaf, Any, Any, dict[str, JsonValue]]]:
        """Single-field edits of ``inputs`` in a random order: (leaf, old, new, twin)."""

        leaves = self.leaves(inputs)
        rng.shuffle(leaves)
        for leaf in leaves:
            current = _get(inputs, leaf.segments) if leaf.present else _OMIT
            for value in self.candidate_values(leaf, current, rng):
                twin = copy.deepcopy(dict(inputs))
                _set(twin, leaf.segments, value)
                yield leaf, current, value, twin

    # Facts ----------------------------------------------------------------

    def _collect_keys(self, spec: Mapping[str, Any], key: str) -> None:
        self.known_keys.add(key)
        kind = spec.get("kind")
        if kind == "object":
            for item in spec["fields"]:
                self._collect_keys(item["type"], f"{key}.{item['name']}")
        elif kind == "array":
            self._collect_keys(spec["items"], f"{key}[]")
        elif kind == "tuple":
            for index, item in enumerate(spec["items"]):
                self._collect_keys(item, f"{key}[{index}]")
        elif kind == "union":
            for variant in spec["variants"]:
                self._collect_keys(variant, key)

    def _path_key(self, expression: Mapping[str, Any]) -> str | None:
        node = expression.get("node")
        if node == "input":
            name = expression["name"]
            return name if name in self.known_keys else None
        if node == "property":
            parent = self._path_key(expression["object"])
            if parent is None:
                return None
            child = f"{parent}.{expression['property']}"
            return child if child in self.known_keys else None
        if node == "index":
            parent = self._path_key(expression["object"])
            if parent is None:
                return None
            index = expression["index"]
            if index.get("node") == "literal" and isinstance(index.get("value"), int):
                exact = f"{parent}[{index['value']}]"
                if exact in self.known_keys:
                    return exact
            wildcard = f"{parent}[]"
            return wildcard if wildcard in self.known_keys else None
        return None

    def _length_key(self, expression: Mapping[str, Any]) -> str | None:
        if expression.get("node") == "property" and expression.get("property") == "length":
            parent = self._path_key(expression["object"])
            if parent is not None and f"{parent}.length" not in self.known_keys:
                return parent
        return None

    def _collect_predicate(self, expression: Mapping[str, Any]) -> None:
        node = expression.get("node")
        if node == "literal":
            value = expression.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.global_thresholds.add(float(value))
            return
        key = self._path_key(expression)
        if key is not None:
            self.facts.setdefault(key, _PathFacts()).referenced = True
        if node == "binary" and expression["operator"] in _COMPARISONS:
            for path_side, literal_side in (
                (expression["left"], expression["right"]),
                (expression["right"], expression["left"]),
            ):
                literal = _literal_value(literal_side)
                if literal is _NO_LITERAL:
                    continue
                length_key = self._length_key(path_side)
                path_key = self._path_key(path_side)
                if length_key is not None and _is_number(literal):
                    self.facts.setdefault(length_key, _PathFacts()).length_thresholds.add(
                        float(cast(float, literal))
                    )
                elif path_key is not None and _is_number(literal):
                    self.facts.setdefault(path_key, _PathFacts()).thresholds.add(
                        float(cast(float, literal))
                    )
                elif path_key is not None and isinstance(literal, str):
                    self.facts.setdefault(path_key, _PathFacts()).strings.add(literal)
        for child_name in ("object", "index", "operand", "left", "right"):
            child = expression.get(child_name)
            if isinstance(child, Mapping):
                self._collect_predicate(child)

    def _collect_example(self, spec: Mapping[str, Any], key: str, value: Any) -> None:
        kind = spec.get("kind")
        if kind == "object" and isinstance(value, Mapping):
            for item in spec["fields"]:
                if item["name"] in value:
                    self._collect_example(
                        item["type"], f"{key}.{item['name']}", value[item["name"]]
                    )
        elif kind == "array" and isinstance(value, list):
            for item in value:
                self._collect_example(spec["items"], f"{key}[]", item)
        elif kind == "tuple" and isinstance(value, list):
            for index, item in enumerate(spec["items"]):
                if index < len(value):
                    self._collect_example(item, f"{key}[{index}]", value[index])
        elif kind == "union":
            for variant in spec["variants"]:
                if variant.get("kind") in ("object", "array", "tuple"):
                    self._collect_example(variant, key, value)
            if not isinstance(value, (dict, list)):
                self.facts.setdefault(key, _PathFacts()).examples.append(value)
        else:
            self.facts.setdefault(key, _PathFacts()).examples.append(value)


@dataclass(frozen=True, slots=True)
class _Undecided:
    allowed: tuple[JsonValue, ...]


class _Unsupported(Exception):
    """The expression cannot be labelled by its constraints at all."""


# --- teacher ------------------------------------------------------------------


class ConstraintsTeacher:
    """A ``Teacher`` and ``AdversarialTeacher`` whose labels are the constraints' own.

    ``fallback`` (mixed mode) is a language-model teacher for the inputs the
    constraints leave open; ``create_teacher`` builds it from the
    ``[teacher.fallback]`` table.
    """

    def __init__(
        self,
        config: ConstraintsTeacherConfig | None = None,
        *,
        fallback: Teacher | None = None,
    ) -> None:
        resolved = config if config is not None else ConstraintsTeacherConfig()
        if not isinstance(resolved, ConstraintsTeacherConfig):
            raise TeacherConfigurationError("config must be a ConstraintsTeacherConfig")
        if fallback is None and resolved.fallback is not None:
            from semantscript_trainer.teacher_config import create_teacher

            fallback = create_teacher(resolved.fallback)
        if fallback is not None and not isinstance(fallback, Teacher):
            raise TeacherConfigurationError("the fallback does not implement the Teacher protocol")
        self._config = resolved
        self._fallback = fallback
        if fallback is None:
            self._descriptor = TeacherDescriptor(
                provider=CONSTRAINTS_BACKEND,
                model=CONSTRAINTS_TEACHER_MODEL,
                configuration_sha256=resolved.sampling_sha256,
            )
        else:
            inner = fallback.descriptor
            self._descriptor = TeacherDescriptor(
                provider=f"{CONSTRAINTS_BACKEND}+{inner.provider}",
                model=inner.model,
                configuration_sha256=_sha256_json(
                    {
                        "sampling": resolved.sampling_projection(),
                        "fallback": {
                            "provider": inner.provider,
                            "model": inner.model,
                            "configurationSha256": inner.configuration_sha256,
                        },
                    }
                ),
            )
        self._samplers: dict[str, ConstraintSampler | str] = {}
        self._attempts: dict[str, int] = {}
        self._shares: dict[str, float] = {}

    @property
    def descriptor(self) -> TeacherDescriptor:
        return self._descriptor

    @property
    def config(self) -> ConstraintsTeacherConfig:
        return self._config

    @property
    def fallback(self) -> Teacher | None:
        return self._fallback

    # Teacher protocol -------------------------------------------------------

    def generate(self, ir: NeuralFunctionIr, n: int, /) -> tuple[GeneratedCase, ...]:
        if n <= 0:
            return ()
        sampler = self._sampler(ir)
        if self._fallback is None:
            return self._decided_samples(
                sampler, n, "train", strict=True, twin_filter=self._config.twin_filter
            )
        if sampler is None:
            return tuple(self._fallback.generate(ir, n))
        share = self.decided_share(ir)
        decided_count = n if share >= 1 else round(share * n)
        decided = self._decided_samples(
            sampler, decided_count, "train", strict=False, twin_filter=False
        )
        rest = n - len(decided)
        cases = list(decided)
        if rest:
            seen = {_key(case.inputs) for case in decided}
            for case in self._fallback.generate(ir, rest):
                relabelled = _relabel(sampler, case)
                if _key(relabelled.inputs) in seen:
                    continue
                seen.add(_key(relabelled.inputs))
                cases.append(relabelled)
            shortfall = n - len(cases)
            if shortfall > 0 and decided_count > 0:
                cases.extend(
                    self._decided_samples(
                        sampler,
                        shortfall,
                        "train-topup",
                        strict=False,
                        twin_filter=False,
                        exclude=seen,
                    )
                )
            elif shortfall > 0:
                # Nothing is decided and the fallback repeated inputs: return what it
                # gave (repeats and all) so the count is exact.
                cases.extend(
                    _relabel(sampler, case) for case in self._fallback.generate(ir, shortfall)
                )
        return tuple(cases[:n])

    def generate_boundary_pair(self, ir: NeuralFunctionIr, index: int, /) -> BoundaryPairProposal:
        sampler = self._sampler(ir)
        mixed = self._fallback is not None
        if sampler is not None:
            rng = self._rng(ir, f"boundary:{index}")
            attempts = min(
                MIXED_SEARCH_ATTEMPTS if mixed else BOUNDARY_SEARCH_ATTEMPTS,
                self._config.maximum_sampling_attempts,
            )
            proposal = self._constraint_boundary_pair(
                sampler, index, rng, attempts, strict=not mixed
            )
            if proposal is not None:
                return proposal
        if isinstance(self._fallback, AdversarialTeacher):
            proposal = self._fallback.generate_boundary_pair(ir, index)
            if sampler is None:
                return proposal
            return BoundaryPairProposal(
                predicate_false=_relabel(sampler, proposal.predicate_false),
                predicate_true=_relabel(sampler, proposal.predicate_true),
            )
        where = describe_expression(ir)
        raise AdversarialGenerationError(
            f"{where}: constraint {index} has no two-sided boundary the constraints decide on "
            "both sides within the sampling budget"
        )

    def generate_counterfactual(
        self, ir: NeuralFunctionIr, anchor: GeneratedCase, /
    ) -> CounterfactualProposal:
        sampler = self._sampler(ir)
        if sampler is not None:
            proposal = _counterfactual(sampler, anchor, self._rng(ir, "counterfactual"))
            if proposal is not None:
                return proposal
        if isinstance(self._fallback, AdversarialTeacher):
            proposal = self._fallback.generate_counterfactual(ir, anchor)
            if sampler is None:
                return proposal
            return CounterfactualProposal(
                twin=_relabel(sampler, proposal.twin), reason=proposal.reason
            )
        raise AdversarialGenerationError(
            f"{describe_expression(ir)}: no single-field edit of {_key(anchor.inputs)} "
            "moves the constraints to another output"
        )

    # Public helpers -----------------------------------------------------------

    def sample_decided(
        self,
        ir: NeuralFunctionIr,
        n: int,
        /,
        *,
        stream: str = "heldout",
        exclude: Iterable[Mapping[str, JsonValue]] = (),
    ) -> tuple[GeneratedCase, ...]:
        """``n`` distinct inputs the constraints decide, labelled, from a named stream.

        For held-out sets: pass a ``stream`` the training corpus never uses and the
        training inputs as ``exclude`` so no held-out input was trained on.
        Undecided inputs are skipped, never labelled.
        """

        if stream in ("train", "train-topup", "pilot", "counterfactual") or stream.startswith(
            "boundary:"
        ):
            raise TeacherConfigurationError(f"stream {stream!r} is reserved for training")
        sampler = self._sampler(ir)
        if sampler is None:
            raise TeacherConfigurationError(
                f"{describe_expression(ir)}: {self._samplers[_function_key(ir)]}; "
                "the constraints cannot label it"
            )
        return self._decided_samples(
            sampler,
            n,
            stream,
            strict=False,
            twin_filter=False,
            exclude={_key(inputs) for inputs in exclude},
            repeat_when_exhausted=False,
        )

    def decided_share(self, ir: NeuralFunctionIr) -> float:
        """The share of sampled inputs the constraints decide, from a seeded pilot sample."""

        function_key = _function_key(ir)
        share = self._shares.get(function_key)
        if share is not None:
            return share
        sampler = self._sampler(ir)
        if sampler is None:
            share = 0.0
        else:
            rng = self._rng(ir, "pilot")
            decided = evaluated = 0
            for _ in range(PILOT_SAMPLES):
                try:
                    allowed = sampler.admissible(sampler.sample(rng))
                except ConstraintEvaluationError:
                    continue
                evaluated += 1
                decided += len(allowed) == 1
            share = decided / evaluated if evaluated else 0.0
        self._shares[function_key] = share
        return share

    # Internals ------------------------------------------------------------------

    def _sampler(self, ir: NeuralFunctionIr) -> ConstraintSampler | None:
        function_key = _function_key(ir)
        cached = self._samplers.get(function_key)
        if cached is None:
            try:
                cached = ConstraintSampler(ir, self._config)
            except _Unsupported as reason:
                cached = str(reason)
            self._samplers[function_key] = cached
        if isinstance(cached, str):
            if self._fallback is None:
                raise TeacherConfigurationError(
                    f"{describe_expression(ir)}: {cached}, so the constraints teacher cannot "
                    "label it. Add always/never constraints that decide every input, or add a "
                    "[teacher.fallback] table with a language-model teacher (docs/teachers.md)"
                )
            return None
        return cached

    def _rng(self, ir: NeuralFunctionIr, purpose: str) -> random.Random:
        # Distinct, reproducible streams per function and purpose; a retry of the
        # same purpose continues the stream instead of repeating it.
        key = f"{ir['id']}:{purpose}"
        attempt = self._attempts.get(key, 0)
        self._attempts[key] = attempt + 1
        digest = hashlib.sha256(f"{self._config.seed}:{key}:{attempt}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _decided_samples(
        self,
        sampler: ConstraintSampler | None,
        n: int,
        purpose: str,
        *,
        strict: bool,
        twin_filter: bool,
        exclude: set[str] | None = None,
        repeat_when_exhausted: bool = True,
    ) -> tuple[GeneratedCase, ...]:
        if n <= 0:
            return ()
        assert sampler is not None
        rng = self._rng(sampler.ir, purpose)
        excluded = exclude if exclude is not None else set()
        cases: dict[str, GeneratedCase] = {}
        attempts = since_new = 0
        last_error: ConstraintEvaluationError | None = None
        while len(cases) < n:
            attempts += 1
            since_new += 1
            if attempts > self._config.maximum_sampling_attempts or (
                since_new > STALL_ATTEMPTS and cases
            ):
                if cases and repeat_when_exhausted:
                    # The input space holds fewer distinct inputs than requested:
                    # repeat the ones found, in order, to reach the exact count.
                    found = list(cases.values())
                    return tuple(found[index % len(found)] for index in range(n))
                detail = f" (last evaluation error: {last_error})" if last_error else ""
                raise TeacherConfigurationError(
                    f"{sampler.where}: found {len(cases)} of {n} distinct inputs the "
                    f"constraints decide in {attempts - 1} draws{detail}; widen the sampling "
                    "ranges in the teacher TOML"
                    + (" or set twin_filter = false" if twin_filter else "")
                )
            inputs = sampler.sample(rng)
            key = _key(inputs)
            if key in cases or key in excluded:
                continue
            try:
                label = sampler.label(inputs)
            except ConstraintEvaluationError as error:
                last_error = error
                continue
            if isinstance(label, _Undecided):
                if strict:
                    raise incomplete_constraints_error(sampler, inputs, label.allowed)
                continue
            case = GeneratedCase(inputs, label)
            # Every training case may be picked as a counterfactual anchor, so the
            # pure-mode corpus keeps only inputs that one field edit can move across
            # the policy; interior points several edits from any boundary are dropped.
            if twin_filter and _counterfactual(sampler, case, rng) is None:
                continue
            cases[key] = case
            since_new = 0
        return tuple(cases.values())

    def _constraint_boundary_pair(
        self,
        sampler: ConstraintSampler,
        index: int,
        rng: random.Random,
        attempts: int,
        *,
        strict: bool,
    ) -> BoundaryPairProposal | None:
        # Two labelled inputs one field apart with the predicate false on one side
        # and true on the other.
        for _ in range(attempts):
            inputs = sampler.sample(rng)
            try:
                label = sampler.label(inputs)
                side = sampler.predicate(index, inputs)
            except ConstraintEvaluationError:
                continue
            if isinstance(label, _Undecided):
                if strict:
                    raise incomplete_constraints_error(sampler, inputs, label.allowed)
                continue
            for _leaf, _old, _new, twin in sampler.edits(inputs, rng):
                try:
                    if sampler.predicate(index, twin) == side:
                        continue
                    twin_label = sampler.label(twin)
                except ConstraintEvaluationError:
                    continue
                if isinstance(twin_label, _Undecided):
                    continue
                pair = {
                    side: GeneratedCase(inputs, label),
                    not side: GeneratedCase(twin, twin_label),
                }
                return BoundaryPairProposal(predicate_false=pair[False], predicate_true=pair[True])
        return None


def _counterfactual(
    sampler: ConstraintSampler, anchor: GeneratedCase, rng: random.Random
) -> CounterfactualProposal | None:
    for leaf, old, new, twin in sampler.edits(anchor.inputs, rng):
        try:
            label = sampler.label(twin)
        except ConstraintEvaluationError:
            continue
        if isinstance(label, _Undecided) or json_values_equal(label, anchor.output):
            continue
        return CounterfactualProposal(
            twin=GeneratedCase(twin, label),
            reason=(
                f"{'removing' if new is _OMIT else 'setting'} {_display(leaf.segments)}"
                + ("" if old is _OMIT else f" from {_key(old)}")
                + ("" if new is _OMIT else f" to {_key(new)}")
                + f" moves the constraints' answer to {_key(label)}"
            ),
        )
    return None


def _relabel(sampler: ConstraintSampler, case: GeneratedCase) -> GeneratedCase:
    """The constraints' label for an input they decide; the teacher's label otherwise."""

    try:
        label = sampler.label(case.inputs)
    except ConstraintEvaluationError:
        return case
    if isinstance(label, _Undecided):
        return case
    return GeneratedCase(case.inputs, label)


# --- messages -----------------------------------------------------------------


def describe_expression(ir: Mapping[str, Any]) -> str:
    """``src/refunds.sem.ts:28 (nf_38d661b1)``: where an expression is, for messages."""

    source = ir.get("source")
    function_id = str(ir.get("id", "?"))
    short = function_id[:11] if function_id.startswith("nf_") else function_id
    if isinstance(source, Mapping) and source.get("path"):
        line = source.get("line")
        at = f"{source['path']}:{line}" if line is not None else str(source["path"])
        return f"{at} ({short})"
    return short


def incomplete_constraints_error(
    sampler: ConstraintSampler, inputs: Mapping[str, JsonValue], allowed: Sequence[JsonValue]
) -> TeacherConfigurationError:
    """The pure-mode failure for an input the constraints do not decide."""

    if allowed:
        admits = "admit " + ", ".join(_key(value) for value in allowed)
        why = "the constraints do not decide every input"
    else:
        admits = "admit no output (the active constraints contradict each other)"
        why = "the constraints contradict each other on an input"
    return TeacherConfigurationError(
        f"{sampler.where}: {why}, so the constraints teacher cannot label it alone. "
        f"For the input {_key(inputs)} the constraints {admits}. Add constraints until "
        "exactly one output is admissible for every input, or add a [teacher.fallback] "
        "table with a language-model teacher to label the inputs the constraints leave "
        "open (docs/teachers.md)"
    )


# --- helpers ------------------------------------------------------------------


class _NoLiteral:
    pass


_NO_LITERAL = _NoLiteral()


def _literal_value(expression: Mapping[str, Any]) -> Any:
    if expression.get("node") == "literal":
        return expression.get("value")
    if (
        expression.get("node") == "unary"
        and expression.get("operator") in ("-", "+")
        and isinstance(expression.get("operand"), Mapping)
        and expression["operand"].get("node") == "literal"
        and _is_number(expression["operand"].get("value"))
    ):
        value = expression["operand"]["value"]
        return -value if expression["operator"] == "-" else value
    return _NO_LITERAL


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _decimals_of(value: float) -> int:
    if float(value).is_integer():
        return 0
    text = repr(float(value))
    if "e" in text or "E" in text:
        return 3
    return min(3, len(text.split(".", 1)[1]))


def _step(threshold: float, decimals: int) -> float:
    places = max(decimals, _decimals_of(threshold))
    return 1.0 if places == 0 else 10.0**-places


def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _same(left: Any, right: Any) -> bool:
    if left is _OMIT or right is _OMIT:
        return left is right
    return (_key(left) == _key(right) and type(left) is type(right)) or (
        _is_number(left) and _is_number(right) and float(left) == float(right)
    )


def _get(value: Any, segments: Sequence[str | int]) -> Any:
    for segment in segments:
        value = value[segment]
    return value


def _set(value: Any, segments: Sequence[str | int], new: Any) -> None:
    target = value
    for segment in segments[:-1]:
        target = target[segment]
    if new is _OMIT:
        del target[segments[-1]]
    else:
        target[segments[-1]] = new


def _display(segments: Sequence[str | int]) -> str:
    text = ""
    for segment in segments:
        text += f"[{segment}]" if isinstance(segment, int) else (f".{segment}" if text else segment)
    return text


def _function_key(ir: Mapping[str, Any]) -> str:
    return f"{ir.get('id')}:{ir.get('semanticSha256')}"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CONSTRAINTS_TEACHER_MODEL",
    "ConstraintSampler",
    "ConstraintsTeacher",
    "describe_expression",
    "incomplete_constraints_error",
]
