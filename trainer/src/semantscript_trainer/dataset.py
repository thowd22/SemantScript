"""Deterministic training-dataset assembly and completed-result caching."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal, NoReturn, cast

from semantscript_trainer.case_contract import (
    build_case_schema,
    validate_case,
    validate_case_count,
)
from semantscript_trainer.constraints import (
    CompiledConstraints,
    ConstraintConfigurationError,
    ConstraintEvaluationBudget,
    ConstraintEvaluationError,
    ConstraintViolationError,
    compile_constraints,
)
from semantscript_trainer.teacher import (
    CaseGenerator,
    GeneratedCase,
    JsonValue,
    NeuralFunctionIr,
    Teacher,
    TeacherConfigurationError,
    TeacherDescriptor,
    TeacherResponseError,
)
from semantscript_trainer.teacher_prompt import PROMPT_CONTRACT_VERSION

DATASET_KIND = "semantscript.training-dataset"
DATASET_VERSION = 1
MAXIMUM_DATASET_CASE_COUNT = 20_000
MAXIMUM_DATASET_CACHE_BYTES = 64 * 1024 * 1024
MAXIMUM_DATASET_CACHE_DEPTH = 128
MAXIMUM_DATASET_CACHE_NODES = 2_000_000

_REQUEST_DIGEST_DOMAIN = b"semantscript.dataset-request/v1\0"
_PAYLOAD_DIGEST_DOMAIN = b"semantscript.dataset-payload/v1\0"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FUNCTION_ID = re.compile(r"^nf_[a-f0-9]{64}$")
_CASE_ORIGINS = frozenset(("gold", "synthetic"))

type CaseOrigin = Literal["gold", "synthetic"]


class DatasetError(RuntimeError):
    """Base class for dataset assembly and cache failures."""


class DatasetConfigurationError(DatasetError, ValueError):
    """The requested dataset or source IR is invalid."""


class DatasetCacheError(DatasetError):
    """A dataset cache entry cannot be read, validated, or published safely."""


@dataclass(frozen=True, slots=True, init=False)
class DatasetCase:
    """One labeled row and its trusted-gold or teacher-synthetic origin."""

    _inputs_json: bytes = field(repr=False)
    _output_json: bytes = field(repr=False)
    origin: CaseOrigin

    def __init__(
        self,
        inputs: dict[str, JsonValue],
        output: JsonValue,
        origin: CaseOrigin,
    ) -> None:
        if not isinstance(inputs, dict):
            raise DatasetConfigurationError("dataset case inputs must be an object")
        if not isinstance(origin, str) or origin not in _CASE_ORIGINS:
            raise DatasetConfigurationError("dataset case origin must be 'gold' or 'synthetic'")
        try:
            inputs_json = _canonical_json_bytes(inputs)
            output_json = _canonical_json_bytes(output)
        except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
            raise DatasetConfigurationError(
                f"dataset case cannot be snapshotted as canonical JSON: {error}"
            ) from error
        object.__setattr__(self, "_inputs_json", inputs_json)
        object.__setattr__(self, "_output_json", output_json)
        object.__setattr__(self, "origin", origin)

    @property
    def inputs(self) -> dict[str, JsonValue]:
        """Return a defensive copy so provenance digests cannot become stale."""

        value = json.loads(self._inputs_json)
        if not isinstance(value, dict):
            raise AssertionError("canonical dataset inputs snapshot is not an object")
        return cast(dict[str, JsonValue], value)

    @property
    def output(self) -> JsonValue:
        """Return a defensive copy so provenance digests cannot become stale."""

        return cast(JsonValue, json.loads(self._output_json))


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    """A validated, deterministic dataset and its cache/provenance identities."""

    function_id: str
    semantic_sha256: str
    requested_case_count: int
    teacher: TeacherDescriptor
    cases: tuple[DatasetCase, ...]
    cache_key_sha256: str
    payload_sha256: str
    dataset_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.function_id, str)
            or _FUNCTION_ID.fullmatch(self.function_id) is None
        ):
            raise DatasetConfigurationError(
                "dataset function_id must match nf_<64 lowercase hexadecimal characters>"
            )
        if (
            not isinstance(self.semantic_sha256, str)
            or _SHA256.fullmatch(self.semantic_sha256) is None
        ):
            raise DatasetConfigurationError(
                "dataset semantic_sha256 must be 64 lowercase hexadecimal characters"
            )
        try:
            expected_count = validate_case_count(self.requested_case_count)
        except TeacherConfigurationError as error:
            raise DatasetConfigurationError(str(error)) from error
        if not isinstance(self.teacher, TeacherDescriptor):
            raise DatasetConfigurationError("dataset teacher must be a TeacherDescriptor")
        if not isinstance(self.cases, tuple) or len(self.cases) != expected_count:
            raise DatasetConfigurationError(
                "dataset cases must be a tuple matching requested_case_count"
            )
        seen_synthetic = False
        for case in self.cases:
            if not isinstance(case, DatasetCase):
                raise DatasetConfigurationError("dataset cases must contain DatasetCase values")
            if case.origin == "synthetic":
                seen_synthetic = True
            elif seen_synthetic:
                raise DatasetConfigurationError("gold dataset cases must precede synthetic cases")
        for name in ("cache_key_sha256", "payload_sha256", "dataset_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise DatasetConfigurationError(
                    f"dataset {name} must be 64 lowercase hexadecimal characters"
                )

    @property
    def gold_count(self) -> int:
        return sum(case.origin == "gold" for case in self.cases)

    @property
    def synthetic_count(self) -> int:
        return len(self.cases) - self.gold_count


@dataclass(frozen=True, slots=True)
class _DatasetRequest:
    ir: NeuralFunctionIr
    function_id: str
    semantic_sha256: str
    total_cases: int
    gold_cases: tuple[DatasetCase, ...]
    descriptor: TeacherDescriptor
    constraints: CompiledConstraints
    constraint_budget: ConstraintEvaluationBudget
    cache_key_sha256: str


class SyntheticDatasetGenerator:
    """Assemble exact-size datasets and cache complete results by request identity."""

    __slots__ = ("_cache_directory", "_case_generator")

    def __init__(self, teacher: Teacher, cache_directory: str | os.PathLike[str]) -> None:
        self._case_generator = CaseGenerator(teacher)
        try:
            self._cache_directory = Path(cache_directory)
        except TypeError as error:
            raise DatasetConfigurationError("cache_directory must be path-like") from error

    @property
    def descriptor(self) -> TeacherDescriptor:
        return self._case_generator.descriptor

    def cache_path(self, ir: NeuralFunctionIr, total_cases: int, /) -> Path:
        request = self._prepare_request(ir, total_cases)
        return self._path_for_request(request)

    def generate(self, ir: NeuralFunctionIr, total_cases: int, /) -> TrainingDataset:
        request = self._prepare_request(ir, total_cases)
        path = self._path_for_request(request)
        self._ensure_cache_directory(path.parent)
        lock_path = path.with_suffix(".lock")

        with _exclusive_lock(lock_path):
            if _cache_entry_exists(path):
                return _load_dataset(path, request)

            synthetic_count = request.total_cases - len(request.gold_cases)
            teacher_ir = deepcopy(request.ir)
            generated = self._case_generator.generate(teacher_ir, synthetic_count)
            try:
                teacher_ir_bytes = _canonical_json_bytes(teacher_ir)
            except (
                TypeError,
                ValueError,
                OverflowError,
                UnicodeEncodeError,
                RecursionError,
            ) as error:
                raise TeacherResponseError(
                    f"teacher mutated the IR generation request into invalid JSON: {error}"
                ) from error
            if teacher_ir_bytes != _canonical_json_bytes(request.ir):
                raise TeacherResponseError("teacher mutated the IR generation request")
            cases = _assemble_cases(request, generated)
            document, payload_sha256 = _build_document(request, cases)
            encoded = _canonical_document_bytes(document)
            if len(encoded) > MAXIMUM_DATASET_CACHE_BYTES:
                raise DatasetCacheError(
                    f"dataset cache entry exceeds maximum byte length {MAXIMUM_DATASET_CACHE_BYTES}"
                )
            self._publish(path, encoded, request)
            loaded = _load_dataset(path, request)
            if not hmac.compare_digest(loaded.payload_sha256, payload_sha256):
                raise DatasetCacheError("published dataset payload digest changed unexpectedly")
            return loaded

    def _prepare_request(self, ir: NeuralFunctionIr, total_cases: int) -> _DatasetRequest:
        try:
            requested = validate_case_count(total_cases)
        except TeacherConfigurationError as error:
            raise DatasetConfigurationError(str(error)) from error
        if requested > MAXIMUM_DATASET_CASE_COUNT:
            raise DatasetConfigurationError(
                f"dataset case count exceeds maximum {MAXIMUM_DATASET_CASE_COUNT}"
            )
        if not isinstance(ir, dict):
            raise DatasetConfigurationError("IR must be an object")
        try:
            ir = deepcopy(ir)
        except (TypeError, ValueError, RecursionError) as error:
            raise DatasetConfigurationError(f"IR could not be snapshotted: {error}") from error

        if ir.get("kind") != "semantscript.neural-function":
            raise DatasetConfigurationError("IR kind must be 'semantscript.neural-function'")
        ir_version = ir.get("irVersion")
        if (
            isinstance(ir_version, bool)
            or not isinstance(ir_version, (int, float))
            or ir_version != 1
        ):
            raise DatasetConfigurationError("IR irVersion must be 1")

        function_id = ir.get("id")
        if not isinstance(function_id, str) or _FUNCTION_ID.fullmatch(function_id) is None:
            raise DatasetConfigurationError(
                "IR id must match nf_<64 lowercase hexadecimal characters>"
            )
        semantic_sha256 = ir.get("semanticSha256")
        if not isinstance(semantic_sha256, str) or _SHA256.fullmatch(semantic_sha256) is None:
            raise DatasetConfigurationError(
                "IR semanticSha256 must be 64 lowercase hexadecimal characters"
            )
        definition = ir.get("definition")
        if not isinstance(definition, Mapping):
            raise DatasetConfigurationError("IR definition must be an object")
        raw_examples = definition.get("examples")
        if not isinstance(raw_examples, list):
            raise DatasetConfigurationError("IR definition examples must be an array")
        if requested < len(raw_examples):
            raise DatasetConfigurationError(
                f"total_cases {requested} cannot include all {len(raw_examples)} gold examples"
            )

        try:
            build_case_schema(ir)
            constraints = compile_constraints(ir)
            constraints.ensure_evaluation_budget(requested)
        except TeacherConfigurationError as error:
            raise DatasetConfigurationError(f"IR case contract is invalid: {error}") from error
        except ConstraintConfigurationError as error:
            raise DatasetConfigurationError(
                f"IR constraint contract is invalid: {error}"
            ) from error

        constraint_budget = ConstraintEvaluationBudget()
        gold_cases = tuple(
            _parse_gold_case(ir, example, index, constraints, constraint_budget)
            for index, example in enumerate(raw_examples)
        )
        descriptor = self._case_generator.descriptor
        if not isinstance(descriptor, TeacherDescriptor):
            raise DatasetConfigurationError("teacher descriptor must be a TeacherDescriptor")
        cache_key_sha256 = _request_digest(ir, descriptor, requested)
        return _DatasetRequest(
            ir=ir,
            function_id=function_id,
            semantic_sha256=semantic_sha256,
            total_cases=requested,
            gold_cases=gold_cases,
            descriptor=descriptor,
            constraints=constraints,
            constraint_budget=constraint_budget,
            cache_key_sha256=cache_key_sha256,
        )

    def _path_for_request(self, request: _DatasetRequest) -> Path:
        key = request.cache_key_sha256
        return self._cache_directory / "datasets" / f"v{DATASET_VERSION}" / key[:2] / f"{key}.json"

    @staticmethod
    def _ensure_cache_directory(directory: Path) -> None:
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            raise DatasetCacheError(f"could not create dataset cache directory: {error}") from error
        if not directory.is_dir():
            raise DatasetCacheError("dataset cache path is not a directory")

    @staticmethod
    def _publish(path: Path, encoded: bytes, request: _DatasetRequest) -> None:
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{request.cache_key_sha256}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

            _load_dataset(temporary_path, request)
            os.replace(temporary_path, path)
            temporary_path = None
            _fsync_directory(path.parent)
        except DatasetCacheError:
            raise
        except OSError as error:
            raise DatasetCacheError(f"could not publish dataset cache entry: {error}") from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _parse_gold_case(
    ir: NeuralFunctionIr,
    raw: object,
    index: int,
    constraints: CompiledConstraints,
    constraint_budget: ConstraintEvaluationBudget,
) -> DatasetCase:
    if not isinstance(raw, Mapping) or set(raw) != {"inputs", "output"}:
        raise DatasetConfigurationError(
            f"IR gold example {index} must contain exactly inputs and output"
        )
    inputs = raw["inputs"]
    if not isinstance(inputs, dict):
        raise DatasetConfigurationError(f"IR gold example {index} inputs must be an object")
    generated = GeneratedCase(inputs=inputs, output=cast(JsonValue, raw["output"]))
    try:
        validate_case(ir, generated)
        constraints.validate_case(generated, budget=constraint_budget)
    except (
        TeacherConfigurationError,
        TeacherResponseError,
        ConstraintConfigurationError,
        ConstraintEvaluationError,
        ConstraintViolationError,
    ) as error:
        raise DatasetConfigurationError(f"IR gold example {index} is invalid: {error}") from error
    return DatasetCase(
        inputs=inputs,
        output=cast(JsonValue, raw["output"]),
        origin="gold",
    )


def _assemble_cases(
    request: _DatasetRequest,
    generated: tuple[GeneratedCase, ...],
) -> tuple[DatasetCase, ...]:
    cases = list(request.gold_cases)
    approximate_bytes = sum(len(_canonical_json_bytes(_case_document(case))) for case in cases)
    for index, case in enumerate(generated):
        try:
            validate_case(request.ir, case)
            request.constraints.validate_case(case, budget=request.constraint_budget)
        except (
            TeacherConfigurationError,
            TeacherResponseError,
            ConstraintConfigurationError,
            ConstraintEvaluationError,
            ConstraintViolationError,
        ) as error:
            raise TeacherResponseError(
                f"synthetic dataset case {index} is invalid for the snapshotted IR: {error}"
            ) from error
        try:
            encoded = _canonical_json_bytes(
                {"origin": "synthetic", "inputs": case.inputs, "output": case.output}
            )
        except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
            raise TeacherResponseError(
                f"synthetic dataset case {index} cannot be serialized safely: {error}"
            ) from error
        approximate_bytes += len(encoded)
        if approximate_bytes > MAXIMUM_DATASET_CACHE_BYTES:
            raise DatasetCacheError(
                f"dataset cases exceed maximum byte length {MAXIMUM_DATASET_CACHE_BYTES}"
            )
        cases.append(DatasetCase(inputs=case.inputs, output=case.output, origin="synthetic"))
    return tuple(cases)


def _request_digest(
    ir: NeuralFunctionIr,
    descriptor: TeacherDescriptor,
    total_cases: int,
) -> str:
    try:
        projection: dict[str, object] = {
            "kind": "semantscript.dataset-request",
            "version": 1,
            "ir": {
                "kind": ir["kind"],
                "irVersion": ir["irVersion"],
                "id": ir["id"],
                "semanticSha256": ir["semanticSha256"],
                "definition": ir["definition"],
                "inputs": ir["inputs"],
                "output": ir["output"],
            },
            "teacher": _descriptor_document(descriptor),
            "generation": {
                "totalCases": total_cases,
                "datasetFormatVersion": DATASET_VERSION,
                "generatorContractVersion": 1,
                "promptContractVersion": PROMPT_CONTRACT_VERSION,
                "caseContractVersion": 1,
            },
        }
        encoded = _canonical_json_bytes(projection)
    except KeyError as error:
        raise DatasetConfigurationError(
            f"IR is missing required dataset field {error.args[0]!r}"
        ) from error
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise DatasetConfigurationError(
            f"IR cannot be canonicalized for dataset caching: {error}"
        ) from error
    return hashlib.sha256(_REQUEST_DIGEST_DOMAIN + encoded).hexdigest()


def _build_document(
    request: _DatasetRequest,
    cases: tuple[DatasetCase, ...],
) -> tuple[dict[str, object], str]:
    gold_count = len(request.gold_cases)
    payload: dict[str, object] = {
        "requestSha256": request.cache_key_sha256,
        "function": {
            "id": request.function_id,
            "semanticSha256": request.semantic_sha256,
        },
        "teacher": _descriptor_document(request.descriptor),
        "counts": {
            "total": request.total_cases,
            "gold": gold_count,
            "synthetic": request.total_cases - gold_count,
        },
        "cases": [_case_document(case) for case in cases],
    }
    payload_sha256 = hashlib.sha256(
        _PAYLOAD_DIGEST_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()
    return (
        {
            "kind": DATASET_KIND,
            "datasetVersion": DATASET_VERSION,
            "payloadSha256": payload_sha256,
            "payload": payload,
        },
        payload_sha256,
    )


def _descriptor_document(descriptor: TeacherDescriptor) -> dict[str, str]:
    return {
        "provider": descriptor.provider,
        "model": descriptor.model,
        "configurationSha256": descriptor.configuration_sha256,
    }


def _case_document(case: DatasetCase) -> dict[str, object]:
    return {
        "origin": case.origin,
        "inputs": case.inputs,
        "output": case.output,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")


def _canonical_document_bytes(document: object) -> bytes:
    try:
        return _canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
        raise DatasetCacheError(
            f"dataset cannot be serialized as canonical JSON: {error}"
        ) from error


def _load_dataset(path: Path, request: _DatasetRequest) -> TrainingDataset:
    encoded = _read_bounded_regular_file(path)
    document = _loads_dataset_json(encoded)
    if not isinstance(document, dict) or set(document) != {
        "kind",
        "datasetVersion",
        "payloadSha256",
        "payload",
    }:
        _bad_cache("dataset envelope has unknown or missing fields")
    if (
        document["kind"] != DATASET_KIND
        or type(document["datasetVersion"]) is not int
        or document["datasetVersion"] != DATASET_VERSION
    ):
        _bad_cache("dataset kind or version is unsupported")
    payload_sha256 = document["payloadSha256"]
    payload = document["payload"]
    if not isinstance(payload_sha256, str) or _SHA256.fullmatch(payload_sha256) is None:
        _bad_cache("dataset payloadSha256 is invalid")
    if not isinstance(payload, dict):
        _bad_cache("dataset payload must be an object")
    actual_payload_sha256 = hashlib.sha256(
        _PAYLOAD_DIGEST_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()
    if not hmac.compare_digest(payload_sha256, actual_payload_sha256):
        _bad_cache("dataset payload digest does not match its content")
    if _canonical_document_bytes(document) != encoded:
        _bad_cache("dataset cache entry is not canonical JSON")

    expected_payload_keys = {"requestSha256", "function", "teacher", "counts", "cases"}
    if set(payload) != expected_payload_keys:
        _bad_cache("dataset payload has unknown or missing fields")
    request_sha256 = payload["requestSha256"]
    if not isinstance(request_sha256, str) or not hmac.compare_digest(
        request_sha256, request.cache_key_sha256
    ):
        _bad_cache("dataset request digest does not match this generation request")

    function = payload["function"]
    if not isinstance(function, dict) or set(function) != {"id", "semanticSha256"}:
        _bad_cache("dataset function identity is invalid")
    if (
        function["id"] != request.function_id
        or function["semanticSha256"] != request.semantic_sha256
    ):
        _bad_cache("dataset function identity does not match this IR")

    teacher = _parse_descriptor(payload["teacher"])
    if teacher != request.descriptor:
        _bad_cache("dataset teacher identity does not match this generator")

    counts = payload["counts"]
    expected_gold = len(request.gold_cases)
    expected_synthetic = request.total_cases - expected_gold
    if not isinstance(counts, dict) or set(counts) != {"total", "gold", "synthetic"}:
        _bad_cache("dataset counts are invalid")
    if (
        type(counts["total"]) is not int
        or type(counts["gold"]) is not int
        or type(counts["synthetic"]) is not int
        or counts["total"] != request.total_cases
        or counts["gold"] != expected_gold
        or counts["synthetic"] != expected_synthetic
    ):
        _bad_cache("dataset counts do not match this generation request")

    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != request.total_cases:
        _bad_cache("dataset cases do not match the requested total")
    cases = tuple(
        _parse_cached_case(
            request.ir,
            request.constraints,
            request.constraint_budget,
            raw,
            index,
            expected_gold,
        )
        for index, raw in enumerate(raw_cases)
    )
    cached_gold = [_case_document(case) for case in cases[:expected_gold]]
    expected_gold_documents = [_case_document(case) for case in request.gold_cases]
    if _canonical_json_bytes(cached_gold) != _canonical_json_bytes(expected_gold_documents):
        _bad_cache("cached gold cases do not match the IR examples verbatim")

    return TrainingDataset(
        function_id=request.function_id,
        semantic_sha256=request.semantic_sha256,
        requested_case_count=request.total_cases,
        teacher=teacher,
        cases=cases,
        cache_key_sha256=request.cache_key_sha256,
        payload_sha256=payload_sha256,
        dataset_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _parse_descriptor(raw: object) -> TeacherDescriptor:
    if not isinstance(raw, dict) or set(raw) != {
        "provider",
        "model",
        "configurationSha256",
    }:
        _bad_cache("dataset teacher identity is invalid")
    provider = raw["provider"]
    model = raw["model"]
    configuration_sha256 = raw["configurationSha256"]
    if not all(isinstance(value, str) for value in (provider, model, configuration_sha256)):
        _bad_cache("dataset teacher identity fields must be strings")
    try:
        return TeacherDescriptor(
            provider=cast(str, provider),
            model=cast(str, model),
            configuration_sha256=cast(str, configuration_sha256),
        )
    except TeacherConfigurationError as error:
        raise DatasetCacheError(f"dataset teacher identity is invalid: {error}") from error


def _parse_cached_case(
    ir: NeuralFunctionIr,
    constraints: CompiledConstraints,
    constraint_budget: ConstraintEvaluationBudget,
    raw: object,
    index: int,
    gold_count: int,
) -> DatasetCase:
    if not isinstance(raw, dict) or set(raw) != {"origin", "inputs", "output"}:
        _bad_cache(f"dataset case {index} has unknown or missing fields")
    expected_origin = "gold" if index < gold_count else "synthetic"
    if raw["origin"] != expected_origin:
        _bad_cache(f"dataset case {index} has the wrong origin")
    inputs = raw["inputs"]
    if not isinstance(inputs, dict):
        _bad_cache(f"dataset case {index} inputs must be an object")
    generated = GeneratedCase(inputs=inputs, output=cast(JsonValue, raw["output"]))
    try:
        validate_case(ir, generated)
        constraints.validate_case(generated, budget=constraint_budget)
    except (
        TeacherConfigurationError,
        TeacherResponseError,
        ConstraintConfigurationError,
        ConstraintEvaluationError,
        ConstraintViolationError,
    ) as error:
        raise DatasetCacheError(f"dataset case {index} is invalid: {error}") from error
    return DatasetCase(
        inputs=inputs,
        output=cast(JsonValue, raw["output"]),
        origin=cast(CaseOrigin, expected_origin),
    )


def _cache_entry_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise DatasetCacheError(f"could not inspect dataset cache entry: {error}") from error
    return True


def _read_bounded_regular_file(path: Path) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                _bad_cache("dataset cache entry is not a regular file")
            size = metadata.st_size
            if size > MAXIMUM_DATASET_CACHE_BYTES:
                _bad_cache(
                    f"dataset cache entry exceeds maximum byte length {MAXIMUM_DATASET_CACHE_BYTES}"
                )
            encoded = stream.read(MAXIMUM_DATASET_CACHE_BYTES + 1)
    except DatasetCacheError:
        raise
    except OSError as error:
        raise DatasetCacheError(f"could not read dataset cache entry: {error}") from error
    if len(encoded) > MAXIMUM_DATASET_CACHE_BYTES:
        _bad_cache(f"dataset cache entry exceeds maximum byte length {MAXIMUM_DATASET_CACHE_BYTES}")
    return encoded


def _loads_dataset_json(encoded: bytes) -> JsonValue:
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise DatasetCacheError("dataset cache entry is not valid UTF-8") from error
    if text.startswith("\ufeff"):
        _bad_cache("dataset cache entry must not contain a byte order mark")
    _check_lexical_depth(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except DatasetCacheError:
        raise
    except (ValueError, OverflowError, RecursionError) as error:
        raise DatasetCacheError(f"dataset cache entry is invalid JSON: {error}") from error
    _check_json_tree(value)
    return cast(JsonValue, value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            _bad_cache(f"dataset cache entry contains duplicate property {name!r}")
        result[name] = value
    return result


def _reject_constant(token: str) -> NoReturn:
    _bad_cache(f"dataset cache entry contains forbidden non-finite constant {token}")


def _check_lexical_depth(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAXIMUM_DATASET_CACHE_DEPTH:
                _bad_cache(
                    f"dataset cache entry exceeds maximum depth {MAXIMUM_DATASET_CACHE_DEPTH}"
                )
        elif character in "]}":
            depth -= 1


def _check_json_tree(value: object) -> None:
    work = [value]
    nodes = 0
    while work:
        current = work.pop()
        nodes += 1
        if nodes > MAXIMUM_DATASET_CACHE_NODES:
            _bad_cache(
                f"dataset cache entry exceeds maximum node count {MAXIMUM_DATASET_CACHE_NODES}"
            )
        if isinstance(current, str):
            try:
                current.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise DatasetCacheError(
                    "dataset cache entry contains an unpaired Unicode surrogate"
                ) from error
        elif isinstance(current, float) and not math.isfinite(current):
            _bad_cache("dataset cache entry contains a non-finite number")
        elif isinstance(current, list):
            work.extend(current)
        elif isinstance(current, dict):
            for name, child in current.items():
                try:
                    name.encode("utf-8", errors="strict")
                except UnicodeEncodeError as error:
                    raise DatasetCacheError(
                        "dataset cache entry contains an unpaired Unicode surrogate"
                    ) from error
                work.append(child)


@contextmanager
def _exclusive_lock(path: Path):
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
    except OSError as error:
        raise DatasetCacheError(f"could not open dataset cache lock: {error}") from error
    try:
        try:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DatasetCacheError("dataset cache lock is not a private regular file")
            if metadata.st_size == 0:
                stream.write(b"\0")
                stream.flush()
            _lock_stream(stream)
        except OSError as error:
            raise DatasetCacheError(f"could not lock dataset cache entry: {error}") from error
        try:
            yield
        finally:
            try:
                _unlock_stream(stream)
            except OSError as error:
                raise DatasetCacheError(f"could not unlock dataset cache entry: {error}") from error
    finally:
        stream.close()


def _lock_stream(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _unlock_stream(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        # Windows does not support opening directory handles through os.open;
        # os.replace still provides atomic publication on the same volume.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bad_cache(message: str) -> NoReturn:
    raise DatasetCacheError(message)


__all__ = [
    "DATASET_KIND",
    "DATASET_VERSION",
    "MAXIMUM_DATASET_CASE_COUNT",
    "CaseOrigin",
    "DatasetCacheError",
    "DatasetCase",
    "DatasetConfigurationError",
    "DatasetError",
    "SyntheticDatasetGenerator",
    "TrainingDataset",
]
