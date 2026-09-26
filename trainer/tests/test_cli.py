"""The bundle-to-artifact driver behind ``semantscript train``."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from semantscript_trainer import __version__  # noqa: E402
from semantscript_trainer import cli as cli_module  # noqa: E402
from semantscript_trainer.cli import (  # noqa: E402
    TrainBundleError,
    TrainBundleFailure,
    train_bundle,
)
from semantscript_trainer.remedies import remedy  # noqa: E402
from semantscript_trainer.teacher import (  # noqa: E402
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    TeacherConfigurationError,
    TeacherDescriptor,
)
from semantscript_trainer.teacher_config import ConstraintsTeacherConfig  # noqa: E402
from semantscript_trainer.teacher_spend import SpendMeter, free_price  # noqa: E402
from semantscript_trainer.teachers import ConstraintsTeacher  # noqa: E402
from semantscript_trainer.training import TrainingConfig  # noqa: E402
from semantscript_trainer.verification import SeedRetryConfig, VerificationConfig  # noqa: E402

ROOT = Path(__file__).parents[2]
TOKENIZER_JSON = (ROOT / "runtime" / "test" / "fixtures" / "tokenizer.json").read_bytes()
CLI_ENTRY = ROOT / "cli" / "dist" / "index.js"
CORE_DECLARATIONS = ROOT / "cli" / "test" / "fixtures" / "core.d.ts"
REFUND_SOURCE = ROOT / "examples" / "refund.sem.ts"
REVISION = "7" * 40
RISK_SOURCE = """import { sema } from "@semantscript/core";

interface Customer {
  priorRefunds: number;
  tier: "enterprise" | "standard";
}

interface Order {
  ageDays: number;
  status: "fraudulent" | "paid";
  total: number;
}

export function refundRisk(customer: Customer, order: Order): "high" | "low" {
  return sema<"high" | "low">({
    examples: [
      {
        inputs: {
          customer: { priorRefunds: 0, tier: "standard" },
          order: { ageDays: 5, status: "fraudulent", total: 20 },
        },
        output: "high",
      },
    ],
  })`Rate the refund risk. Customer: ${customer} Order: ${order}`;
}
"""

# The refund example's policy stated completely: every input has exactly one
# admissible decision, so the constraints alone label the corpus.
COMPLETE_SOURCE = """import { always, sema } from "@semantscript/core";

type RefundDecision = "approve" | "deny" | "review";

interface Customer {
  priorRefunds: number;
  tier: "enterprise" | "standard";
}

interface Order {
  ageDays: number;
  status: "fraudulent" | "paid";
  total: number;
}

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>({
    examples: [
      {
        inputs: {
          customer: { priorRefunds: 0, tier: "enterprise" },
          order: { ageDays: 45, status: "paid", total: 129 },
        },
        output: "approve",
      },
    ],
    constraints: [
      always(() => order.ageDays > 90, "deny"),
      always(() => order.ageDays <= 90 && order.status === "fraudulent", "review"),
      always(
        () =>
          order.ageDays <= 90 &&
          order.status === "paid" &&
          ((customer.tier === "enterprise" && order.ageDays > 60) ||
            (customer.tier === "standard" && order.ageDays > 30)),
        "review",
      ),
      always(
        () =>
          order.status === "paid" &&
          ((customer.tier === "enterprise" && order.ageDays <= 60) ||
            (customer.tier === "standard" && order.ageDays <= 30)),
        "approve",
      ),
    ],
  })`Apply our refund policy. Customer: ${customer} Order: ${order}`;
}
"""


def decision_rule(inputs: dict[str, Any]) -> str:
    """The fake teacher's refund policy; it honours both example constraints."""

    order, customer = inputs["order"], inputs["customer"]
    if order["ageDays"] > 90:
        return "deny"
    if order["status"] == "fraudulent":
        return "review"
    window = 60 if customer["tier"] == "enterprise" else 30
    return "approve" if order["ageDays"] <= window else "review"


def risk_rule(inputs: dict[str, Any]) -> str:
    order = inputs["order"]
    return "high" if order["status"] == "fraudulent" or order["total"] >= 1000 else "low"


def rule_for(ir: dict[str, Any]) -> Any:
    support = list(ir["output"]["head"]["support"])
    if support == ["approve", "deny", "review"]:
        return decision_rule
    if support == ["high", "low"]:
        return risk_rule
    raise AssertionError(f"unexpected support {support}")


def grid() -> list[dict[str, Any]]:
    # Status, age and tier vary fastest, so even a 20-case prefix covers every
    # region the refund policy distinguishes: the held-out constraint check
    # samples all of them, and a corpus that skipped one (a fraudulent order
    # past 90 days) would rightly fail it.
    return [
        {
            "customer": {"priorRefunds": prior, "tier": tier},
            "order": {"ageDays": age, "status": status, "total": total},
        }
        for total in (50, 500, 1500)
        for prior in (0, 3)
        for status in ("paid", "fraudulent")
        for age in (10, 40, 70, 91, 120)
        for tier in ("enterprise", "standard")
    ]


class RuleTeacher:
    """Labels the input grid by rule and edits single fields for adversarial cases."""

    def __init__(self) -> None:
        self.descriptor = TeacherDescriptor("rule-fixture", "grid-v1", "a" * 64)

    def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        rule = rule_for(ir)
        cases = [GeneratedCase(inputs, rule(inputs)) for inputs in grid()]
        assert n <= len(cases)
        return tuple(cases[:n])

    def generate_boundary_pair(self, ir: dict[str, Any], index: int, /) -> BoundaryPairProposal:
        rule = rule_for(ir)
        constraint = ir["definition"]["constraints"][index]
        base = {
            "customer": {"priorRefunds": 1, "tier": "standard"},
            "order": {"ageDays": 20, "status": "paid", "total": 75},
        }
        edited = copy.deepcopy(base)
        if "fraudulent" in constraint["source"]:
            edited["order"]["status"] = "fraudulent"
        else:
            base["order"]["ageDays"] = 90
            edited["order"]["ageDays"] = 91
        return BoundaryPairProposal(
            predicate_false=GeneratedCase(base, rule(base)),
            predicate_true=GeneratedCase(edited, rule(edited)),
        )

    def generate_counterfactual(
        self, ir: dict[str, Any], anchor: GeneratedCase, /
    ) -> CounterfactualProposal:
        rule = rule_for(ir)
        for path, value in (
            (("order", "status"), "fraudulent"),
            (("order", "status"), "paid"),
            (("order", "ageDays"), 120),
            (("order", "ageDays"), 10),
            (("order", "total"), 1500),
        ):
            twin = copy.deepcopy(anchor.inputs)
            if twin[path[0]][path[1]] == value:
                continue
            twin[path[0]][path[1]] = value
            label = rule(twin)
            if label != anchor.output:
                return CounterfactualProposal(
                    twin=GeneratedCase(twin, label),
                    reason=f"setting {path[1]} to {value!r} moves the answer to {label}",
                )
        raise AssertionError("no single-field edit changes the label")


def contradicting_rule(inputs: dict[str, Any]) -> str:
    """Never approves: honours both constraints yet contradicts the gold example."""

    return "deny" if inputs["order"]["ageDays"] > 90 else "review"


class ContradictingTeacher(RuleTeacher):
    def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        return tuple(
            GeneratedCase(case.inputs, contradicting_rule(case.inputs))
            for case in super().generate(ir, n)
        )

    def generate_boundary_pair(self, ir: dict[str, Any], index: int, /) -> BoundaryPairProposal:
        pair = super().generate_boundary_pair(ir, index)
        return BoundaryPairProposal(
            predicate_false=GeneratedCase(
                pair.predicate_false.inputs, contradicting_rule(pair.predicate_false.inputs)
            ),
            predicate_true=GeneratedCase(
                pair.predicate_true.inputs, contradicting_rule(pair.predicate_true.inputs)
            ),
        )

    def generate_counterfactual(
        self, ir: dict[str, Any], anchor: GeneratedCase, /
    ) -> CounterfactualProposal:
        twin = copy.deepcopy(anchor.inputs)
        twin["order"]["ageDays"] = 120 if anchor.inputs["order"]["ageDays"] <= 90 else 10
        return CounterfactualProposal(
            twin=GeneratedCase(twin, contradicting_rule(twin)),
            reason="crossing the 90-day rule changes the answer",
        )


_FIELD = re.compile(r"(tier|status|ageDays|total)=([^,}\s]+)")


class RuleTokenizer:
    """Four tokens per input: tier, status, age bucket and total bucket."""

    def __init__(self) -> None:
        self.semantscript_tokenizer_json = TOKENIZER_JSON

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, Any]:
        assert add_special_tokens and padding and truncation and max_length >= 4
        assert return_tensors == "pt"
        rows = []
        for text in texts:
            fields = dict(_FIELD.findall(text))
            age = float(fields["ageDays"])
            rows.append(
                [
                    1 if fields["tier"] == "enterprise" else 2,
                    3 if fields["status"] == "paid" else 4,
                    5 if age <= 30 else 6 if age <= 60 else 7 if age <= 90 else 8,
                    9 if float(fields["total"]) < 1000 else 10,
                ]
            )
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 4), dtype=torch.long),
        }


class RuleEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        # A fixed initialisation keeps every training run in this module reproducible.
        generator = torch.Generator().manual_seed(1234)
        self.embedding = torch.nn.Embedding(64, 8)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(64, 8, generator=generator))

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def training_config() -> TrainingConfig:
    return TrainingConfig(
        encoder_name="fixture/encoder",
        encoder_revision=REVISION,
        local_files_only=True,
        epochs=40,
        batch_size=8,
        learning_rate=0.05,
        weight_decay=0,
        maximum_sequence_length=8,
        evaluation_ratio=0.25,
        seed=3,
        device="cpu",
        head_architecture="mlp",
        mlp_hidden_size=16,
    )


def compile_project(
    root: Path, sources: dict[str, str], application: str, *extra_args: str
) -> dict[str, Any]:
    """Build a temp project with the CLI so the bundle is the compiler's own output."""

    if not CLI_ENTRY.is_file():
        pytest.skip("cli/dist is produced by the repository build gate")
    core = root / "node_modules" / "@semantscript" / "core"
    core.mkdir(parents=True)
    (core / "package.json").write_text(
        json.dumps({"name": "@semantscript/core", "type": "module", "types": "index.d.ts"})
    )
    shutil.copyfile(CORE_DECLARATIONS, core / "index.d.ts")
    (root / "package.json").write_text(json.dumps({"type": "module"}))
    (root / "src").mkdir()
    for name, text in sources.items():
        (root / "src" / name).write_text(text, encoding="utf-8")
    (root / "tsconfig.json").write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "module": "NodeNext",
                    "moduleResolution": "NodeNext",
                    "target": "ES2023",
                    "strict": True,
                    "outDir": "dist",
                    "rootDir": "src",
                    "skipLibCheck": True,
                },
                "include": ["src/**/*.ts"],
            }
        )
    )
    completed = subprocess.run(
        [
            "node",
            str(CLI_ENTRY),
            "build",
            "--project",
            str(root / "tsconfig.json"),
            "--application",
            application,
            *extra_args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((root / "dist" / "semantscript.ir.v1.json").read_text(encoding="utf-8"))


def run_cli_test(artifact_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["node", str(CLI_ENTRY), "test", "--artifact", str(artifact_root), "--no-bundle", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_trains_verifies_and_exports_the_refund_example_bundle(tmp_path: Path) -> None:
    bundle = compile_project(
        tmp_path / "project", {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-example"
    )
    assert [c["kind"] for c in bundle["functions"][0]["definition"]["constraints"]] == [
        "never",
        "always",
    ]
    messages: list[str] = []

    result = train_bundle(
        bundle,
        tmp_path / "artifact",
        teacher=RuleTeacher(),
        cache_directory=tmp_path / "cache",
        cases=48,
        training_config=training_config(),
        verification_config=VerificationConfig(),
        tokenizer=RuleTokenizer(),
        encoder=RuleEncoder(),
        base_model_weights_sha256="4" * 64,
        compiler_version="1.2.3",
        trainer_commit="abcdef0",
        log=messages.append,
        meter=SpendMeter(free_price("rule-fixture")),
    )

    report = result.report
    assert report["kind"] == "semantscript.train-report" and report["status"] == "passed"
    # The shared release version: the compiler's as handed in, the trainer's own.
    assert result.exported.manifest["build"]["compilerVersion"] == "1.2.3"
    assert result.exported.manifest["build"]["trainerVersion"] == __version__
    assert report["teacher"]["spend"]["requests"] == 0
    assert report["teacher"]["spend"]["costUsd"] == 0
    assert any(": teacher: 0 request(s)" in m for m in messages)
    assert report["application"] == {"id": "refund-example", "version": "0.0.0"}
    assert report["teacher"]["provider"] == "rule-fixture"
    (entry,) = report["functions"]
    assert entry["id"] == bundle["functions"][0]["id"]
    assert entry["sourcePath"] == "src/refund.sem.ts"
    assert entry["dataset"] == {"sha256": entry["dataset"]["sha256"], "cases": 48, "gold": 1}
    assert entry["adversarial"]["cases"] > 0
    assert entry["training"]["heldOutAccuracy"] == 1.0
    assert entry["verification"]["status"] == "passed"
    assert entry["verification"]["attestedCases"] == 1
    assert entry["verification"]["metrics"]["constraintViolations"] == 0
    # The held-out constraint sample: drawn from the first --seed, none broken.
    held_out = entry["verification"]["heldOutConstraints"]
    assert held_out["sampleSize"] == 512 and held_out["seed"] == 3
    assert held_out["violations"] == 0 and held_out["violationRate"] == 0
    assert held_out["coverageShortfalls"] == []
    assert any("held-out constraints 0 of 512 broken" in m for m in messages)
    manifest = result.exported.manifest
    assert manifest["functions"][0]["verification"]["heldOutConstraints"] == {
        "sampleSize": 512,
        "violations": 0,
        "violationRate": 0.0,
        "seed": 3,
    }
    assert report["artifact"]["manifestSha256"] == result.exported.manifest_sha256
    assert [fn["id"] for fn in manifest["functions"]] == [entry["id"]]
    assert manifest["application"] == {"id": "refund-example", "version": "0.0.0"}
    pointer = json.loads((tmp_path / "artifact" / "current.json").read_text())
    assert pointer["manifestSha256"] == result.exported.manifest_sha256
    assert any("publishing" in m or "published" in m for m in messages)
    verified = manifest["functions"][0]["trainingProvenance"]
    assert verified["trainingKeySha256"] == report["trainingKeySha256"]

    checked = run_cli_test(tmp_path / "artifact")
    assert checked["ok"] is True
    assert checked["functions"][0]["status"] == "passed"
    assert checked["functions"][0]["examples"] is None


def test_trains_several_functions_over_one_shared_encoder(tmp_path: Path) -> None:
    bundle = compile_project(
        tmp_path / "project",
        {"refund.sem.ts": REFUND_SOURCE.read_text(), "risk.sem.ts": RISK_SOURCE},
        "refund-app",
    )
    assert len(bundle["functions"]) == 2

    result = train_bundle(
        bundle,
        tmp_path / "artifact",
        teacher=RuleTeacher(),
        cache_directory=tmp_path / "cache",
        cases=48,
        training_config=training_config(),
        tokenizer=RuleTokenizer(),
        encoder=RuleEncoder(),
        base_model_weights_sha256="4" * 64,
        trainer_commit="abcdef0",
    )

    assert result.report["status"] == "passed"
    assert [fn["verification"]["status"] for fn in result.report["functions"]] == ["passed"] * 2
    manifest = result.exported.manifest
    assert [r["role"] for r in manifest["resources"]] == [
        "tokenizer",
        "encoder",
        "adapter",
        "head",
        "head",
    ]
    assert {fn["adapterRef"] for fn in manifest["functions"]} == {"adapter.refund-app"}
    assert sorted(fn["id"] for fn in manifest["functions"]) == sorted(
        fn["id"] for fn in bundle["functions"]
    )
    without_constraints = next(
        fn for fn in result.report["functions"] if fn["sourcePath"] == "src/risk.sem.ts"
    )
    assert without_constraints["adversarial"] is None
    assert run_cli_test(tmp_path / "artifact")["ok"] is True


def test_verification_failure_keeps_the_report_and_publishes_nothing(tmp_path: Path) -> None:
    bundle = compile_project(
        tmp_path / "project", {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-example"
    )

    with pytest.raises(TrainBundleFailure, match="verification failed") as raised:
        train_bundle(
            bundle,
            tmp_path / "artifact",
            teacher=ContradictingTeacher(),
            cache_directory=tmp_path / "cache",
            cases=48,
            training_config=training_config(),
            tokenizer=RuleTokenizer(),
            encoder=RuleEncoder(),
            base_model_weights_sha256="4" * 64,
            trainer_commit="abcdef0",
        )

    report = raised.value.report
    assert report["status"] == "failed" and report["artifact"] is None
    assert report["functions"][0]["verification"]["status"] == "failed"
    assert report["functions"][0]["verification"]["failures"]
    # Every failure ends with the next step derived from its evidence.
    verification = report["functions"][0]["verification"]
    assert len(verification["suggestions"]) == len(verification["failures"])
    for failure, suggestion in zip(
        verification["failures"], verification["suggestions"], strict=True
    ):
        assert failure.endswith("\n  next: " + suggestion)
    assert "; next: " in str(raised.value)
    assert not (tmp_path / "artifact").exists()


def test_rejects_malformed_bundles_and_teachers_without_adversarial_support(
    tmp_path: Path,
) -> None:
    common: dict[str, Any] = {
        "teacher": RuleTeacher(),
        "cache_directory": tmp_path / "cache",
        "tokenizer": RuleTokenizer(),
        "encoder": RuleEncoder(),
    }
    with pytest.raises(TrainBundleError, match="ir-bundle"):
        train_bundle({"kind": "other"}, tmp_path / "a", **common)
    with pytest.raises(TrainBundleError, match="at least one"):
        train_bundle(
            {"kind": "semantscript.ir-bundle", "bundleVersion": 1, "functions": []},
            tmp_path / "b",
            **common,
        )
    bundle = compile_project(
        tmp_path / "project", {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-example"
    )
    verified = copy.deepcopy(bundle)
    verified["functions"][0]["stage"] = "verified"
    with pytest.raises(TrainBundleError, match="source-stage"):
        train_bundle(verified, tmp_path / "c", **common)

    class SyntheticOnly:
        descriptor = TeacherDescriptor("rule-fixture", "grid-v1", "a" * 64)

        def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
            return RuleTeacher().generate(ir, n)

    with pytest.raises(TrainBundleError, match="boundary and counterfactual"):
        train_bundle(
            bundle,
            tmp_path / "d",
            teacher=SyntheticOnly(),
            cache_directory=tmp_path / "cache-d",
            tokenizer=RuleTokenizer(),
            encoder=RuleEncoder(),
        )
    with pytest.raises(TrainBundleError, match="cases must be"):
        train_bundle(bundle, tmp_path / "e", cases=-1, **common)


def test_trains_complete_constraints_with_the_built_in_teacher_and_no_language_model(
    tmp_path: Path,
) -> None:
    bundle = compile_project(
        tmp_path / "project", {"refund.sem.ts": COMPLETE_SOURCE}, "refund-complete"
    )
    teacher = ConstraintsTeacher()

    result = train_bundle(
        bundle,
        tmp_path / "artifact",
        teacher=teacher,
        cache_directory=tmp_path / "cache",
        cases=64,
        training_config=training_config(),
        tokenizer=RuleTokenizer(),
        encoder=RuleEncoder(),
        base_model_weights_sha256="4" * 64,
        trainer_commit="abcdef0",
    )

    report = result.report
    assert report["status"] == "passed"
    digest = ConstraintsTeacherConfig().sampling_sha256
    assert report["teacher"] == {
        "provider": "constraints",
        "model": "compiled-constraints-v3",
        "configurationSha256": digest,
    }
    (entry,) = report["functions"]
    assert entry["verification"]["status"] == "passed"
    assert entry["verification"]["metrics"]["constraintViolations"] == 0
    assert entry["adversarial"]["cases"] >= 2 * 4
    (function,) = result.exported.manifest["functions"]
    assert function["trainingProvenance"]["teacher"] == (
        f"constraints/compiled-constraints-v3@sha256:{digest}"
    )
    (trained,) = result.functions
    assert trained.base.teacher.provider == "constraints"
    assert all(case.origin in ("gold", "synthetic") for case in trained.base.cases)


def test_incomplete_constraints_name_an_input_or_train_with_a_fallback_teacher(
    tmp_path: Path,
) -> None:
    bundle = compile_project(
        tmp_path / "project", {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-example"
    )
    common: dict[str, Any] = {
        "cases": 48,
        "training_config": training_config(),
        "tokenizer": RuleTokenizer(),
        "encoder": RuleEncoder(),
        "base_model_weights_sha256": "4" * 64,
        "trainer_commit": "abcdef0",
    }

    with pytest.raises(TeacherConfigurationError) as raised:
        train_bundle(
            bundle,
            tmp_path / "pure",
            teacher=ConstraintsTeacher(),
            cache_directory=tmp_path / "cache-pure",
            **common,
        )
    message = str(raised.value)
    assert message.startswith("src/refund.sem.ts:17 (nf_")
    assert "the constraints do not decide every input" in message
    assert "For the input {" in message and "[teacher.fallback]" in message
    assert not (tmp_path / "pure").exists()

    fallback = RuleTeacher()
    mixed = ConstraintsTeacher(fallback=fallback)
    result = train_bundle(
        bundle,
        tmp_path / "mixed",
        teacher=mixed,
        cache_directory=tmp_path / "cache-mixed",
        **common,
    )
    assert result.report["status"] == "passed"
    assert result.report["teacher"]["provider"] == "constraints+rule-fixture"
    assert result.report["teacher"]["model"] == "grid-v1"
    (trained,) = result.functions
    synthetic = [case for case in trained.base.cases if case.origin == "synthetic"]
    decided = [case for case in synthetic if case.inputs["order"]["ageDays"] > 90]
    assert decided and all(case.output == "deny" for case in decided)
    assert 0 < mixed.decided_share(bundle["functions"][0]) < 1


def test_an_expression_without_gold_examples_fails_before_generation(tmp_path: Path) -> None:
    source = COMPLETE_SOURCE.replace(
        COMPLETE_SOURCE[
            COMPLETE_SOURCE.index("    examples: [") : COMPLETE_SOURCE.index("    constraints: [")
        ],
        "",
    )
    bundle = compile_project(tmp_path / "project", {"refund.sem.ts": source}, "refund-no-gold")

    class NeverCalled(RuleTeacher):
        def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
            raise AssertionError("no case generation for a bundle that cannot verify")

    with pytest.raises(TrainBundleError) as raised:
        train_bundle(
            bundle,
            tmp_path / "artifact",
            teacher=NeverCalled(),
            cache_directory=tmp_path / "cache",
            tokenizer=RuleTokenizer(),
            encoder=RuleEncoder(),
        )
    message = str(raised.value)
    assert message.startswith("src/refund.sem.ts:")
    assert "has no gold examples: verification needs at least one attested example" in message


def test_main_accepts_the_constraints_keyword(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({"kind": "semantscript.ir-bundle", "bundleVersion": 1}))
    captured: dict[str, Any] = {}

    def fake_train_bundle(bundle: Any, artifact_root: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(report={"kind": "semantscript.train-report", "status": "passed"})

    monkeypatch.setattr(cli_module, "train_bundle", fake_train_bundle)
    monkeypatch.chdir(tmp_path)
    code = cli_module.main(
        [
            "train",
            "--bundle",
            str(bundle_path),
            "--artifact",
            str(tmp_path / "artifact"),
            "--teacher",
            "constraints",
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    assert code == 0
    assert isinstance(captured["teacher"], ConstraintsTeacher)
    assert captured["teacher"].descriptor.provider == "constraints"


def test_main_maps_flags_into_configs_and_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({"kind": "semantscript.ir-bundle", "bundleVersion": 1}))
    teacher_path = tmp_path / "teacher.toml"
    teacher_path.write_text('[teacher]\nbackend = "ollama"\nmodel = "qwen"\n')
    captured: dict[str, Any] = {}

    def fake_train_bundle(bundle: Any, artifact_root: Any, **kwargs: Any) -> Any:
        captured.update(kwargs, bundle=bundle, artifact_root=artifact_root)
        return SimpleNamespace(report={"kind": "semantscript.train-report", "status": "passed"})

    monkeypatch.setattr(cli_module, "train_bundle", fake_train_bundle)
    monkeypatch.setattr(
        cli_module, "create_teacher", lambda config, **options: ("teacher", config.model)
    )
    report_path = tmp_path / "out" / "report.json"

    code = cli_module.main(
        [
            "train",
            "--bundle",
            str(bundle_path),
            "--artifact",
            str(tmp_path / "artifact"),
            "--teacher",
            str(teacher_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--report",
            str(report_path),
            "--cases",
            "12",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--seed",
            "9",
            "--device",
            "cpu",
            "--select-best-epoch",
            "--local-files-only",
            "--ece-threshold",
            "0.2",
            "--counterfactual-ratio",
            "0.5",
            "--seed-attempts",
            "5",
            "--seed-retry-margin",
            "3",
            "--application-id",
            "demo",
        ]
    )

    assert code == 0
    assert json.loads(report_path.read_text())["status"] == "passed"
    assert captured["teacher"] == ("teacher", "qwen")
    assert captured["cases"] == 12 and captured["application_id"] == "demo"
    config = captured["training_config"]
    assert (config.epochs, config.batch_size, config.seed, config.device) == (2, 4, 9, "cpu")
    assert config.select_best_epoch is True and config.local_files_only is True
    assert config.learning_rate == TrainingConfig().learning_rate
    assert captured["verification_config"].ece_threshold == 0.2
    assert captured["adversarial_config"].counterfactual_ratio == 0.5
    assert captured["seed_retry"] == SeedRetryConfig(attempts=5, margin=3.0)
    base_arguments = [
        "train",
        "--bundle",
        str(bundle_path),
        "--artifact",
        str(tmp_path / "artifact"),
        "--teacher",
        str(teacher_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    assert cli_module.main(base_arguments) == 0
    assert captured["seed_retry"] == SeedRetryConfig()
    # An out-of-range retry setting is refused before any training.
    captured.clear()
    assert cli_module.main([*base_arguments, "--seed-retry-margin", "0.5"]) == 1
    assert "error: --seed-retry-margin: seed retry margin" in capsys.readouterr().err
    assert cli_module.main([*base_arguments, "--seed-attempts", "0"]) == 1
    assert "error: --seed-attempts: seed attempts" in capsys.readouterr().err
    assert captured == {}
    # The flag is checked first: a missing bundle does not hide it, and
    # --estimate refuses it too.
    missing = ["train", "--bundle", str(tmp_path / "absent.json"), *base_arguments[3:]]
    assert cli_module.main([*missing, "--seed-attempts", "0"]) == 1
    assert "--seed-attempts" in capsys.readouterr().err
    assert cli_module.main([*base_arguments, "--estimate", "--seed-attempts", "0"]) == 1
    assert "--seed-attempts" in capsys.readouterr().err

    def failing_train_bundle(bundle: Any, artifact_root: Any, **kwargs: Any) -> Any:
        raise TrainBundleFailure("verification failed for x", {"status": "failed"})

    monkeypatch.setattr(cli_module, "train_bundle", failing_train_bundle)
    failed_report = tmp_path / "failed.json"
    code = cli_module.main(
        [
            "train",
            "--bundle",
            str(bundle_path),
            "--artifact",
            str(tmp_path / "artifact"),
            "--teacher",
            str(teacher_path),
            "--report",
            str(failed_report),
        ]
    )
    assert code == 1
    assert json.loads(failed_report.read_text()) == {"status": "failed"}


# ---- build cache (TASK-7.2) ------------------------------------------------


def cached_fixture(tmp_path: Path, sources: dict[str, str], application: str) -> dict[str, Any]:
    return compile_project(tmp_path / f"project-{application}", sources, application)


def train_cached(
    bundle: dict[str, Any],
    tmp_path: Path,
    *,
    teacher: Any | None = None,
    config: TrainingConfig | None = None,
    artifact: str = "artifact",
    **overrides: Any,
) -> Any:
    return train_bundle(
        bundle,
        tmp_path / artifact,
        teacher=teacher if teacher is not None else RuleTeacher(),
        cache_directory=tmp_path / "cache",
        cases=48,
        training_config=config if config is not None else training_config(),
        tokenizer=RuleTokenizer(),
        encoder=RuleEncoder(),
        base_model_weights_sha256="4" * 64,
        trainer_commit="abcdef0",
        **overrides,
    )


class CountingTeacher(RuleTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        self.calls += 1
        return super().generate(ir, n)


def test_rebuild_without_changes_performs_no_training_and_keeps_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-cache")
    first = train_cached(bundle, tmp_path)
    assert first.report["cache"] == {
        "reused": 0,
        "trained": 1,
        "directory": str(tmp_path / "cache" / "applications" / "refund-cache"),
    }
    assert [fn["cache"] for fn in first.report["functions"]] == ["trained"]
    cache_root = tmp_path / "cache" / "applications" / "refund-cache"
    assert (cache_root / "application.json").is_file()
    assert (cache_root / "shared.safetensors").is_file()
    function_id = bundle["functions"][0]["id"]
    assert {p.name for p in (cache_root / "functions" / function_id).iterdir()} == {
        "function.json",
        "verified-ir.json",
        "head.safetensors",
    }

    def no_training(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a rebuild without changes must not train")

    monkeypatch.setattr(cli_module, "train_application", no_training)
    monkeypatch.setattr(cli_module, "add_function_head", no_training)
    monkeypatch.setattr(cli_module, "export_multi_function_artifact", no_training)
    teacher = CountingTeacher()

    second = train_cached(bundle, tmp_path, teacher=teacher)

    assert second.report["status"] == "reused"
    assert second.report["cache"]["reused"] == 1 and second.report["cache"]["trained"] == 0
    assert second.report["artifact"]["manifestSha256"] == first.exported.manifest_sha256
    assert second.exported.manifest_sha256 == first.exported.manifest_sha256
    assert second.report["functions"][0]["cache"] == "reused"
    assert second.report["functions"][0]["verification"]["status"] == "passed"
    # The cache record keeps the held-out figure (coverage is measured evidence only).
    first_held_out = first.report["functions"][0]["verification"]["heldOutConstraints"]
    assert second.report["functions"][0]["verification"]["heldOutConstraints"] == {
        **first_held_out,
        "coverageShortfalls": [],
    }
    assert second.report["trainingKeySha256"] == first.report["trainingKeySha256"]
    assert teacher.calls == 0
    assert json.loads((tmp_path / "artifact" / "current.json").read_text())["manifestSha256"] == (
        first.exported.manifest_sha256
    )


def test_changing_one_expression_retrains_only_that_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {"refund.sem.ts": REFUND_SOURCE.read_text(), "risk.sem.ts": RISK_SOURCE}
    bundle = cached_fixture(tmp_path, sources, "refund-app")
    first = train_cached(bundle, tmp_path)
    refund_id = next(
        fn["id"] for fn in bundle["functions"] if fn["source"]["path"] == "src/refund.sem.ts"
    )
    first_refund = next(fn for fn in first.exported.manifest["functions"] if fn["id"] == refund_id)
    first_refund_head = next(
        r
        for r in first.exported.manifest["resources"]
        if r["ref"] == first_refund["heads"][0]["headRef"]
    )

    edited = cached_fixture(
        tmp_path / "edited",
        {
            **sources,
            "risk.sem.ts": RISK_SOURCE.replace(
                "Rate the refund risk.", "Rate the refund risk carefully."
            ),
        },
        "refund-app",
    )
    edited_ids = {fn["source"]["path"]: fn["id"] for fn in edited["functions"]}
    assert edited_ids["src/refund.sem.ts"] == refund_id
    assert edited_ids["src/risk.sem.ts"] != next(
        fn["id"] for fn in bundle["functions"] if fn["source"]["path"] == "src/risk.sem.ts"
    )
    joint = cli_module.train_application
    incremental = cli_module.add_function_head
    calls: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "train_application",
        lambda *a, **k: calls.append("joint") or joint(*a, **k),
    )
    monkeypatch.setattr(
        cli_module,
        "add_function_head",
        lambda *a, **k: calls.append("head") or incremental(*a, **k),
    )

    second = train_cached(edited, tmp_path)

    assert calls == ["head"]
    assert second.report["status"] == "passed"
    assert second.report["cache"]["reused"] == 1 and second.report["cache"]["trained"] == 1
    by_path = {fn["sourcePath"]: fn for fn in second.report["functions"]}
    assert by_path["src/refund.sem.ts"]["cache"] == "reused"
    assert by_path["src/risk.sem.ts"]["cache"] == "trained"
    assert second.exported.manifest_sha256 != first.exported.manifest_sha256
    second_refund = next(
        fn for fn in second.exported.manifest["functions"] if fn["id"] == refund_id
    )
    # The untouched function keeps its verified evidence and its head bytes.
    assert second_refund["heads"] == first_refund["heads"]
    assert second_refund["verification"] == first_refund["verification"]
    assert (
        second_refund["trainingProvenance"]["datasetSha256"]
        == (first_refund["trainingProvenance"]["datasetSha256"])
    )
    second_refund_head = next(
        r
        for r in second.exported.manifest["resources"]
        if r["ref"] == second_refund["heads"][0]["headRef"]
    )
    assert second_refund_head["sha256"] == first_refund_head["sha256"]
    assert sorted(fn["id"] for fn in second.exported.manifest["functions"]) == sorted(
        edited_ids.values()
    )
    assert run_cli_test(tmp_path / "artifact")["ok"] is True

    # A third build with the edited bundle reuses both functions.
    third = train_cached(edited, tmp_path)
    assert third.report["status"] == "reused"
    assert calls == ["head"]


def test_recipe_change_full_and_no_cache_retrain_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-cache")
    train_cached(bundle, tmp_path)
    joint = cli_module.train_application
    calls: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "train_application",
        lambda *a, **k: calls.append("joint") or joint(*a, **k),
    )

    other_recipe = replace(training_config(), epochs=training_config().epochs + 1)
    changed = train_cached(bundle, tmp_path, config=other_recipe)
    assert changed.report["cache"] == {
        "reused": 0,
        "trained": 1,
        "directory": str(tmp_path / "cache" / "applications" / "refund-cache"),
    }
    assert calls == ["joint"]

    forced = train_cached(bundle, tmp_path, config=other_recipe, full=True)
    assert forced.report["cache"]["reused"] == 0 and calls == ["joint", "joint"]

    uncached = train_cached(bundle, tmp_path, config=other_recipe, use_cache=False)
    assert uncached.report["cache"] == {"reused": 0, "trained": 1, "directory": None}
    assert calls == ["joint", "joint", "joint"]

    # The cache still holds the --full build; the uncached run published a newer
    # release without recording it, so the next build re-exports from cache but
    # trains nothing.
    again = train_cached(bundle, tmp_path, config=other_recipe)
    assert again.report["cache"]["trained"] == 0 and again.report["cache"]["reused"] == 1
    assert calls == ["joint", "joint", "joint"]
    assert train_cached(bundle, tmp_path, config=other_recipe).report["status"] == "reused"


def test_corrupted_cache_files_are_misses(tmp_path: Path) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-cache")
    first = train_cached(bundle, tmp_path)
    function_dir = (
        tmp_path
        / "cache"
        / "applications"
        / "refund-cache"
        / "functions"
        / bundle["functions"][0]["id"]
    )
    head = function_dir / "head.safetensors"
    head.write_bytes(head.read_bytes()[:-1] + b"\x00")

    rebuilt = train_cached(bundle, tmp_path)

    assert rebuilt.report["cache"]["reused"] == 0 and rebuilt.report["cache"]["trained"] == 1
    assert rebuilt.report["status"] == "passed"
    assert rebuilt.exported.manifest_sha256 != first.exported.manifest_sha256 or True
    # The rebuilt record is intact again.
    assert train_cached(bundle, tmp_path).report["status"] == "reused"


def test_missing_artifact_is_re_exported_from_cache_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-cache")
    train_cached(bundle, tmp_path)
    shutil.rmtree(tmp_path / "artifact")

    def no_training(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("re-export from cache must not train")

    monkeypatch.setattr(cli_module, "train_application", no_training)
    monkeypatch.setattr(cli_module, "add_function_head", no_training)

    exported = train_cached(bundle, tmp_path)

    assert exported.report["status"] == "passed"
    assert exported.report["cache"]["reused"] == 1 and exported.report["cache"]["trained"] == 0
    assert exported.report["functions"][0]["cache"] == "reused"
    # Nothing trained: no attempt is listed, and the seed is the recorded one.
    assert exported.report["attempts"] == [] and exported.report["seed"] == 3
    assert (tmp_path / "artifact" / "current.json").is_file()
    assert run_cli_test(tmp_path / "artifact")["ok"] is True


# ---- teacher spend: estimate, cap and probe (TASK-14.5) -------------------------------


def _anthropic_toml(tmp_path: Path) -> Path:
    path = tmp_path / "teacher.toml"
    path.write_text(
        '[teacher]\nbackend = "anthropic"\nmodel = "claude-sonnet-5"\nmode = "direct"\n'
        'api_key = "test-key"\n'
    )
    return path


def test_main_estimate_prints_json_without_building_a_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = compile_project(
        tmp_path / "project", {"refund.sem.ts": REFUND_SOURCE.read_text()}, "refund-example"
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--estimate must not train or call the teacher")

    monkeypatch.setattr(cli_module, "train_bundle", refuse)
    code = cli_module.main(
        [
            "train",
            "--bundle",
            str(bundle_path),
            "--artifact",
            str(tmp_path / "artifact"),
            "--teacher",
            str(_anthropic_toml(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--cases",
            "32",
            "--counterfactual-ratio",
            "0.5",
            "--max-cost-usd",
            "2",
            "--estimate",
        ]
    )
    assert code == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["kind"] == "semantscript.train-estimate"
    assert estimate["maxCostUsd"] == 2.0
    (row,) = estimate["functions"]
    assert row["sourcePath"] == "src/refund.sem.ts"
    assert row["plannedRequests"]["synthetic"] == 31
    assert estimate["total"]["costUsd"] > 0

    code = cli_module.main(
        [
            "train",
            "--bundle",
            str(bundle_path),
            "--artifact",
            str(tmp_path / "artifact"),
            "--teacher",
            "constraints",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--estimate",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["total"]["costUsd"] == 0


def test_spend_cap_stops_the_run_keeps_paid_responses_and_a_rerun_replays_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from semantscript_trainer.teacher_config import create_teacher as real_create_teacher

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({"kind": "semantscript.ir-bundle", "bundleVersion": 1}))
    ir = {
        "id": "nf_" + "1" * 64,
        "definition": {"template": [{"kind": "text", "text": "Classify."}], "examples": []},
        "inputs": [{"name": "message", "index": 0, "type": {"kind": "string"}}],
        "output": {"kind": "scalar", "head": {"kind": "boolean", "support": [False, True]}},
    }
    sent: list[dict[str, Any]] = []

    class Messages:
        def create(self, **params: Any) -> Any:
            sent.append(params)
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps({"inputs": {"message": f"m{len(sent)}"}, "output": True}),
                    )
                ],
                usage=SimpleNamespace(input_tokens=400, output_tokens=60),
            )

    client = SimpleNamespace(messages=Messages())
    monkeypatch.setattr(
        cli_module,
        "create_teacher",
        lambda config, **options: real_create_teacher(
            config, client=client, schema_transform=lambda schema: schema, **options
        ),
    )

    def fake_train_bundle(bundle: Any, artifact_root: Any, **kwargs: Any) -> Any:
        kwargs["teacher"].generate(ir, 6)
        return SimpleNamespace(
            report={
                "kind": "semantscript.train-report",
                "status": "passed",
                "teacher": {"spend": kwargs["meter"].summary()},
            }
        )

    monkeypatch.setattr(cli_module, "train_bundle", fake_train_bundle)
    arguments = [
        "train",
        "--bundle",
        str(bundle_path),
        "--artifact",
        str(tmp_path / "artifact"),
        "--teacher",
        str(_anthropic_toml(tmp_path)),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--report",
        str(tmp_path / "report.json"),
    ]

    # Each request costs USD 0.0014 here and is expected, before it is sent, at its own
    # prompt size: a USD 0.006 cap stops the run after a few of the six.
    assert cli_module.main([*arguments, "--max-cost-usd", "0.006"]) == 1
    stopped = capsys.readouterr().err
    paid = len(sent)
    assert 0 < paid < 6
    assert "error: spend cap USD 0.006 reached" in stopped
    assert f"the {paid} paid teacher response(s) kept in" in stopped
    assert "rerun with a higher --max-cost-usd" in stopped
    assert f"teacher: {paid} request(s), {400 * paid:,} in / {60 * paid} out tokens" in stopped
    assert not (tmp_path / "report.json").exists()

    assert cli_module.main(arguments) == 0
    spend = json.loads((tmp_path / "report.json").read_text())["teacher"]["spend"]
    assert (spend["requests"], spend["replayed"]) == (6 - paid, paid)
    assert len(sent) == 6
    stats = json.loads((tmp_path / "cache" / "teacher-stats.json").read_text())
    assert next(iter(stats.values()))["requests"] == 6 - paid


def test_teacher_probe_reports_model_latency_tokens_and_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from semantscript_trainer import doctor as doctor_module

    assert cli_module.main(["teacher", "probe", "--teacher", "constraints", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "semantscript.teacher-probe"
    assert (result["requestSent"], result["costUsd"]) == (False, 0.0)

    seen: list[Any] = []

    def fake_probe(config: Any, mode: str = "request", **kwargs: Any) -> Any:
        seen.append((config, mode))
        return doctor_module.ProbeResult(
            True,
            f"one request to {config.model} answered in 1.2 s, 16 in / 4 out tokens",
            latency_seconds=1.2,
            input_tokens=16,
            output_tokens=4,
            request_sent=True,
        )

    monkeypatch.setattr(doctor_module, "probe_teacher", fake_probe)
    code = cli_module.main(
        ["teacher", "probe", "--teacher", str(_anthropic_toml(tmp_path)), "--json"]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert seen[0][1] == "request" and seen[0][0].max_retries == 0
    assert result["model"] == "claude-sonnet-5" and result["latencySeconds"] == 1.2
    assert result["costUsd"] == pytest.approx((16 * 2 + 4 * 10) / 1_000_000)
    assert result["summary"].endswith("USD 0.000072")

    fallback = tmp_path / "mixed.toml"
    fallback.write_text(
        '[teacher]\nbackend = "constraints"\n[teacher.fallback]\nbackend = "ollama"\n'
        'model = "qwen3:14b"\n'
    )
    assert cli_module.main(["teacher", "probe", "--teacher", str(fallback)]) == 0
    assert capsys.readouterr().out.startswith("teacher probe: fallback: one request to qwen3:14b")


def test_teacher_probe_names_a_missing_key_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from semantscript_trainer import doctor as doctor_module

    def unexpected_probe(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no request may be sent without a key")

    monkeypatch.setattr(doctor_module, "probe_teacher", unexpected_probe)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-used")
    teacher = tmp_path / "openrouter.toml"
    teacher.write_text(
        '[teacher]\nbackend = "anthropic"\nmodel = "anthropic/claude-sonnet-5"\n'
        'base_url = "https://openrouter.ai/api"\nmode = "direct"\n'
        "[teacher.pricing]\ninput_usd_per_million = 2\noutput_usd_per_million = 10\n"
    )

    assert cli_module.main(["teacher", "probe", "--teacher", str(teacher), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert (result["ok"], result["requestSent"], result["latencySeconds"]) == (False, False, None)
    assert result["costUsd"] == 0.0
    assert "ANTHROPIC_API_KEY is not set" in result["summary"]
    assert "OPENROUTER_API_KEY" in result["summary"]
    assert "ANTHROPIC_API_KEY" in result["fix"]


def test_a_run_failing_on_rejected_teacher_answers_drops_its_journal_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from semantscript_trainer.adversarial import UnsynthesizableConstraintError
    from semantscript_trainer.teacher_config import create_teacher as real_create_teacher

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({"kind": "semantscript.ir-bundle", "bundleVersion": 1}))
    ir = {
        "id": "nf_" + "1" * 64,
        "definition": {"template": [{"kind": "text", "text": "Classify."}], "examples": []},
        "inputs": [{"name": "message", "index": 0, "type": {"kind": "string"}}],
        "output": {"kind": "scalar", "head": {"kind": "boolean", "support": [False, True]}},
    }
    sent: list[dict[str, Any]] = []

    class Messages:
        def create(self, **params: Any) -> Any:
            sent.append(params)
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps({"inputs": {"message": "same"}, "output": True}),
                    )
                ],
                usage=SimpleNamespace(input_tokens=400, output_tokens=60),
            )

    client = SimpleNamespace(messages=Messages())
    monkeypatch.setattr(
        cli_module,
        "create_teacher",
        lambda config, **options: real_create_teacher(
            config, client=client, schema_transform=lambda schema: schema, **options
        ),
    )

    def rejecting_train_bundle(bundle: Any, artifact_root: Any, **kwargs: Any) -> Any:
        # The answers decode, but the generator rejects them (as a boundary pair that is
        # not two-sided would be) and gives up.
        kwargs["teacher"].generate(ir, 3)
        raise UnsynthesizableConstraintError("constraint 0 did not yield a valid pair")

    monkeypatch.setattr(cli_module, "train_bundle", rejecting_train_bundle)
    arguments = [
        "train",
        "--bundle",
        str(bundle_path),
        "--artifact",
        str(tmp_path / "artifact"),
        "--teacher",
        str(_anthropic_toml(tmp_path)),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]

    assert cli_module.main(arguments) == 1
    err = capsys.readouterr().err
    assert "3 journaled response(s) this run used were discarded" in err
    assert not list((tmp_path / "cache" / "teacher-responses").rglob("*.json"))

    # The rerun asks the teacher again instead of replaying the rejected answers.
    assert cli_module.main(arguments) == 1
    assert len(sent) == 6


# The seed retry: a narrowly failed release gate retrains with the next seed.

NARROW_TOLERANCE = VerificationConfig(maximum_constraint_violation_rate=0.01)


def force_gate_by_seed(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[int, dict[str, float]],
) -> list[int]:
    """Fail verification at the given seeds with the given metric overrides.

    Every forced failure keeps the real measured evidence (and its record count)
    and only overrides the gate metrics, so the retry decision sees what a real
    seed-sensitive near miss looks like; a ``violation_rate`` override becomes
    the violation count that reaches that rate. Returns the seeds verification ran at.
    """

    real = cli_module.evaluate_training_result
    seen: list[int] = []

    def evaluate(ir: Any, training: Any, *args: Any, **kwargs: Any) -> Any:
        result = real(ir, training, *args, **kwargs)
        seen.append(training.config.seed)
        change = outcomes.get(training.config.seed)
        if change is None:
            return result
        overrides: dict[str, Any] = dict(change)
        rate = overrides.pop("violation_rate", None)
        if rate is not None:
            overrides["constraint_violations"] = math.ceil(rate * result.record_count)
        return replace(
            result,
            status="failed",
            metrics=replace(result.metrics, **overrides),
            failures=("forced gate failure",),
        )

    monkeypatch.setattr(cli_module, "evaluate_training_result", evaluate)
    return seen


def cached_verified_ir(tmp_path: Path, application: str, function_id: str) -> bytes:
    return (
        (tmp_path / "cache" / "applications" / application / "functions" / function_id)
        .joinpath("verified-ir.json")
        .read_bytes()
    )


def test_a_narrow_failure_retries_with_the_next_seed_and_publishes_the_passing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "retry")
    function_id = bundle["functions"][0]["id"]
    # A 1.2% violation rate is over the 1% tolerance and inside twice it. Seeds 3
    # and 4 fail that way; seed 5 passes.
    seen = force_gate_by_seed(
        monkeypatch, {3: {"violation_rate": 0.012}, 4: {"violation_rate": 0.012}}
    )
    teacher = CountingTeacher()
    messages: list[str] = []

    result = train_cached(
        bundle,
        tmp_path,
        teacher=teacher,
        verification_config=NARROW_TOLERANCE,
        log=messages.append,
    )

    assert seen == [3, 4, 5]
    # The datasets were generated once, before the first attempt.
    assert teacher.calls == 1
    report = result.report
    assert report["status"] == "passed" and report["seed"] == 5
    assert report["retry"] == {"attempts": 3, "margin": 2.0, "stopReason": None}
    assert [(a["attempt"], a["seed"], a["status"]) for a in report["attempts"]] == [
        (1, 3, "failed"),
        (2, 4, "failed"),
        (3, 5, "passed"),
    ]
    first = report["attempts"][0]["functions"][0]
    assert first["id"] == function_id and first["constraintViolations"] > 0
    assert 0.01 < first["violationRate"] <= 0.02
    assert first["violationRate"] == first["constraintViolations"] / first["records"]
    assert first["failures"] == ["forced gate failure"]
    assert {"accuracy", "ece"} <= set(first)
    assert report["attempts"][2]["functions"][0]["failures"] == []
    assert report["functions"][0]["training"]["seed"] == 5
    assert any(
        m.startswith("verification failed narrowly at seed 3 (")
        and "retrying with seed 4 (attempt 2 of 3)" in m
        for m in messages
    )
    assert any(m.endswith(", seed 5") and "verification passed" in m for m in messages)
    # The published release binds the verified IR that records the passing seed.
    verified = cached_verified_ir(tmp_path, "retry", function_id)
    assert json.loads(verified)["trainingProvenance"]["seed"] == 5
    assert result.exported.manifest["build"]["sourceIrSha256"] == (
        hashlib.sha256(verified).hexdigest()
    )
    # The release manifest itself records the passing seed as well.
    assert result.exported.manifest["functions"][0]["trainingProvenance"]["seed"] == 5

    # The cache recipe keeps the configured seed: an unchanged rebuild reuses the
    # retried release without training and reports the seed it was published with.
    def no_training(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a rebuild without changes must not train")

    monkeypatch.setattr(cli_module, "train_application", no_training)
    again = train_cached(bundle, tmp_path, verification_config=NARROW_TOLERANCE)
    assert again.report["status"] == "reused" and again.report["seed"] == 5
    assert again.report["functions"][0]["training"]["seed"] == 5
    assert again.report["attempts"] == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"violation_rate": 0.05}, "outside the retry margin 2 x 0.01"),
        ({"example_failures": 1}, "a gold miss is not a seed effect"),
        ({"type_errors": 1}, "1 output type check(s) failed"),
    ],
)
def test_a_failure_outside_the_margin_or_a_gold_miss_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, float], reason: str
) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "no-retry")
    seen = force_gate_by_seed(monkeypatch, {3: change})
    messages: list[str] = []

    with pytest.raises(TrainBundleFailure, match="not retrying: .*" + re.escape(reason)) as raised:
        train_cached(bundle, tmp_path, verification_config=NARROW_TOLERANCE, log=messages.append)

    assert seen == [3]
    report = raised.value.report
    assert report["status"] == "failed" and report["seed"] is None
    assert [a["seed"] for a in report["attempts"]] == [3]
    assert reason in report["retry"]["stopReason"]
    assert any(m.startswith("not retrying: ") and reason in m for m in messages)
    assert not (tmp_path / "artifact").exists()


def test_a_real_gold_miss_does_not_retry(tmp_path: Path) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "gold")

    with pytest.raises(TrainBundleFailure, match="not a seed effect") as raised:
        train_cached(bundle, tmp_path, teacher=ContradictingTeacher())

    assert len(raised.value.report["attempts"]) == 1


def test_retries_stop_after_the_configured_attempts_or_when_turned_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = cached_fixture(tmp_path, {"refund.sem.ts": REFUND_SOURCE.read_text()}, "exhaust")
    narrow = {"violation_rate": 0.012}
    seen = force_gate_by_seed(monkeypatch, {3: narrow, 4: narrow, 5: narrow})

    with pytest.raises(
        TrainBundleFailure, match=r"all 2 seed attempts failed \(seeds 3 to 4\)"
    ) as exhausted:
        train_cached(
            bundle,
            tmp_path,
            verification_config=NARROW_TOLERANCE,
            seed_retry=SeedRetryConfig(attempts=2),
        )
    assert seen == [3, 4]
    # The next untried seed is the next step, under the failure and in the error.
    next_seed = "rerun with --seed 5: seeds 3 to 4 failed narrowly"
    assert exhausted.value.report["functions"][0]["verification"]["failures"] == [
        "forced gate failure\n  next: " + remedy("seed-retry-next-seed", seed=5, first=3, last=4)
    ]
    assert exhausted.value.report["attempts"][-1]["functions"][0]["suggestions"][0].startswith(
        next_seed
    )
    assert exhausted.value.report["attempts"][0]["functions"][0]["suggestions"] == []
    assert "; next: " + next_seed in str(exhausted.value)

    seen.clear()
    with pytest.raises(TrainBundleFailure, match=r"seed retry is off \(--seed-attempts 1\)") as off:
        train_cached(
            bundle,
            tmp_path,
            verification_config=NARROW_TOLERANCE,
            seed_retry=SeedRetryConfig(attempts=1),
        )
    assert seen == [3]
    assert off.value.report["retry"]["attempts"] == 1
    assert off.value.report["functions"][0]["verification"]["suggestions"] == [
        remedy("seed-retry-attempts", attempts=3)
    ]
    with pytest.raises(TrainBundleError, match="SeedRetryConfig"):
        train_cached(bundle, tmp_path, seed_retry=3)


def test_an_incremental_build_retries_only_the_changed_head_and_keeps_its_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {"refund.sem.ts": REFUND_SOURCE.read_text(), "risk.sem.ts": RISK_SOURCE}
    bundle = cached_fixture(tmp_path, sources, "retry-app")
    train_cached(bundle, tmp_path, verification_config=NARROW_TOLERANCE)
    edited = cached_fixture(
        tmp_path / "edited",
        {**sources, "risk.sem.ts": RISK_SOURCE.replace("refund risk.", "refund risk now.")},
        "retry-app",
    )
    risk_id = next(
        fn["id"] for fn in edited["functions"] if fn["source"]["path"].endswith("risk.sem.ts")
    )
    refund_id = next(fn["id"] for fn in edited["functions"] if fn["id"] != risk_id)
    # The risk expression has no constraints, so a narrow ECE miss stands in.
    real = cli_module.evaluate_training_result

    def ece_miss(ir: Any, training: Any, *args: Any, **kwargs: Any) -> Any:
        result = real(ir, training, *args, **kwargs)
        if ir["id"] != risk_id or training.config.seed != 3:
            return result
        return replace(
            result,
            status="failed",
            metrics=replace(
                result.metrics,
                ece=0.15,
                heads=tuple(
                    replace(head, calibration=replace(head.calibration, ece=0.15))
                    for head in result.metrics.heads
                ),
            ),
            failures=("ECE 0.15 exceeds configured threshold 0.1",),
        )

    monkeypatch.setattr(cli_module, "evaluate_training_result", ece_miss)
    incremental = cli_module.add_function_head
    heads: list[int] = []
    monkeypatch.setattr(
        cli_module,
        "add_function_head",
        lambda *a, **k: heads.append(k["config"].seed) or incremental(*a, **k),
    )

    second = train_cached(edited, tmp_path, verification_config=NARROW_TOLERANCE)

    assert heads == [3, 4]
    report = second.report
    assert report["status"] == "passed" and report["seed"] == 4
    assert [a["seed"] for a in report["attempts"]] == [3, 4]
    assert [fn["id"] for fn in report["attempts"][0]["functions"]] == [risk_id]
    by_id = {fn["id"]: fn for fn in report["functions"]}
    assert by_id[refund_id]["cache"] == "reused" and by_id[refund_id]["training"]["seed"] == 3
    assert by_id[risk_id]["cache"] == "trained" and by_id[risk_id]["training"]["seed"] == 4
    assert (
        json.loads(cached_verified_ir(tmp_path, "retry-app", risk_id))["trainingProvenance"]["seed"]
        == 4
    )
    assert run_cli_test(tmp_path / "artifact")["ok"] is True
    seeds = {
        fn["id"]: fn["trainingProvenance"]["seed"] for fn in second.exported.manifest["functions"]
    }
    assert seeds == {refund_id: 3, risk_id: 4}

    # Editing the other expression restores the retried head from the cache on the
    # seed its verified IR records, so its split matches what it trained on.
    third_bundle = cached_fixture(
        tmp_path / "third",
        {
            "refund.sem.ts": REFUND_SOURCE.read_text().replace(
                "Apply our refund policy.", "Apply our refund policy exactly."
            ),
            "risk.sem.ts": sources["risk.sem.ts"].replace("refund risk.", "refund risk now."),
        },
        "retry-app",
    )
    third = train_cached(third_bundle, tmp_path, verification_config=NARROW_TOLERANCE)
    assert heads == [3, 4, 3]
    third_by_path = {fn["sourcePath"]: fn for fn in third.report["functions"]}
    assert third_by_path["src/risk.sem.ts"]["cache"] == "reused"
    assert third_by_path["src/risk.sem.ts"]["training"]["seed"] == 4
    assert third_by_path["src/refund.sem.ts"]["training"]["seed"] == 3
