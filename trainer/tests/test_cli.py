"""The bundle-to-artifact driver behind ``semantscript train``."""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from semantscript_trainer import cli as cli_module  # noqa: E402
from semantscript_trainer.cli import (  # noqa: E402
    TrainBundleError,
    TrainBundleFailure,
    train_bundle,
)
from semantscript_trainer.teacher import (  # noqa: E402
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    TeacherDescriptor,
)
from semantscript_trainer.training import TrainingConfig  # noqa: E402
from semantscript_trainer.verification import VerificationConfig  # noqa: E402

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
    return [
        {
            "customer": {"priorRefunds": prior, "tier": tier},
            "order": {"ageDays": age, "status": status, "total": total},
        }
        for status in ("paid", "fraudulent")
        for age in (10, 40, 70, 91, 120)
        for tier in ("enterprise", "standard")
        for prior in (0, 3)
        for total in (50, 500, 1500)
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
        self.embedding = torch.nn.Embedding(64, 8)

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


def compile_project(root: Path, sources: dict[str, str], application: str) -> dict[str, Any]:
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
        ["node", str(CLI_ENTRY), "test", "--artifact", str(artifact_root), "--json"],
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
        trainer_commit="abcdef0",
        log=messages.append,
    )

    report = result.report
    assert report["kind"] == "semantscript.train-report" and report["status"] == "passed"
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
    manifest = result.exported.manifest
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


def test_main_maps_flags_into_configs_and_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(cli_module, "create_teacher", lambda config: ("teacher", config.model))
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
