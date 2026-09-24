"""Compile-time routed domains with depth routing, end to end (TASK-6.7)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from semantscript_trainer import cli as cli_module  # noqa: E402
from semantscript_trainer.cli import train_bundle  # noqa: E402

from .test_cli import (  # noqa: E402
    REFUND_SOURCE,
    RISK_SOURCE,
    RuleTeacher,
    RuleTokenizer,
    compile_project,
    run_cli_test,
    training_config,
)


class LayeredRuleEncoder(torch.nn.Module):
    """The CLI test encoder with three tiny layers and hidden states for depth routing."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=3)
        generator = torch.Generator().manual_seed(1234)
        self.embedding = torch.nn.Embedding(64, 8)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(3))
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(64, 8, generator=generator))
            # Exact identity layers: every depth sees the same embedding, so the
            # test exercises the routing plumbing rather than representation quality.
            for layer in self.layers:
                layer.weight.copy_(torch.eye(8))
                layer.bias.zero_()

    def forward(self, *, input_ids, attention_mask, return_dict, output_hidden_states=False):
        del attention_mask
        assert return_dict is True
        state = self.embedding(input_ids)
        states = [state]
        for layer in self.layers:
            state = layer(state)
            states.append(state)
        return SimpleNamespace(
            last_hidden_state=state,
            hidden_states=tuple(states) if output_hidden_states else None,
        )


ROUTED_RISK_SOURCE = RISK_SOURCE.replace(
    "Rate the refund risk.", "@domain(risk)\n  Rate the refund risk."
)


def routed_bundle(root: Path, risk_source: str = ROUTED_RISK_SOURCE) -> dict[str, Any]:
    return compile_project(
        root,
        {"refund.sem.ts": REFUND_SOURCE.read_text(), "risk.sem.ts": risk_source},
        "routed-app",
        "--domain-depth",
        "refund=1",
    )


def train_routed(bundle: dict[str, Any], tmp_path: Path, **overrides: Any) -> Any:
    return train_bundle(
        bundle,
        tmp_path / "artifact",
        teacher=RuleTeacher(),
        cache_directory=tmp_path / "cache",
        cases=48,
        training_config=training_config(),
        tokenizer=RuleTokenizer(),
        encoder=LayeredRuleEncoder(),
        base_model_weights_sha256="4" * 64,
        trainer_commit="abcdef0",
        **overrides,
    )


def test_routed_bundle_trains_one_adapter_per_domain_at_its_depth_and_node_runs_it(
    tmp_path: Path,
) -> None:
    bundle = routed_bundle(tmp_path / "project")
    domains = {d["name"]: d for d in bundle["executionPlan"]["domains"]}
    assert set(domains) == {"refund", "risk"}
    assert domains["refund"]["encoderDepth"] == 1 and domains["risk"]["encoderDepth"] is None
    assert domains["refund"]["adapterRef"] == "adapter.routed-app.refund"
    assert domains["refund"]["encoderRef"] == "encoder.routed-app.depth-001"
    messages: list[str] = []

    result = train_routed(bundle, tmp_path, log=messages.append)

    assert result.report["status"] == "passed"
    assert any("2 adapter(s) with depth routing" in m for m in messages), messages
    model = result.exported.functions[0].training.model if False else None  # noqa: F841
    manifest = result.exported.manifest
    roles = {}
    for resource in manifest["resources"]:
        roles.setdefault(resource["role"], []).append(resource)
    assert sorted(r["ref"] for r in roles["encoder"]) == [
        "encoder.routed-app",
        "encoder.routed-app.depth-001",
    ]
    assert sorted(r["path"] for r in roles["encoder"]) == [
        "models/encoder/depth-001.onnx",
        "models/encoder/model.onnx",
    ]
    assert sorted(r["ref"] for r in roles["adapter"]) == [
        "adapter.routed-app.refund",
        "adapter.routed-app.risk",
    ]
    assert manifest["model"]["encoderRef"] == "encoder.routed-app"
    by_path = {fn["id"]: fn for fn in manifest["functions"]}
    refund_id = next(
        f["id"] for f in bundle["functions"] if f["source"]["path"] == "src/refund.sem.ts"
    )
    risk_id = next(f["id"] for f in bundle["functions"] if f["source"]["path"] == "src/risk.sem.ts")
    assert by_path[refund_id]["encoderRef"] == "encoder.routed-app.depth-001"
    assert by_path[refund_id]["adapterRef"] == "adapter.routed-app.refund"
    assert "encoderRef" not in by_path[risk_id]
    assert by_path[risk_id]["adapterRef"] == "adapter.routed-app.risk"
    # The prefix graph is smaller than the full stack and both were parity-checked.
    sizes = {r["ref"]: r["byteLength"] for r in roles["encoder"]}
    assert sizes["encoder.routed-app.depth-001"] < sizes["encoder.routed-app"]
    # The trained refund view really embeds at depth 1.
    refund_training = (
        result.functions[0].training
        if result.functions[0].ir["id"] == refund_id
        else result.functions[1].training
    )
    assert refund_training.model.depth == 1
    # The Node runtime loads the routed artifact and replays every example.
    assert run_cli_test(tmp_path / "artifact")["ok"] is True


def test_editing_one_domain_leaves_the_other_domain_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = routed_bundle(tmp_path / "project")
    first = train_routed(bundle, tmp_path)
    refund_id = next(
        f["id"] for f in bundle["functions"] if f["source"]["path"] == "src/refund.sem.ts"
    )
    first_refund = next(fn for fn in first.exported.manifest["functions"] if fn["id"] == refund_id)
    first_resources = {r["ref"]: r["sha256"] for r in first.exported.manifest["resources"]}

    edited = routed_bundle(
        tmp_path / "edited",
        ROUTED_RISK_SOURCE.replace("Rate the refund risk.", "Rate the refund risk carefully."),
    )
    calls: list[str] = []
    joint, incremental = cli_module.train_application, cli_module.add_function_head
    monkeypatch.setattr(
        cli_module, "train_application", lambda *a, **k: calls.append("joint") or joint(*a, **k)
    )
    monkeypatch.setattr(
        cli_module,
        "add_function_head",
        lambda *a, **k: calls.append("head") or incremental(*a, **k),
    )

    second = train_routed(edited, tmp_path)

    # The risk domain's only function changed, so its adapter is retrained with its
    # head on the frozen encoder; the refund domain is reused byte for byte.
    assert calls == ["head"]
    assert second.report["cache"] == {
        "reused": 1,
        "trained": 1,
        "directory": str(tmp_path / "cache" / "applications" / "routed-app"),
    }
    second_refund = next(
        fn for fn in second.exported.manifest["functions"] if fn["id"] == refund_id
    )
    assert second_refund["heads"] == first_refund["heads"]
    assert second_refund["verification"] == first_refund["verification"]
    second_resources = {r["ref"]: r["sha256"] for r in second.exported.manifest["resources"]}
    for ref in (
        "encoder.routed-app",
        "encoder.routed-app.depth-001",
        "adapter.routed-app.refund",
        first_refund["heads"][0]["headRef"],
    ):
        assert second_resources[ref] == first_resources[ref], ref
    assert second_resources["adapter.routed-app.risk"] != first_resources["adapter.routed-app.risk"]
    assert run_cli_test(tmp_path / "artifact")["ok"] is True

    # A depth change for a domain is a cache miss for that domain only.
    deeper = compile_project(
        tmp_path / "deeper",
        {
            "refund.sem.ts": REFUND_SOURCE.read_text(),
            "risk.sem.ts": ROUTED_RISK_SOURCE.replace(
                "Rate the refund risk.", "Rate the refund risk carefully."
            ),
        },
        "routed-app",
        "--domain-depth",
        "refund=2",
    )
    third = train_routed(deeper, tmp_path)
    assert calls == ["head", "head"]
    assert third.report["cache"]["reused"] == 1 and third.report["cache"]["trained"] == 1
    third_resources = {r["ref"]: r["sha256"] for r in third.exported.manifest["resources"]}
    assert "encoder.routed-app.depth-002" in third_resources
    assert third_resources["adapter.routed-app.risk"] == second_resources["adapter.routed-app.risk"]
