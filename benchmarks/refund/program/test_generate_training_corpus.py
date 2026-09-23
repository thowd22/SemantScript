from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from benchmarks.refund.program.claude_cli_teacher import CLAUDE_CLI_VERSION, ClaudeCliTeacherConfig
from benchmarks.refund.program.generate_training_corpus import (
    MANIFEST_KIND,
    MANIFEST_NAME,
    CorpusGenerationError,
    generate_training_corpus,
    main,
)

STUB_SOURCE = f"""#!{sys.executable}
import json, sys
argv = sys.argv[1:]
if argv == ["--version"]:
    print({CLAUDE_CLI_VERSION!r}); raise SystemExit(0)
schema = json.loads(argv[argv.index("--json-schema") + 1])
text = sys.stdin.read()
payload, _ = json.JSONDecoder().raw_decode(text[text.index("{{"):])

def case(position, age=None):
    return {{
        "inputs": {{
            "customer": {{"priorRefunds": position, "tier": "standard" if position % 2 else "enterprise"}},
            "order": {{"ageDays": 10 + position if age is None else age, "status": "paid", "total": 100 + position}},
        }},
        "output": "approve" if (10 + position if age is None else age) <= 90 else "deny",
    }}

properties = schema["properties"]
if "predicateFalse" in properties:
    index = payload["selectedConstraint"]["index"]
    if index == 0:
        true_side = case(0); true_side["inputs"]["order"]["status"] = "fraudulent"; true_side["output"] = "review"
        out = {{"predicateFalse": case(0), "predicateTrue": true_side}}
    else:
        out = {{"predicateFalse": case(1, age=30), "predicateTrue": case(1, age=91)}}
elif "twin" in properties:
    anchor = payload["anchor"]
    twin = json.loads(json.dumps(anchor))
    twin["inputs"]["order"]["ageDays"] = 120
    twin["output"] = "deny"
    out = {{"twin": twin, "reason": "Only order.ageDays moved past the 90-day always-deny boundary."}}
else:
    out = case(payload["casePosition"]["index"] + 1)
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "num_turns": 3,
    "permission_denials": [], "modelUsage": {{"claude-sonnet-5": {{}}}}, "structured_output": out}}))
"""


@pytest.fixture
def stub_cli(tmp_path: Path) -> Path:
    stub = tmp_path / "claude-stub"
    stub.write_text(STUB_SOURCE)
    stub.chmod(0o700)
    return stub


def test_generates_frozen_corpus_with_manifest(tmp_path: Path, stub_cli: Path) -> None:
    output = tmp_path / "corpus"
    config = ClaudeCliTeacherConfig(executable=str(stub_cli), concurrency=2, timeout_seconds=60)

    manifest = generate_training_corpus(
        output,
        synthetic_count=4,
        counterfactual_ratio=0.5,
        config=config,
    )

    written = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert written == manifest
    assert manifest["kind"] == MANIFEST_KIND
    assert manifest["dataClassification"] == "synthetic-training-only"
    assert manifest["teacher"]["provider"] == "anthropic-claude-cli-training-only"
    assert manifest["teacher"]["cliVersion"] == CLAUDE_CLI_VERSION
    assert manifest["teacher"]["configuration"]["generation"]["concurrency"] == 2
    assert manifest["function"]["id"].startswith("nf_955824")
    assert (output / manifest["function"]["bundlePath"]).is_file()

    synthetic = manifest["synthetic"]
    assert synthetic["syntheticCount"] == 4
    assert synthetic["goldCount"] == 0
    assert synthetic["uniqueInputCount"] == 4
    assert synthetic["labelCounts"] == {'"approve"': 4}
    assert synthetic["runReport"]["requests"] == 4
    assert synthetic["runReport"]["residualDuplicates"] == 0
    assert (output / synthetic["cachePath"]).is_file()

    adversarial = manifest["adversarial"]
    assert adversarial["config"] == {"counterfactualRatio": 0.5, "maximumAttempts": 3}
    assert adversarial["caseCountsByTag"]["constraint-boundary"] == 4
    assert adversarial["pairCount"] == 2
    assert adversarial["baseDatasetSha256"] == synthetic["datasetSha256"]
    assert (output / adversarial["cachePath"]).is_file()


def test_refuses_nonempty_output_directory(tmp_path: Path, stub_cli: Path) -> None:
    output = tmp_path / "corpus"
    output.mkdir()
    (output / "stale").write_text("x")

    with pytest.raises(CorpusGenerationError, match="absent or empty"):
        generate_training_corpus(
            output,
            synthetic_count=1,
            counterfactual_ratio=0.0,
            config=ClaudeCliTeacherConfig(executable=str(stub_cli)),
        )


def test_main_reports_summary_and_failure(
    tmp_path: Path, stub_cli: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "corpus"
    status = main(
        [
            "--output-dir",
            str(output),
            "--synthetic-count",
            "2",
            "--counterfactual-ratio",
            "0",
            "--executable",
            str(stub_cli),
            "--require-cli-version",
            CLAUDE_CLI_VERSION,
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    summary = json.loads(captured.out)
    assert summary["synthetic"]["syntheticCount"] == 2
    assert summary["adversarial"]["pairCount"] == 0

    status = main(
        [
            "--output-dir",
            str(output),
            "--synthetic-count",
            "2",
            "--executable",
            str(stub_cli),
        ]
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "corpus generation failed" in captured.err
