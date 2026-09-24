"""Derive an N-function artifact from one verified scalar release without retraining.

The shared-encoder thesis (TASK-6.5) is about runtime cost, and a head's runtime
cost does not depend on its weights. So this script clones the source release's
one verified function N times over the same tokenizer, encoder and adapter: every
clone gets a fresh function id and head ref, its head graph is a byte copy of the
verified head, and the large shared resources are hard-linked (copied when the
filesystem refuses). The manifest keeps every other field of the source function,
so the runtime loader's contracts still hold, and the derivation record next to
the artifact names the source manifest digest so nobody mistakes the clones for
independently trained functions.

Usage:
    python derive_multihead_artifact.py --source <artifact-root> --output <artifact-root> --functions 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAXIMUM_FUNCTIONS = 64  # the runtime's largest stage; more clones would never be timed
_FUNCTION_PREFIX = "nf_"


class DerivationError(RuntimeError):
    """The source release cannot be cloned safely."""


def derive_multihead_artifact(
    source_root: Path,
    output_root: Path,
    *,
    functions: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write ``output_root`` as a content-addressed artifact with ``functions`` clones."""

    if isinstance(functions, bool) or not isinstance(functions, int):
        raise DerivationError("functions must be an integer")
    if not 1 <= functions <= MAXIMUM_FUNCTIONS:
        raise DerivationError(f"functions must be between 1 and {MAXIMUM_FUNCTIONS}")
    if output_root.exists():
        raise DerivationError(f"output {output_root} already exists")

    source_release, source_manifest, source_manifest_sha256 = _load_source(source_root)
    source_function, head_resource = _single_scalar_function(source_manifest)
    shared = [r for r in source_manifest["resources"] if r["role"] != "head"]

    staging = output_root / "staging"
    staging.mkdir(parents=True)
    try:
        for resource in shared:
            _link_or_copy(source_release / resource["path"], staging / resource["path"])
        head_bytes = (source_release / head_resource["path"]).read_bytes()
        head_sha256 = hashlib.sha256(head_bytes).hexdigest()
        if head_sha256 != head_resource["sha256"] or len(head_bytes) != head_resource["byteLength"]:
            raise DerivationError("source head resource does not match its manifest entry")

        clones: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = [dict(r) for r in shared]
        for index in range(functions):
            function_id, head_ref = _clone_identity(source_function["id"], index)
            path = f"models/heads/{function_id}/head-000.onnx"
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(head_bytes)
            resource = dict(head_resource)
            resource["ref"] = head_ref
            resource["path"] = path
            resources.append(resource)
            clone = json.loads(json.dumps(source_function))
            clone["id"] = function_id
            clone["heads"][0]["headRef"] = head_ref
            clones.append(clone)

        manifest = dict(source_manifest)
        manifest["application"] = {
            "id": f"{source_manifest['application']['id']}-multihead",
            "version": source_manifest["application"]["version"],
        }
        manifest["build"] = dict(source_manifest["build"])
        manifest["build"]["createdAt"] = created_at or _utc_now()
        manifest["functions"] = clones
        manifest["resources"] = resources
        manifest_bytes = _dump(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        (staging / "manifest.json").write_bytes(manifest_bytes)

        releases = output_root / "releases"
        releases.mkdir()
        release = releases / f"sha256-{manifest_sha256}"
        staging.rename(release)
        pointer = {
            "kind": "semantscript.artifact-pointer",
            "pointerVersion": 1,
            "release": f"releases/sha256-{manifest_sha256}",
            "manifestSha256": manifest_sha256,
        }
        (output_root / "current.json").write_bytes(_dump(pointer))
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise

    record = {
        "kind": "semantscript.multihead-derivation",
        "derivationVersion": 1,
        "createdAt": manifest["build"]["createdAt"],
        "sourceArtifactRoot": str(source_root),
        "sourceManifestSha256": source_manifest_sha256,
        "sourceFunctionId": source_function["id"],
        "sourceHeadSha256": head_sha256,
        "functions": functions,
        "functionIds": [clone["id"] for clone in clones],
        "manifestSha256": manifest_sha256,
        "note": (
            "Every function is a byte copy of the source function's verified head over the "
            "source encoder and adapter; the clones exist to time heads per stage, not to "
            "claim independently trained behaviour."
        ),
    }
    return record


def _load_source(source_root: Path) -> tuple[Path, dict[str, Any], str]:
    pointer_path = source_root / "current.json"
    if not pointer_path.is_file():
        raise DerivationError(f"source {source_root} has no current.json pointer")
    pointer = json.loads(pointer_path.read_bytes())
    release = source_root / pointer["release"]
    manifest_bytes = (release / "manifest.json").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != pointer.get("manifestSha256"):
        raise DerivationError("source manifest digest does not match its pointer")
    manifest = json.loads(manifest_bytes)
    if manifest.get("kind") != "semantscript.application-artifact":
        raise DerivationError("source is not an application artifact")
    return release, manifest, manifest_sha256


def _single_scalar_function(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    functions = manifest.get("functions")
    if not isinstance(functions, list) or len(functions) != 1:
        raise DerivationError("multihead derivation needs a single-function source release")
    function = functions[0]
    heads = function.get("heads")
    if not isinstance(heads, list) or len(heads) != 1 or heads[0].get("outputPath") != []:
        raise DerivationError("multihead derivation needs a scalar source function")
    head_ref = heads[0]["headRef"]
    head_resources = [
        r for r in manifest["resources"] if r["role"] == "head" and r["ref"] == head_ref
    ]
    if len(head_resources) != 1:
        raise DerivationError("source head resource is missing")
    return function, head_resources[0]


def _clone_identity(source_id: str, index: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"multihead:{source_id}:{index}".encode()).hexdigest()
    return f"{_FUNCTION_PREFIX}{digest}", f"head.{digest}.000"


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _dump(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, type=Path, help="source artifact root")
    parser.add_argument("--output", required=True, type=Path, help="derived artifact root")
    parser.add_argument("--functions", required=True, type=int, help="number of clones")
    parser.add_argument("--record", type=Path, help="where to write the derivation record")
    args = parser.parse_args(argv)
    try:
        record = derive_multihead_artifact(args.source, args.output, functions=args.functions)
    except DerivationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    record_bytes = _dump(record)
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_bytes(record_bytes)
    sys.stdout.write(record_bytes.decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
