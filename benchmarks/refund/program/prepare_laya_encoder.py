"""Extract the Laya typed-decisions agent's ModernBERT-large encoder as a Hugging Face directory.

The pinned checkpoint stores the whole RL agent (encoder, scorer, action head,
temperatures) in one ``model.safetensors`` with ``encoder.`` prefixes and fp16
tensors. The encoder sweep (TASK-5.14) needs those weights as an ordinary
``AutoModel`` directory, so this script writes ``config.json`` (the checkpoint's
own encoder config), ``model.safetensors`` (the ``encoder.*`` tensors, prefix
stripped, cast to float32) and the checkpoint's tokenizer files next to it, and
records the source revision and digests in ``extraction.json``.

Usage::

    python3 -m benchmarks.refund.program.prepare_laya_encoder \\
        --checkpoint ~/.cache/huggingface/hub/models--convaiinnovations--laya-typed-decisions/snapshots/<rev> \\
        --output benchmarks/refund/data/encoders/laya-typed-decisions-encoder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.refund.live.laya_worker import LAYA_CHECKPOINT, LAYA_CHECKPOINT_REVISION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    checkpoint = arguments.checkpoint.resolve()
    if checkpoint.name != LAYA_CHECKPOINT_REVISION:
        raise SystemExit(f"checkpoint path must be the pinned snapshot {LAYA_CHECKPOINT_REVISION}")
    output: Path = arguments.output
    if output.exists():
        raise SystemExit(f"{output} already exists")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = checkpoint / "model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(source), "pt") as handle:
        for key in handle.keys():
            if key.startswith("encoder."):
                tensors[key[len("encoder.") :]] = (
                    handle.get_tensor(key).to(torch.float32).contiguous()
                )
    if not tensors:
        raise SystemExit("no encoder.* tensors in the checkpoint")
    output.mkdir(parents=True)
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    config = json.loads((checkpoint / "encoder" / "config.json").read_text(encoding="utf-8"))
    config["architectures"] = ["ModernBertModel"]
    config["dtype"] = "float32"
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        candidate = checkpoint / "tokenizer" / name
        if candidate.is_file():
            shutil.copyfile(candidate, output / name)
    record = {
        "kind": "semantscript.laya-encoder-extraction",
        "sourceCheckpoint": LAYA_CHECKPOINT,
        "sourceRevision": LAYA_CHECKPOINT_REVISION,
        "sourceModelSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tensors": len(tensors),
        "parameters": sum(t.numel() for t in tensors.values()),
        "encoderWeightsSha256": hashlib.sha256(
            (output / "model.safetensors").read_bytes()
        ).hexdigest(),
        "note": "encoder.* tensors of the RL agent, prefix stripped, fp16 cast to float32; scorer, action head and temperatures dropped",
    }
    (output / "extraction.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
