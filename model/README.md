# Model

The model package implements the shared encoder, statically selected application
adapters, typed per-function heads, training modules, calibration-compatible logits,
and ONNX artifact export.

Teacher prompting and TypeScript analysis belong to the trainer and compiler,
respectively.

## Encoder policy

The accepted Phase 1 encoder is
[`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base),
a roughly 149-million-parameter encoder. `SentenceEncoder` token-pools its final
hidden state with the attention mask and emits one float32 embedding per canonical
input. The tokenizer input is the UTF-8 text of `semantscript.canonical-input/v1`;
natural-language definitions and teacher prompts are not model features.

The code defaults to the verified immutable Hub revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`, and training records that value with
its result. A branch, tag, or `revision=None` is not an immutable model identity.
When intentionally selecting another encoder revision, use its full commit hash.
Keep
`EncoderConfig.trust_remote_code=False`; ModernBERT is supported directly by
Transformers, and a model that requires downloaded Python code is outside this
trust boundary. For an offline run, pre-cache that exact revision and also set
`local_files_only=True`.

```python
from semantscript_model.encoder import EncoderConfig

encoder = EncoderConfig(
    model_name="answerdotai/ModernBERT-base",
    revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
    local_files_only=True,
    trust_remote_code=False,
)
```

Hugging Face documents full commit hashes as immutable `revision` values in its
[download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

## Classification-head ABI

Heads return raw, finite logits; neither the encoder wrapper nor a head applies a
sigmoid, softmax, threshold, or calibration transform.

| Head kind | Stable support | Logit ABI | Probability interpretation |
| --- | --- | --- | --- |
| `binary-sigmoid` | `[false, true]` | float tensor `[batch, 1]` | `sigmoid(z)` is `P(true)`; `1 - sigmoid(z)` is `P(false)` |
| `categorical-softmax` | IR support order | float tensor `[batch, K]` | `softmax(logits)[i]` is the probability of support entry `i` |

The one-logit binary ABI is converted to the equivalent two-class logits `[0, z]`
only inside the loss calculation. The default `proper` objective is categorical
log loss plus `0.5 * spherical_loss`; ordinal categorical heads also add normalized
ranked probability score. `cross_entropy` is the explicit baseline and adds neither
spherical loss nor ranked probability score. Raw logits remain available for later
calibration and artifact export.

`semantscript_model.calibrate` fits a bounded positive scalar temperature by
held-out negative log likelihood without retaining an autograd graph. Its metrics
use stable support-order tie breaking, equal-width top-1 ECE bins, and the IR's
normalized multiclass Brier definition. For the one-logit binary ABI, calibration
uses the equivalent categorical logits `[0, z]`; the learned temperature remains
artifact metadata and is not baked into the classifier weights.

## ONNX component export

`export_onnx_components` writes the sentence encoder, an identity application
adapter, and the trained head as three independent opset-17 models. The graphs use
the runtime ABI names `input_ids`, `attention_mask`, `sentence_embedding`,
`function_embedding`, and `logits`. Export is accepted only after ONNX container
inspection, live ONNX Runtime metadata checks, and independent plus chained parity
against PyTorch on a caller-supplied one-row test batch. The exporter rejects
custom operator domains, local functions, training graphs, external tensor data,
existing destinations, and non-finite or incorrectly shaped outputs. Each graph
is size-checked before ONNX parsing or runtime loading (1 GiB by default), and
caller-supplied parity tolerances cannot exceed `1e-3`.

ONNX and ONNX Runtime are part of the `training` extra. They are imported lazily,
so importing the base model package remains supported without the extra.

## Installing training dependencies

PyTorch and Transformers are optional so compiler-only and teacher-generation
environments do not pay their installation cost. A CPU or otherwise
platform-resolved development installation is:

```bash
python3 -m pip install -e '.[dev,training]'
```

Importing `semantscript_model` without the `training` extra is supported. Constructing
a PyTorch-backed model then fails with an actionable missing-dependency error.

### AMD ROCm 7.2 on WSL

AMD's current WSL support matrix pairs ROCm 7.2 with PyTorch 2.9.1 and Triton 3.5.1.
The generic `torch==2.9.1` dependency does not by itself select AMD's validated ROCm
wheel, so install the official wheels explicitly. For Ubuntu 24.04 / CPython 3.12,
this project-local installation keeps the large stack out of the source tree's
normal Python environment:

```bash
python3 -m pip install --upgrade --target .python-packages \
  '.[dev,training]' \
  'numpy==1.26.4' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl'
```

Use a fresh `.python-packages` directory when changing PyTorch builds; overlaying a
ROCm wheel on files left by a generic build can produce a mixed installation. If a
virtual environment is active, install there instead and omit `--target` and the
`PYTHONPATH` line below.

AMD's WSL instructions require PyTorch to use the WSL-compatible HSA runtime rather
than the runtime bundled in a wheel. On this development machine, the installed
ROCDXG/HSA bridge is `/opt/rocm/core-10.0`; select it explicitly at runtime:

```bash
export PYTHONPATH="$PWD/.python-packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="/opt/rocm/core-10.0/lib/libhsa-runtime64.so.1"
```

`PYTHONNOUSERSITE=1` prevents unrelated packages under `~/.local` from being mixed
into the project-local target. This is required when those packages impose a NumPy
version incompatible with AMD's validated `numpy==1.26.4` stack; a virtual
environment provides the same isolation without this setting.

ROCm intentionally exposes devices through PyTorch's `torch.cuda` API. Verify both
the build and an actual device allocation before starting a training run:

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"); print(torch.ones(1, device="cuda"))'
```

The exact wheels and bridge requirements come from AMD's
[ROCm 7.2 WSL PyTorch installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html),
[WSL support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html),
and [ROCDXG bridge documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html).

## Offline tests and runs

Unit tests must not download a model or require a GPU. Encoder and classifier tests
inject tiny local fake modules, and optional-dependency tests skip cleanly when the
training extra is absent. A live ModernBERT/ROCm smoke test is an explicit integration
check, not part of the default offline suite.

For a disconnected real-model run, download the exact immutable revision while
online, retain the normal Hugging Face cache, then set `HF_HUB_OFFLINE=1` and use
`EncoderConfig(local_files_only=True, revision=<the-same-full-commit>)`. See the
[Transformers offline-mode guide](https://huggingface.co/docs/transformers/main/installation#offline-mode)
for cache and environment details.
