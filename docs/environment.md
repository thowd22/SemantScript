# Environment and `semantscript doctor`

Two toolchains have to be right before a build works: Node with the native
ONNX Runtime and tokenizer bindings, and Python with the trainer, PyTorch, a
usable device and a reachable teacher. `semantscript doctor` checks all of it
and prints one line per check with the fix; `init` runs it at the end and
`train` runs its Python and teacher checks before it starts, so a missing piece
fails in seconds instead of minutes into a run. The
[CLI reference](cli-reference.md#doctor) lists the flags and the report format.

```sh
npx semantscript doctor                 # every check, one teacher request
npx semantscript doctor --probe free    # nothing billed (Ollama: lists models)
npx semantscript doctor --runtime       # Node and the bindings only (serving machines)
npx semantscript doctor --json          # the report as JSON
```

## The checks

Each line reads `pass`, `fail`, `warn` or `skip`, the check id and what was
found; every line that did not pass has a `fix:` line under it. Exit 0 when
nothing failed, 1 otherwise.

| Check              | Passes when                                                                                              | Common failures and their fix                                                                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `node`             | Node is 22.13 or later.                                                                                  | Older Node: install 22.13+ (`.nvmrc` pins the tested release).                                                                                                     |
| `runtime-bindings` | `onnxruntime-node` and `tokenizers`, resolved from `@semantscript/core`, load for this platform.         | A `node_modules` copied from another OS or architecture, or an interrupted install: `npm install`, or `npm rebuild onnxruntime-node tokenizers`.                   |
| `python`           | The interpreter the CLI uses (`--python`, `SEMANTSCRIPT_PYTHON`, `python3`) starts and is 3.12 or later. | Not on the path, or too old: install 3.12 or point the CLI at one.                                                                                                 |
| `trainer`          | `semantscript_trainer` imports with the teacher clients `anthropic` and `openai`.                        | Outside the checkout with the package not installed: `pip install -e '.[training]'` into that interpreter.                                                         |
| `model`            | `semantscript_model` imports.                                                                            | As for `trainer`.                                                                                                                                                  |
| `torch`            | PyTorch and Transformers import; the line names the build (CUDA, ROCm or CPU-only).                      | The training extra is missing (install it), or a user-site package breaks the import (see `platform-env`).                                                         |
| `device`           | A CUDA or ROCm GPU is visible; the line gives its total and free memory.                                 | No GPU: a `warn`, training runs on the CPU (the RAM is shown). Apple MPS is reported but the trainer does not use it yet. `--device cuda` with no GPU is a `fail`. |
| `onnxruntime`      | ONNX Runtime and ONNX import (the export and its parity check need both).                                | The training extra is missing.                                                                                                                                     |
| `platform-env`     | The imports and the GPU work with the current environment.                                               | `PYTHONNOUSERSITE=1` when packages under `~/.local` break the imports; `HSA_ENABLE_DXG_DETECTION=1` when ROCm on WSL2 finds the GPU only with it.                  |
| `teacher-config`   | The teacher file `train` would use exists and is a valid `[teacher]` table.                              | No file and no `ANTHROPIC_API_KEY`, or an invalid table: see [teachers](teachers.md).                                                                              |
| `teacher-key`      | The key is in the environment (Ollama needs none).                                                       | `export ANTHROPIC_API_KEY=…`; with an OpenRouter `base_url` and only `OPENROUTER_API_KEY` set, `export ANTHROPIC_API_KEY="$OPENROUTER_API_KEY"`.                   |
| `teacher-probe`    | One minimal request succeeded; the line gives the latency and tokens.                                    | A refused key, a wrong model name, or an Ollama server that is down or lacks the model (`ollama serve`, `ollama pull <model>`).                                    |

### How the platform variables are detected

Doctor never suggests a variable from a list of known machines. It imports
PyTorch, Transformers and ONNX Runtime in a child interpreter, and when an
import fails it imports them again with `PYTHONNOUSERSITE=1`; the variable is
named only if that second run succeeds. On WSL2 (`/dev/dxg` present) with a
ROCm build of PyTorch that sees no GPU, it retries with
`HSA_ENABLE_DXG_DETECTION=1` the same way. When a variable is already set, the
full `doctor` also runs the imports without it and says whether it is still
needed; the `train` preflight (`--quick`) skips that confirmation and only
notes that the variable is set.

## Recorded runs

### Linux: WSL2 with ROCm (run 2026-09-25)

WSL2 (kernel 6.18 `microsoft-standard-WSL2`), Ubuntu 24.04 with the
system Python 3.12.3, the trainer from the checkout with its packages in
`.python-packages`, PyTorch 2.9.1 for ROCm 7.2, an AMD Radeon RX 9070 XT
(16 GB) and Node 22.22.0. Paths under the home directory are shortened to `~`
and scratch paths to `/tmp/scratch`.

The full `doctor` from `examples/express-app`, whose teacher is Sonnet 5
through OpenRouter, with `ANTHROPIC_API_KEY` set to the OpenRouter key for the
process. The probe cost about USD 0.0001 (16 input and 4 output tokens); the
run took 10 seconds, most of it the three PyTorch imports that establish which
variables are needed:

```text
semantscript doctor: ~/SemantScript/examples/express-app
  pass  node              Node 22.22.0 (linux-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for linux-x64
  pass  python            Python 3.12.3 at /usr/bin/python3
  pass  trainer           semantscript_trainer 0.0.0 from ~/SemantScript/trainer/src/semantscript_trainer
  pass  model             semantscript_model 0.0.0 from ~/SemantScript/model/src/semantscript_model
  pass  torch             torch 2.9.1+rocm7.2.0.git7e1940d4 (ROCm 7.2.26015-fc0010cf6a), transformers 5.17.0
  pass  device            trains on ROCm device AMD Radeon RX 9070 XT: 15.8 GiB total, 11.9 GiB free
  pass  onnxruntime       onnxruntime 1.30.0 and onnx 1.23.0 (export checks: AzureExecutionProvider, CPUExecutionProvider)
  pass  platform-env      PYTHONNOUSERSITE=1 is set and needed (without it: transformers: AttributeError: module 'numpy' has no attribute 'long'); keep it in the shell profile; WSL2 with ROCm: torch sees the GPU with or without HSA_ENABLE_DXG_DETECTION=1 (a WSL Ollama service may still need it)
  pass  teacher-config    ~/SemantScript/examples/express-app/.semantscript/teacher.toml (anthropic anthropic/claude-sonnet-5 via openrouter.ai, mode direct)
  pass  teacher-key       ANTHROPIC_API_KEY is set (sent to OpenRouter)
  pass  teacher-probe     one request to anthropic/claude-sonnet-5 answered in 2.3 s, 16 in / 4 out tokens
12 passed, 0 warnings, 0 failed, 0 skipped
```

The same machine with the user site enabled (no `PYTHONNOUSERSITE`) and a
local Ollama teacher. NumPy 2 under `~/.local` breaks Transformers; doctor
proves the fix by importing again with the variable set:

```text
semantscript doctor: ~/SemantScript/examples/express-app
  pass  node              Node 22.22.0 (linux-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for linux-x64
  pass  python            Python 3.12.3 at /usr/bin/python3
  pass  trainer           semantscript_trainer 0.0.0 from ~/SemantScript/trainer/src/semantscript_trainer
  pass  model             semantscript_model 0.0.0 from ~/SemantScript/model/src/semantscript_model
  fail  torch             torch 2.9.1+rocm7.2.0.git7e1940d4 (ROCm 7.2.26015-fc0010cf6a); transformers does not import: AttributeError: module 'numpy' has no attribute 'long'
        fix: export PYTHONNOUSERSITE=1 (add it to the shell profile so train and dev inherit it) (see platform-env)
  pass  device            trains on ROCm device AMD Radeon RX 9070 XT: 15.8 GiB total, 11.9 GiB free
  pass  onnxruntime       onnxruntime 1.30.0 and onnx 1.23.0 (export checks: AzureExecutionProvider, CPUExecutionProvider)
  fail  platform-env      PYTHONNOUSERSITE=1 needed: packages in the user site (/home/admin2/.local/lib/python3.12/site-packages) break the imports: transformers: AttributeError: module 'numpy' has no attribute 'long'
        fix: export PYTHONNOUSERSITE=1 (add it to the shell profile so train and dev inherit it)
  pass  teacher-config    /tmp/scratch/ollama.toml (ollama qwen2.5:1.5b-instruct-q4_K_M, mode auto)
  pass  teacher-key       the Ollama backend needs no key
  pass  teacher-probe     one request to qwen2.5:1.5b-instruct-q4_K_M answered in 0.1 s, 36 in / 2 out tokens
10 passed, 0 warnings, 2 failed, 0 skipped
```

`semantscript train` in `examples/express-app` without a key stops after the
five-second preflight, before any teacher request or model load:

```text
semantscript train: environment preflight (5.0 s)
  pass  python            Python 3.12.3 at /usr/bin/python3
  pass  trainer           semantscript_trainer 0.0.0 from ~/SemantScript/trainer/src/semantscript_trainer
  pass  model             semantscript_model 0.0.0 from ~/SemantScript/model/src/semantscript_model
  pass  torch             torch 2.9.1+rocm7.2.0.git7e1940d4 (ROCm 7.2.26015-fc0010cf6a), transformers 5.17.0
  pass  device            trains on ROCm device AMD Radeon RX 9070 XT: 15.8 GiB total, 12.1 GiB free
  pass  onnxruntime       onnxruntime 1.30.0 and onnx 1.23.0 (export checks: AzureExecutionProvider, CPUExecutionProvider)
  pass  platform-env      PYTHONNOUSERSITE=1 and HSA_ENABLE_DXG_DETECTION=1 set (semantscript doctor checks whether they are needed)
  pass  teacher-config    ~/SemantScript/examples/express-app/.semantscript/teacher.toml (anthropic anthropic/claude-sonnet-5 via openrouter.ai, mode direct)
  fail  teacher-key       ANTHROPIC_API_KEY is not set
        fix: export ANTHROPIC_API_KEY=<key> (an OpenRouter key when base_url is OpenRouter), or switch to the ollama backend (docs/teachers.md)
  skip  teacher-probe     not probed: the key is missing
8 passed, 0 warnings, 1 failed, 1 skipped
semantscript train: stopped before training; fix the failed checks above (semantscript doctor explains each one) or pass --no-preflight
```

And with an interpreter that does not exist, in a quarter of a second:

```text
semantscript train: environment preflight (0.0 s)
  fail  python            unable to run python3.99: spawn python3.99 ENOENT
        fix: install Python 3.12 or later, or point the CLI at one with --python <exe> or SEMANTSCRIPT_PYTHON
  skip  trainer           not checked: the interpreter does not start
  skip  model             not checked: the interpreter does not start
  skip  torch             not checked: the interpreter does not start
  skip  device            not checked: the interpreter does not start
  skip  onnxruntime       not checked: the interpreter does not start
  skip  platform-env      not checked: the interpreter does not start
  skip  teacher-config    not checked: the interpreter does not start
  skip  teacher-key       not checked: the interpreter does not start
  skip  teacher-probe     not checked: the interpreter does not start
0 passed, 0 warnings, 1 failed, 9 skipped
semantscript train: stopped before training; fix the failed checks above (semantscript doctor explains each one) or pass --no-preflight
```

### Linux: GitHub Actions (Ubuntu, CPU only; CI run 36164465531, 2026-09-25)

CI runs doctor on every push (see
[CONTRIBUTING](CONTRIBUTING.md#continuous-integration)). The fresh-install job
runs `npx semantscript doctor --runtime` in `examples/express-app` after a
first `npm install`:

```text
  pass  node              Node 22.22.0 (linux-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for linux-x64
2 passed, 0 warnings, 0 failed, 0 skipped
```

The Python job with the training extra runs
`doctor --python .venv/bin/python --no-teacher` against the CPU-only PyTorch
install, where `device` is a warning and nothing fails (repository paths
shortened to `~/SemantScript`):

```text
  pass  node              Node 22.22.0 (linux-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for linux-x64
  pass  python            Python 3.12.14 at ~/SemantScript/.venv/bin/python
  pass  trainer           semantscript_trainer 0.0.0 from ~/SemantScript/trainer/src/semantscript_trainer
  pass  model             semantscript_model 0.0.0 from ~/SemantScript/model/src/semantscript_model
  pass  torch             torch 2.9.1+cpu (CPU-only build), transformers 5.17.0
  warn  device            no CUDA or ROCm device: trains on the CPU (15.6 GiB RAM), expect a slow run
        fix: if this machine has an NVIDIA or AMD GPU, install the matching torch build (https://pytorch.org/get-started/locally/); otherwise lower --cases or --epochs
  pass  onnxruntime       onnxruntime 1.30.0 and onnx 1.23.0 (export checks: AzureExecutionProvider, CPUExecutionProvider)
  pass  platform-env      no extra environment variables needed
  skip  teacher-config    teacher checks not requested (--no-teacher)
  skip  teacher-key       teacher checks not requested (--no-teacher)
  skip  teacher-probe     teacher checks not requested (--no-teacher)
8 passed, 1 warnings, 0 failed, 3 skipped
```

### Windows (native) and macOS: not run

No doctor run has been recorded on native Windows or on macOS: this project's
only machines are the WSL2 host above and the Ubuntu CI runners. The checks
are written for both (the interpreter default is `python` on Windows, the fix
for a platform variable is printed as `set NAME=1` there, the RAM comes from
`GlobalMemoryStatusEx` on Windows and `sysconf` elsewhere, and on Apple
silicon the `device` line reports MPS as present but unused), but none of that
has been observed. A run from either platform belongs in this section.
