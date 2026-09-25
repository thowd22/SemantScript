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

| Check              | Passes when                                                                                                                           | Common failures and their fix                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `node`             | Node is 22.13 or later.                                                                                                               | Older Node: install 22.13+ (`.nvmrc` pins the tested release).                                                                                                     |
| `runtime-bindings` | `onnxruntime-node` and `tokenizers`, resolved from `@semantscript/core`, load for this platform.                                      | A `node_modules` copied from another OS or architecture, or an interrupted install: `npm install`, or `npm rebuild onnxruntime-node tokenizers`.                   |
| `python`           | The interpreter the CLI uses (`--python`, `SEMANTSCRIPT_PYTHON`, else `python3`, or `python` on Windows) starts and is 3.12 or later. | Not on the path, or too old (checked even when the trainer's 3.12 syntax stops it importing): install 3.12 or point the CLI at one.                                |
| `trainer`          | `semantscript_trainer` imports with the teacher clients `anthropic` and `openai`.                                                     | Outside the checkout with the package not installed: `pip install -e '.[training]'` into that interpreter.                                                         |
| `model`            | `semantscript_model` imports.                                                                                                         | As for `trainer`.                                                                                                                                                  |
| `torch`            | PyTorch and Transformers import; the line names the build (CUDA, ROCm or CPU-only).                                                   | The training extra is missing (install it), or a user-site package breaks the import (see `platform-env`).                                                         |
| `device`           | A CUDA or ROCm GPU is visible; the line gives its total and free memory.                                                              | No GPU: a `warn`, training runs on the CPU (the RAM is shown). Apple MPS is reported but the trainer does not use it yet. `--device cuda` with no GPU is a `fail`. |
| `onnxruntime`      | ONNX Runtime and ONNX import (the export and its parity check need both).                                                             | The training extra is missing.                                                                                                                                     |
| `platform-env`     | The imports and the GPU work with the current environment.                                                                            | `PYTHONNOUSERSITE=1` when packages under `~/.local` break the imports; `HSA_ENABLE_DXG_DETECTION=1` when ROCm on WSL2 finds the GPU only with it.                  |
| `teacher-config`   | The teacher file `train` would use exists and is a valid `[teacher]` table.                                                           | No file and no `ANTHROPIC_API_KEY`, or an invalid table: see [teachers](teachers.md).                                                                              |
| `teacher-key`      | The key is in the environment (Ollama needs none); `ANTHROPIC_AUTH_TOKEN` counts too.                                                 | `export ANTHROPIC_API_KEY=…`; with an OpenRouter `base_url` and only `OPENROUTER_API_KEY` set, `export ANTHROPIC_API_KEY="$OPENROUTER_API_KEY"`.                   |
| `teacher-probe`    | One minimal request succeeded; the line gives the latency and tokens.                                                                 | A refused key, a wrong model name, or an Ollama server that is down or lacks the model (`ollama serve`, `ollama pull <model>`; a name without a tag is `:latest`). |

When the interpreter was left at its default and a `.venv` exists in the
working directory or above it, every failed `python`, `trainer`, `model`,
`torch` or `onnxruntime` line also names that venv's interpreter: the CLI
uses a venv only when it is activated or passed with `--python` or
`SEMANTSCRIPT_PYTHON`.

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
local Ollama teacher (re-run 2026-09-25 after the review fixes). NumPy 2 under
`~/.local` breaks Transformers; doctor proves the fix by importing again with
the variable set:

```text
semantscript doctor: ~/SemantScript/examples/express-app
  pass  node              Node 22.22.0 (linux-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for linux-x64
  pass  python            Python 3.12.3 at /usr/bin/python3
  pass  trainer           semantscript_trainer 0.0.0 from ~/SemantScript/trainer/src/semantscript_trainer
  pass  model             semantscript_model 0.0.0 from ~/SemantScript/model/src/semantscript_model
  fail  torch             torch 2.9.1+rocm7.2.0.git7e1940d4 (ROCm 7.2.26015-fc0010cf6a); transformers does not import: AttributeError: module 'numpy' has no attribute 'long'
        fix: export PYTHONNOUSERSITE=1 (see platform-env)
  pass  device            trains on ROCm device AMD Radeon RX 9070 XT: 15.8 GiB total, 13.3 GiB free
  pass  onnxruntime       onnxruntime 1.30.0 and onnx 1.23.0 (export checks: AzureExecutionProvider, CPUExecutionProvider)
  fail  platform-env      PYTHONNOUSERSITE=1 needed: packages in the user site (~/.local/lib/python3.12/site-packages) break the imports: transformers: AttributeError: module 'numpy' has no attribute 'long'
        fix: export PYTHONNOUSERSITE=1, and add it to the shell profile so train and dev inherit it
  pass  teacher-config    /tmp/scratch/ollama.toml (ollama qwen2.5:1.5b-instruct-q4_K_M, mode auto)
  pass  teacher-key       the Ollama backend needs no key
  pass  teacher-probe     one request to qwen2.5:1.5b-instruct-q4_K_M answered in 2.1 s, 36 in / 2 out tokens
10 passed, 0 warnings, 2 failed, 0 skipped
```

A teacher file naming an Ollama model without a tag (`model = "glm-4.7-flash"`)
passes the free probe, which the `train` preflight uses, when the server lists
`glm-4.7-flash:latest`, as Ollama itself resolves the name:

```text
  pass  teacher-config    /tmp/scratch/glm.toml (ollama glm-4.7-flash, mode auto)
  pass  teacher-key       the Ollama backend needs no key
  pass  teacher-probe     the Ollama server answered in 0.0 s and has glm-4.7-flash (no request sent)
```

An interpreter older than 3.12 cannot import the trainer at all (a
`SyntaxError` on its `type` statements), so the version is checked from
Node. This machine has no older Python; the run below uses a shim that
answers the version query as 3.11.9 and fails the import the way 3.11 does:

```text
  fail  python            Python 3.11.9 (./py311) is older than 3.12, which the trainer needs
        fix: install Python 3.12 or later, or point the CLI at one with --python <exe> or SEMANTSCRIPT_PYTHON
  skip  trainer           not checked: the interpreter cannot run the trainer
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

### Linux: GitHub Actions (Ubuntu, CPU only; CI run 36166910601, 2026-09-25)

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
8 passed, 1 warning, 0 failed, 3 skipped
```

### Windows (native): GitHub Actions `windows-latest` (CI run 36166910601, 2026-09-25)

The `doctor-platforms` CI job runs doctor on every push on a native Windows
runner (Windows Server, x64, no GPU) and on a macOS runner, after `npm ci`,
`npm run build` and a `.venv` with the CPU PyTorch build and
`pip install -e '.[training]'`. The job runs doctor three times (checkout path
shortened to `D:\SemantScript`).

With nothing activated the CLI's default interpreter is `python`, the runner's
bare Python 3.12, which lacks the trainer's packages; the fix names the venv
that has them (this step's exit status 1 is expected):

```text
semantscript doctor: D:\SemantScript
  pass  node              Node 22.22.0 (win32-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for win32-x64
  pass  python            Python 3.12.10 at C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe
  fail  trainer           semantscript_trainer 0.0.0 from D:\SemantScript\trainer\src\semantscript_trainer; missing anthropic, openai
        fix: pip install -e . from the SemantScript checkout (installs the teacher clients anthropic and openai); or if the packages are in .venv\Scripts\python.exe rather than python (the default interpreter), pass --python .venv\Scripts\python.exe or set SEMANTSCRIPT_PYTHON=.venv\Scripts\python.exe
  pass  model             semantscript_model 0.0.0 from D:\SemantScript\model\src\semantscript_model
  fail  torch             torch does not import: ModuleNotFoundError: No module named 'torch'
        fix: install the training extra into this interpreter: pip install -e '.[training]' from the SemantScript checkout (for a GPU, install the CUDA or ROCm torch build from https://pytorch.org/get-started/locally/ first); or if the packages are in .venv\Scripts\python.exe rather than python (the default interpreter), pass --python .venv\Scripts\python.exe or set SEMANTSCRIPT_PYTHON=.venv\Scripts\python.exe
  skip  device            torch does not import, so no device was checked
  fail  onnxruntime       onnxruntime does not import: ModuleNotFoundError: No module named 'onnxruntime'
        fix: install the training extra into this interpreter: pip install -e '.[training]' from the SemantScript checkout (for a GPU, install the CUDA or ROCm torch build from https://pytorch.org/get-started/locally/ first); or if the packages are in .venv\Scripts\python.exe rather than python (the default interpreter), pass --python .venv\Scripts\python.exe or set SEMANTSCRIPT_PYTHON=.venv\Scripts\python.exe
  pass  platform-env      no extra environment variables needed
  skip  teacher-config    teacher checks not requested (--no-teacher)
  skip  teacher-key       teacher checks not requested (--no-teacher)
  skip  teacher-probe     teacher checks not requested (--no-teacher)
5 passed, 0 warnings, 3 failed, 4 skipped
```

With the venv activated, the default `python` is the venv's; the RAM on the
`device` line comes from `GlobalMemoryStatusEx`:

```text
semantscript doctor: D:\SemantScript
  pass  node              Node 22.22.0 (win32-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for win32-x64
  pass  python            Python 3.12.10 at D:\SemantScript\.venv\Scripts\python.exe
  pass  trainer           semantscript_trainer 0.0.0 from D:\SemantScript\trainer\src\semantscript_trainer
  pass  model             semantscript_model 0.0.0 from D:\SemantScript\model\src\semantscript_model
  pass  torch             torch 2.9.1+cpu (CPU-only build), transformers 5.17.0
  warn  device            no CUDA or ROCm device: trains on the CPU (16.0 GiB RAM), expect a slow run
        fix: if this machine has an NVIDIA or AMD GPU, install the matching torch build (https://pytorch.org/get-started/locally/); otherwise lower --cases or --epochs
  pass  onnxruntime       onnxruntime 1.30.0 and onnx 1.23.0 (export checks: AzureExecutionProvider, CPUExecutionProvider)
  pass  platform-env      no extra environment variables needed
  skip  teacher-config    teacher checks not requested (--no-teacher)
  skip  teacher-key       teacher checks not requested (--no-teacher)
  skip  teacher-probe     teacher checks not requested (--no-teacher)
8 passed, 1 warning, 0 failed, 3 skipped
```

And `doctor --runtime`, the check a serving machine needs:

```text
semantscript doctor: D:\SemantScript
  pass  node              Node 22.22.0 (win32-x64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for win32-x64
2 passed, 0 warnings, 0 failed, 0 skipped
```

### macOS: GitHub Actions `macos-latest` (Apple silicon; CI run 36166910601, 2026-09-25)

The same job on the macOS 26 arm64 runner (checkout path shortened to
`~/SemantScript`). The default interpreter is `python3`:

```text
semantscript doctor: ~/SemantScript
  pass  node              Node 22.22.0 (darwin-arm64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for darwin-arm64
  pass  python            Python 3.12.10 at /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
  fail  trainer           semantscript_trainer 0.0.0 from ~/SemantScript/trainer/src/semantscript_trainer; missing anthropic, openai
        fix: pip install -e . from the SemantScript checkout (installs the teacher clients anthropic and openai); or if the packages are in .venv/bin/python rather than python3 (the default interpreter), pass --python .venv/bin/python or set SEMANTSCRIPT_PYTHON=.venv/bin/python
  pass  model             semantscript_model 0.0.0 from ~/SemantScript/model/src/semantscript_model
  fail  torch             torch does not import: ModuleNotFoundError: No module named 'torch'
        fix: install the training extra into this interpreter: pip install -e '.[training]' from the SemantScript checkout (for a GPU, install the CUDA or ROCm torch build from https://pytorch.org/get-started/locally/ first); or if the packages are in .venv/bin/python rather than python3 (the default interpreter), pass --python .venv/bin/python or set SEMANTSCRIPT_PYTHON=.venv/bin/python
  skip  device            torch does not import, so no device was checked
  fail  onnxruntime       onnxruntime does not import: ModuleNotFoundError: No module named 'onnxruntime'
        fix: install the training extra into this interpreter: pip install -e '.[training]' from the SemantScript checkout (for a GPU, install the CUDA or ROCm torch build from https://pytorch.org/get-started/locally/ first); or if the packages are in .venv/bin/python rather than python3 (the default interpreter), pass --python .venv/bin/python or set SEMANTSCRIPT_PYTHON=.venv/bin/python
  pass  platform-env      no extra environment variables needed
  skip  teacher-config    teacher checks not requested (--no-teacher)
  skip  teacher-key       teacher checks not requested (--no-teacher)
  skip  teacher-probe     teacher checks not requested (--no-teacher)
5 passed, 0 warnings, 3 failed, 4 skipped
```

With the venv activated. MPS is available on this runner, and the `device`
line says the trainer does not use it yet:

```text
semantscript doctor: ~/SemantScript
  pass  node              Node 22.22.0 (darwin-arm64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for darwin-arm64
  pass  python            Python 3.12.10 at ~/SemantScript/.venv/bin/python3
  pass  trainer           semantscript_trainer 0.0.0 from ~/SemantScript/trainer/src/semantscript_trainer
  pass  model             semantscript_model 0.0.0 from ~/SemantScript/model/src/semantscript_model
  pass  torch             torch 2.9.1 (CPU-only build), transformers 5.17.0
  warn  device            Apple MPS is available but the trainer does not use it yet: trains on the CPU (7.0 GiB RAM)
        fix: expect slower runs; lower --cases or --epochs for a first try
  pass  onnxruntime       onnxruntime 1.30.0 and onnx 1.23.0 (export checks: CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider)
  pass  platform-env      no extra environment variables needed
  skip  teacher-config    teacher checks not requested (--no-teacher)
  skip  teacher-key       teacher checks not requested (--no-teacher)
  skip  teacher-probe     teacher checks not requested (--no-teacher)
8 passed, 1 warning, 0 failed, 3 skipped
```

```text
semantscript doctor: ~/SemantScript
  pass  node              Node 22.22.0 (darwin-arm64)
  pass  runtime-bindings  onnxruntime-node 1.30.0 and tokenizers 0.23.2 loaded for darwin-arm64
2 passed, 0 warnings, 0 failed, 0 skipped
```

### Not yet observed

No run so far has needed a platform variable on Windows, so the `set NAME=1`
form of the `platform-env` fix has not been printed on a real machine (it is
covered by the unit tests only). No run has used a GPU on Windows or macOS:
the runners have none that PyTorch uses for training.
