# Developing SemantScript

Install the Node and Python development environments from the repository root:

```sh
npm install
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

Install `.[dev,training]` when running model training or ONNX artifact-export
tests; the training extra includes PyTorch, Transformers, ONNX, and ONNX Runtime.

On Windows, use `.venv\Scripts\python.exe` for the two Python commands.
If the host Python lacks `venv`/`ensurepip`, install into the ignored local target
instead without changing the system environment:

```sh
python3 -m pip install --upgrade --target .python-packages '.[dev]'
```

Recreate `.python-packages` when package paths are removed or renamed so obsolete
modules cannot survive a target refresh.

Run every lint, build, and smoke-test gate with one command:

```sh
npm run check
```

The check script prefers `SEMANTSCRIPT_PYTHON`, an active virtual environment, and
then the repository `.venv`; it puts the working-tree Python sources first and also
adds `.python-packages` when present. It never
installs dependencies or accesses the network; environment setup remains an
explicit step.
