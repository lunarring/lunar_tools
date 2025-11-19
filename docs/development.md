# Development notes

Guidelines for hacking on Lunar Tools, running tests, and managing environment secrets.

## Local setup

```bash
git clone https://github.com/lunarring/lunar_tools.git
cd lunar_tools
python -m pip install -e ".[full]"
```

Extras map to optional feature sets:

```
audio, camera, comms, display, imaging, inputs, llm, video, full
```

Install only what you need—for example `python -m pip install -e ".[audio,display]"`.

## Testing & linting

Pytest covers the pure-Python layers. Run it from the project root:

```bash
python -m pytest
```

Phase E introduces `ruff` (lint), `mypy` (optional type checks), and `coverage` settings in `pyproject.toml`. Run the default lint pass with:

```bash
python -m pip install ruff
ruff check .
```

Use `tox` to exercise the canonical test environment (installs key extras):

```bash
python -m pip install tox
tox
```

Mark-specific suites are coming soon; for now skip hardware-dependent tests by setting environment variables or running inside CI.

## Optional dependency errors

Each optional feature raises an `OptionalDependencyError` that tells you which extra to install. Example:

```
>>> import lunar_tools as lt
>>> lt.Renderer()
OptionalDependencyError: Renderer requires optional dependencies that are not installed. Install the feature extras via `pip install lunar_tools[display]`.
```

Installing the suggested extra fixes the error without changing your code.

## API key management

You can keep secrets in shell files (`~/.zshrc`, `~/.bashrc`, etc.) or create the file `~/.lunar_tools_env_vars` with lines like `OPENAI_API_KEY=sk-...`. Read access is available via:

Use standard environment variable lookups in your code—for example:

```python
import os

openai_key = os.getenv("OPENAI_API_KEY")
```

Lunar Tools only checks `os.environ`; managing any local key files is up to your deployment or shell configuration.

## Documenting new features

- Update the main [`README`](../README.md) only with high-level messaging.
- Add detailed walkthroughs to an appropriate page inside `docs/`.
- Provide a runnable snippet or add a new file under [`examples/`](../examples) if the feature requires multiple moving parts.

## Useful tools

- `pipreqs . --force` regenerates `requirements.txt` from imports.
- `pytest -k "not slow"` runs only the fast tests once markers land.
- `python -m pip check` validates dependency resolution.
