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

Pytest now runs under coverage and will fail when coverage drops below the
configured threshold (see `pyproject.toml`). Run it locally with:

```bash
coverage run -m pytest && coverage report
```

Static analysis and test automation are wired through `tox`:

```bash
tox -e lint      # ruff + mypy
tox -e py311     # pytest with coverage
tox -e extras-audio extras-llm extras-vision extras-presentation  # smoke optional extras
```

Each `extras-*` environment installs the matching optional dependency set and
runs the `tests/extras_smoke.py` script to ensure the modules import cleanly.
Mark-specific suites are coming soon; for now skip hardware-dependent tests by
setting environment variables or running inside CI.

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
