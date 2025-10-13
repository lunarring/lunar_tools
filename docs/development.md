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

## Testing

Pytest covers the pure-Python layers. Run it from the project root:

```bash
python -m pytest
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

You can keep secrets in shell files (`~/.zshrc`, `~/.bashrc`, etc.) or let Lunar Tools manage a simple key-value store under `~/.lunar_tools_env_vars`.

```python
import lunar_tools as lt

lt.save_api_key_to_lunar_config("OPENAI_API_KEY", "sk-...")
print(lt.read_api_key_from_lunar_config("OPENAI_API_KEY"))
lt.delete_api_key_from_lunar_config("OPENAI_API_KEY")
```

Keys set via the environment always take precedence when the toolkit looks them up.

## Documenting new features

- Update the main [`README`](../README.md) only with high-level messaging.
- Add detailed walkthroughs to an appropriate page inside `docs/`.
- Provide a runnable snippet or add a new file under [`examples/`](../examples) if the feature requires multiple moving parts.

## Useful tools

- `pipreqs . --force` regenerates `requirements.txt` from imports.
- `pytest -k "not slow"` runs only the fast tests once markers land.
- `python -m pip check` validates dependency resolution.
