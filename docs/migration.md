# Migration guide

Phase E focuses on hardening the modern service architecture without breaking
existing installations overnight. This guide explains how the compatibility
layer behaves today, how to migrate old imports, and what to expect over the
deprecation window.

## Compatibility layer overview

- Legacy symbols are still available from `lunar_tools/__init__.py`. Accessing
  one triggers a `DeprecationWarning` nudging you toward the modern module.
- Warnings point at the replacement import along with this file (`docs/migration.md`)
  and the config-first presentation docs (`docs/configuration.md`).
- The shims only load optional dependencies on demand, so the import cost
  matches the old behaviour.

Show the warnings locally by enabling default warnings:

```bash
python -Wd - <<'PY'
import lunar_tools as lt
lt.AudioRecorder  # shows the migration hint
PY
```

Expect the shims to disappear after the Phase E grace period (roughly one
release after this branch lands).

## Quick reference (before → after)

| Legacy usage | Modern replacement |
| --- | --- |
| `from lunar_tools import AudioRecorder` | `from lunar_tools.presentation.audio_stack import bootstrap_audio_stack` to get `services.recorder_service` |
| `from lunar_tools import RealTimeVoice` | `python -m lunar_tools.presentation.realtime_voice --config <file>` or import `RealTimeVoice` from `lunar_tools.presentation.realtime_voice` |
| `from lunar_tools import OSCSender` | `from lunar_tools import MessageBusConfig, create_message_bus` (service-layer bus chooses OSC/ZMQ adapters) |
| `from lunar_tools import Renderer` | `from lunar_tools.presentation.display_stack import bootstrap_display_stack` |
| `from lunar_tools import MovieSaver` | `from lunar_tools.presentation.movie_stack import bootstrap_movie_stack` |
| `from lunar_tools.image_gen import Dalle3ImageGenerator` | `from lunar_tools.services.vision import image_service` (provider registry handles adapters) |
| `from lunar_tools import OpenAIWrapper` | `from lunar_tools.services.llm.conversation_service import ConversationService` via presentation stack factories |

## Legacy modules retired

Phase E removed several alias modules that only emitted `DeprecationWarning`s:

- `lunar_tools.movie`
- `lunar_tools.display_window`
- `lunar_tools.control_input`
- `lunar_tools.realtime_voice`

Import directly from `lunar_tools.presentation.*` (or the relevant service module) as
shown above. Existing package-level shims (e.g., `from lunar_tools import MovieSaver`)
continue to work because `lunar_tools/__init__.py` already points to the modern
modules.

## Audio stack example

**Before**

```python
from lunar_tools import AudioRecorder, Speech2Text

recorder = AudioRecorder()
speech_to_text = Speech2Text()
```

**After**

```python
from lunar_tools.presentation.audio_stack import bootstrap_audio_stack, AudioStackConfig

services, controller = bootstrap_audio_stack(AudioStackConfig(enable_playback=True))
recorder = services.recorder_service
speech_to_text = services.speech_to_text
```

You still get `Speech2Text` via the compatibility shim when needed, but the
service bundle exposes the higher-level orchestration that future features rely
on.

## Communications example

**Before**

```python
from lunar_tools import OSCSender

osc = OSCSender(port=9001)
osc.send("/lighting/state", {"scene": "night"})
```

**After**

```python
from lunar_tools import MessageBusConfig, create_message_bus

bus = create_message_bus(MessageBusConfig(osc_port=9001)).message_bus
bus.send("osc", {"scene": "night"}, address="/lighting/state")
```

The unified `MessageBusService` manages OSC and ZMQ endpoints from config,
making it easier to switch transports without touching call sites.

## Display + movie stack example

**Before**

```python
from lunar_tools import Renderer, MovieSaver
```

**After**

```python
from lunar_tools.presentation.display_stack import bootstrap_display_stack
from lunar_tools.presentation.movie_stack import bootstrap_movie_stack
```

Both stacks enforce optional dependency guards and feed into the presentation
CLIs that now live under `lunar_tools.presentation`.

## Deprecation timeline

1. Phase E release: shims emit warnings (current stage).
2. Next minor release: warnings become `FutureWarning` and docs highlight last
   call for legacy imports.
3. Following release: shims and redundant modules (`audio.py`, `comms.py`, etc.)
   are removed once downstream projects confirm the migration.

Please file issues if any modern module is missing a feature you relied on—the
goal is to remove only redundant paths, not functionality.
