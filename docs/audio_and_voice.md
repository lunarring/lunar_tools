# Audio and Voice

Install the audio extra before running the snippets on this page:

```bash
python -m pip install lunar_tools[audio]
```

Optional LLM features (OpenAI realtime, Gemini, Deepseek) require the `llm` extra as well.

## Bootstrapping the audio stack

Phase B introduced a service-first audio stack. Presentation scripts now compose
adapters via `AudioStackConfig`, `bootstrap_audio_stack`, and the
`AudioConversationController`.

```python
from lunar_tools.presentation.audio_stack import (
    AudioStackConfig,
    AudioConversationController,
    bootstrap_audio_stack,
)

config = AudioStackConfig(
    include_elevenlabs=False,   # only use OpenAI voices
    enable_playback=True,       # attach simpleaudio playback when available
    blocking_playback=True,
)

services, synthesiser = bootstrap_audio_stack(config)
controller = AudioConversationController(services, synthesiser=synthesiser)
```

`services` exposes the service layer (`RecorderService`, `SpeechToTextService`,
`TranscriptionService`, and `TextToSpeechService` instances). The controller
adds opinionated helpers that wire those services together for common flows.

## Recording and transcription

Capture a short snippet, wait for transcription, and log the result:

```python
transcript = controller.capture_transcript(max_time=3.0, minimum_duration=0.5)
if transcript:
    print("You said:", transcript)
```

`capture_transcript` proxies to `services.speech_to_text.record_and_translate`,
which accepts optional `output_filename` and `max_time` overrides. If you want
direct access to the service:

```python
from lunar_tools.presentation.audio_stack import bootstrap_audio_stack

services, _ = bootstrap_audio_stack()
with services.speech_to_text.recording(max_time=4):
    ...
text = services.speech_to_text.stop()
```

### Real-time (Deepgram) transcription

When the `audio` extra includes `deepgram-sdk`, the service bundle exposes a
`TranscriptionService` backed by `RealTimeTranscribe`. Start the stream when your
experience begins and poll `get_text()` for incremental captions.

```python
import time

transcriptions = services.realtime_transcription
if transcriptions:
    transcriptions.start()
    try:
        while True:
            print(transcriptions.transcript())
            time.sleep(2)
    finally:
        transcriptions.stop()
```

The adapter raises friendly `OptionalDependencyError` messages if `deepgram-sdk`
is not installed or the `DEEPGRAM_API_KEY` is missing.

## Speech synthesis and playback

`AudioConversationController.respond` generates speech and optionally plays it
through the configured playback adapter:

```python
controller.respond(
    "Welcome to Lunar Tools.",
    play=True,
    playback_kwargs={"pan": -0.1},
)
```

Prefer `controller.speak(text)` when you always want playback, or use the raw
service for fine-grained control:

```python
path = services.openai_tts.synthesize("Queued announcement")
# Later
services.openai_tts.set_playback(playback_adapter)
services.openai_tts.play("Immediate cue", pan=0.5)
```

## Realtime voice conversations

`RealTimeVoice` builds on the audio stack, wiring microphone capture, OpenAI
Realtime responses, and playback into a single session loop. Provide an
`AudioStackConfig` or a pre-built controller to reuse the adapters you already
bootstrapped.

```python
import time

from lunar_tools.presentation.audio_stack import AudioStackConfig
from lunar_tools.presentation.realtime_voice import RealTimeVoice

voice = RealTimeVoice(
    instructions="Chat like a friendly gallery guide.",
    audio_stack_config=AudioStackConfig(enable_playback=True),
)
voice.start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    voice.stop()
```

Hooks (`on_user_transcript`, `on_ai_transcript`, `on_ai_audio_complete`) let you
mirror transcripts or trigger side effects. See
[`examples/realtime_voice_example.py`](../examples/realtime_voice_example.py)
for an interactive CLI walkthrough.

## Migration guide (legacy adapters)

Legacy entry points remain available during the deprecation window via
`lunar_tools.audio` and the top-level package exports, but the service-oriented
API provides better testability and composability.

| Legacy name | New pattern |
| --- | --- |
| `lt.AudioRecorder` | `services.recorder_service` |
| `lt.Speech2Text` | `services.speech_to_text` (via controller or `bootstrap_audio_stack`) |
| `lt.RealTimeTranscribe` | `services.realtime_transcription` |
| `lt.Text2SpeechOpenAI` / `lt.Text2SpeechElevenlabs` | `services.openai_tts` / `services.elevenlabs_tts` |
| `lt.SoundPlayer` | Playback handled through `AudioConversationController` or `TextToSpeechService.set_playback` |

When migrating existing scripts:

1. Add `AudioStackConfig` and call `bootstrap_audio_stack`.
2. Replace direct adapter calls with service or controller helpers.
3. Keep the legacy imports temporarily if you need to compare behaviour; both
   stacks share the same underlying adapters.

The shims emit deprecation warnings in Phase E before complete removal.
