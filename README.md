# Lunar Tools

Lunar Tools is a modular toolbox for building interactive exhibitions that react to sound, voice, cameras, MIDI devices, LLMs, and custom render pipelines. Each capability lives behind a lightweight interface so you can stitch together realtime experiences without fighting heavy frameworks.

## Highlights
- Compose audio, vision, language, and hardware inputs into bespoke installations.
- Opt-in dependency extras keep the base install light while unlocking rich integrations.
- Realtime-friendly building blocks: render loops, telemetry, logging, and health reporting.
- Battle-tested scripts in `examples/` that you can copy, extend, or run as-is.

## Installation

### Base install
Make sure your environment uses Python 3.10 or newer.

```bash
python -m pip install git+https://github.com/lunarring/lunar_tools
```

### Feature extras
Install the extras you need to pull in optional dependencies. Combine extras with commas.

| Extra | Installs | Unlocks |
| --- | --- | --- |
| `audio` | OpenAI, ElevenLabs, Deepgram, sounddevice, pydub, simpleaudio | `AudioRecorder`, `Speech2Text`, `RealTimeVoice`, playback helpers |
| `llm` | OpenAI + Google Gemini clients | `OpenAIWrapper`, `Deepseek`, `Gemini` |
| `imaging` | DALL·E, FAL, Replicate, Pillow | `Dalle3ImageGenerator`, `FluxImageGenerator`, SDXL helpers |
| `camera` | OpenCV + Pillow | `WebCam` |
| `display` | OpenGL/SDL/pygame, CUDA runtime, torch | `Renderer`, `GridRenderer`, font rendering, torch image utils |
| `video` | ffmpeg/ffmpeg-python, moviepy, tqdm | `MovieSaver`, video composition helpers |
| `inputs` | pynput, pyusb, pygame, PyYAML | `KeyboardInput`, `MetaInput`, `MidiInput` |
| `comms` | python-osc, pyzmq, OpenCV | OSC/ZMQ endpoints (`OSCSender`, `ZMQPairEndpoint`) |
| `full` | Everything above | Matches the legacy all-in-one install |

```bash
python -m pip install lunar_tools[audio,display]
```

If a module is missing an optional dependency you will see an `OptionalDependencyError` telling you which extra to install.

### Editable / development install

```bash
git clone https://github.com/lunarring/lunar_tools.git
cd lunar_tools
python -m pip install -e ".[full]"
```

## Quickstart

### Record, play back, and monitor audio
Install the `audio` extra first.

```bash
python -m pip install lunar_tools[audio]
```

```python
import time
import lunar_tools as lt

recorder = lt.AudioRecorder()
player = lt.SoundPlayer()

recorder.start_recording("hello.mp3")
time.sleep(3)
recorder.stop_recording()

player.play_sound("hello.mp3")
```

### Render a numpy frame stream
Requires the `display` extra.

```python
import numpy as np
import lunar_tools as lt

renderer = lt.Renderer(width=640, height=480)
while True:
    frame = (np.random.rand(480, 640, 4) * 255).astype("uint8")
    renderer.render(frame)
```

Find more ready-to-run scripts in [`examples/`](examples).

## Documentation Map
- [`docs/README.md`](docs/README.md) – navigation hub.
- [`docs/inputs.md`](docs/inputs.md) – MIDI, keyboard, webcam, and controller workflows.
- [`docs/audio_and_voice.md`](docs/audio_and_voice.md) – speech, transcription, text-to-speech, realtime voice.
- [`docs/vision_and_display.md`](docs/vision_and_display.md) – rendering pipelines, movie helpers, image generation.
- [`docs/communication.md`](docs/communication.md) – OSC/ZMQ messaging and remote streaming.
- [`docs/logging_and_monitoring.md`](docs/logging_and_monitoring.md) – FPS tracking, logging, health reporting.
- [`docs/development.md`](docs/development.md) – testing, optional dependencies, API keys.
- [`docs/examples.md`](docs/examples.md) – overview of the Python example scripts.

## API Keys
You can store API keys in shell configuration (`~/.bash_profile`, `~/.zshrc`, etc.) or let Lunar Tools read/write them for you:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export REPLICATE_API_TOKEN="YOUR_KEY"
export ELEVEN_API_KEY="YOUR_KEY"
```

See [`docs/development.md`](docs/development.md) for the built-in key management helpers.

## Operating-system notes
- Linux users may need `sudo apt-get install libasound2-dev libportaudio2` for audio recording/playback.
- For ALSA-based MIDI controllers on Ubuntu, you might need `ln -s alsa-lib/libasound_module_conf_pulse.so libasound_module_conf_pulse.so` inside `/usr/lib/x86_64-linux-gnu/`.

## Contributing
Issues and pull requests are welcome. Start with [`docs/development.md`](docs/development.md) for testing commands and repository conventions.

## License
Distributed under the [MIT License](LICENSE).
