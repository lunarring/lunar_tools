# Introduction
Welcome to Lunar Tools, a comprehensive toolkit designed to fascilitate the programming of interactive exhibitions. Our suite of simple, modular tools is crafted to offer a seamless and hopefully bug-free experience for both exhibitors and visitors.

# Installation
Make sure you have python>=3.10.

## From PyPI (recommended)
```bash
pip install lunar-tools
```

For PyTorch-accelerated features (GPU rendering, torch tensor support):
```bash
pip install lunar-tools[torch]
```

## From GitHub (latest development version)
```bash
pip install git+https://github.com/lunarring/lunar_tools
```
## Ubuntu
On Ubuntu, you may have to install additional dependencies for sound playback/recording.

```bash
sudo apt-get install libasound2-dev libportaudio2
```

For running the midi controllers, you might have to create a symlink:
```bash
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s alsa-lib/libasound_module_conf_pulse.so libasound_module_conf_pulse.so
```

## API Keys

Our system includes a convenient automatic mode for reading and writing API keys. This feature enables you to dynamically set your API key as needed, and the file will be stored on your local computer.
However, if you prefer, you can specify your API keys in your shell configuration file (e.g. ~/.bash_profile or ~/.zshrc or ~/.bash_rc). In this case, paste the below lines with the API keys you want to add.
```bash
export OPENAI_API_KEY="XXX"
export REPLICATE_API_TOKEN="XXX"
export ELEVEN_API_KEY="XXX"
```

# Inputs
Runnable input snippets live in [examples/inputs](examples/inputs). Launch them from the repo root
to validate your hardware and copy/paste the relevant code into your own project.

## 🎙️ Audio Recorder
[examples/inputs/audio_recorder_example.py](examples/inputs/audio_recorder_example.py) exposes `lt.AudioRecorder` through two
CLI flags so you can verify your microphone pipeline without touching code.

```bash
python examples/inputs/audio_recorder_example.py --seconds 5 --output myvoice.mp3
```

## 📸 Webcam + Renderer
[examples/inputs/webcam_live_renderer.py](examples/inputs/webcam_live_renderer.py) pairs `lt.WebCam` with `lt.Renderer`
and displays a live preview window for whichever camera ID (or auto-probed
device) you pass in.

```bash
python examples/inputs/webcam_live_renderer.py --cam-id auto
```

## 🎚️ Meta Inputs
[examples/inputs/meta_input_inspector.py](examples/inputs/meta_input_inspector.py) uses `lt.MetaInput` to detect a MIDI
controller (or keyboard fallback) and continuously prints one slider + one
button so you can confirm your mappings on the spot.

```bash
python examples/inputs/meta_input_inspector.py
```

## 🎞️ Movie Reader
[examples/inputs/movie_reader_example.py](examples/inputs/movie_reader_example.py) wraps `lt.MovieReader`
with a CLI so you can inspect frame shapes, counts, and FPS before embedding
any mp4 into your pipeline.

```bash
python examples/inputs/movie_reader_example.py my_movie.mp4 --max-frames 10
```

# Outputs
Runnable output demos live in [examples/outputs](examples/outputs). Each script is a ready-to-run
showcase that you can copy into your own pipeline or execute as-is.

## 🔊 Play Sounds
[examples/outputs/sound_playback_generated_sine.py](examples/outputs/sound_playback_generated_sine.py) demonstrates `lt.SoundPlayer`
by first writing a generated 440 Hz sine to disk, then streaming a 660 Hz tone
directly from memory via `play_audiosegment`.

```bash
python examples/outputs/sound_playback_generated_sine.py
```

## 🖥️ Real-time Display
[examples/outputs/display_multi_backend_example.py](examples/outputs/display_multi_backend_example.py) spins up `lt.Renderer` and
cycles through NumPy, Pillow, and Torch backends (whichever are installed)
to render random RGBA frames in one looping window.

```bash
python examples/outputs/display_multi_backend_example.py
```

note you can speed-up opengl render calls by upto a factor of 3 by disabling
VSYNC on your system
On Ubuntu do: Run nvidia-settings 2. Screen 0 > OpenGl > Sync to VBLank ->
off

## ⏱️ Realtime Console Updates
[examples/outputs/realtime_console_updates_example.py](examples/outputs/realtime_console_updates_example.py) combines
`lt.FPSTracker`, `lt.LogPrint`, and `dynamic_print` to stream live progress
messages while measuring per-segment timings.

```bash
python examples/outputs/realtime_console_updates_example.py
```

## 🧾 Log Printer
[examples/outputs/logprint_example.py](examples/outputs/logprint_example.py) showcases `lt.LogPrint` formatting,
highlighting how to stream colored, timestamped console output.

```bash
python examples/outputs/logprint_example.py
```

## 🎬 Movie Saver
[examples/outputs/movie_saver_example.py](examples/outputs/movie_saver_example.py) creates a short mp4 using
random RGB frames so you can validate codec support and file permissions.

```bash
python examples/outputs/movie_saver_example.py --output my_movie.mp4 --frames 10 --fps 24
```

# 📡 Communication

## 📍 Local IP Detection
`lunar_tools.comms.get_local_ip` inspects network interfaces to determine the best IP to share with peers. Run the example below to print the detected address or see a friendly warning if one cannot be determined (for example, on air-gapped machines).

```bash
python examples/comms/get_local_ip_example.py
```

## 🌐 WebRTC Data Channels
Low-latency data channel built on WebRTC for streaming numpy arrays, JSON blobs, PNG previews, and log text.

Sender (hosts an embedded signaling server and streams mixed payloads):

```bash
python examples/comms/webrtc_data_sender.py --session demo
```

Receiver (auto-discovers the sender session via the cached signaling endpoint):

```bash
python examples/comms/webrtc_data_receiver.py --session demo
```

- `--sender-ip` defaults to the detected local address (via `lunar_tools.comms.utils.get_local_ip`).
- When the sender hosts the embedded signaling server it stores the endpoint details per session in `~/.lunar_tools/webrtc_sessions.json`. Receivers can omit `--sender-ip` to reuse the most recent entry for the requested session, which keeps the bootstrap process simple.
- If you prefer using your own signaling server, start it separately (or pass `--no-server` in the sender example) and point both peers to the same `http://<sender-ip>:<port>` URL.

## 🎛️ OSC
High-level OSC helper built on python-osc. The receiver example spawns the live grid visualizer, and the sender emits demo sine/triangle waves.

Receiver:

```bash
python examples/comms/osc_receiver.py --ip 0.0.0.0 --port 8003
```

Sender:

```bash
python examples/comms/osc_sender.py --ip 127.0.0.1 --port 8003 --channels /env1 /env2 /env3
```

## 🔁 ZMQ Pair Endpoint
One-to-one ZeroMQ stream that carries JSON blobs, compressed images, and raw PCM audio. Start the receiver first on the same machine (or pass `--ip 0.0.0.0` if you want to accept remote peers), then launch the sender.

Receiver (binds locally):

```bash
python examples/comms/zmq_receiver.py --port 5556
```

Sender (connects to the receiver):

```bash
python examples/comms/zmq_sender.py --ip 127.0.0.1 --port 5556
```

`ZMQPairEndpoint` uses ZeroMQ's `PAIR` pattern, which is strictly one-to-one: exactly one sender and one receiver must be connected, and neither side can reconnect while the other is running. If you need fan-out/fan-in or resilient reconnection, prefer `REQ/REP`, `PUB/SUB`, or `ROUTER/DEALER` and stitch together the behavior you need on top of the raw `zmq` library.



# 🔊 Voice & Speech
Voice-focused demos live in [examples/voice](examples/voice). Each script below can be
run directly from the repo root and pairs with the API snippets that follow.

## 🗣️ RealTimeVoice (OpenAI)
[examples/voice/realtime_voice_example.py](examples/voice/realtime_voice_example.py) is an interactive CLI
that lets you start/pause/mute a RealTimeVoice session, inject messages, and update
instructions on the fly.

```bash
python examples/voice/realtime_voice_example.py
```

## 🎤 Deepgram Realtime Transcribe
[examples/voice/deepgram_realtime_transcribe_example.py](examples/voice/deepgram_realtime_transcribe_example.py) uses
`lt.RealTimeTranscribe` (Deepgram SDK) to stream microphone audio and print live transcripts.
Set `DEEPGRAM_API_KEY` before running.

```bash
python examples/voice/deepgram_realtime_transcribe_example.py
```

## 📝 Speech-to-Text (OpenAI)
[examples/voice/openai_speech_to_text_example.py](examples/voice/openai_speech_to_text_example.py)
records a short microphone clip and prints the transcript, with an optional flag
to save the text to disk.

```bash
python examples/voice/openai_speech_to_text_example.py --seconds 5 --output transcript.txt
```

## 🔈 Text-to-Speech (OpenAI)
[examples/voice/openai_text_to_speech_example.py](examples/voice/openai_text_to_speech_example.py)
converts text to speech, saves it to an mp3, and can optionally stream the audio
immediately with `--play-inline`.

```bash
python examples/voice/openai_text_to_speech_example.py --text "Testing 1 2 3" --voice nova --play-inline
```

## 🎶 Text-to-Speech (ElevenLabs)
[examples/voice/elevenlabs_text_to_speech_example.py](examples/voice/elevenlabs_text_to_speech_example.py)
targets the ElevenLabs API with inline playback plus flags for stability, similarity,
style, and speaker boost.

```bash
python examples/voice/elevenlabs_text_to_speech_example.py --text "Hi from ElevenLabs" --voice-id EXAVITQu4vr4xnSDxMaL --play-inline
```

# 🎨 Image Generation

## 🖼️ DALL·E 3
[examples/ai/dalle3_generate_example.py](examples/ai/dalle3_generate_example.py) calls
`lt.Dalle3ImageGenerator`, saves the resulting PNG, and prints the revised prompt.

```bash
python examples/ai/dalle3_generate_example.py --prompt "A red house with snow and a chimney"
```

## 🚀 SDXL Turbo
[examples/ai/sdxl_turbo_example.py](examples/ai/sdxl_turbo_example.py) uses the Replicate-powered
`lt.SDXL_TURBO` helper and stores the PNG plus source URL for reference.

```bash
python examples/ai/sdxl_turbo_example.py --prompt "An astronaut riding a rainbow unicorn" --width 768 --height 512
```

## 🍌 Nano Banana Edit (Gradio)
[examples/ai/nano_banana_edit_gradio.py](examples/ai/nano_banana_edit_gradio.py) launches a Gradio UI
for interactive Flux/Nano Banana edits—drop in prompts, tweak sliders, and preview changes.

```bash
python examples/ai/nano_banana_edit_gradio.py
```

# Health status reporting via telegram
Obtain a bot here: https://docs.tracardi.com/qa/how_can_i_get_telegram_bot/
Next you will need to update your bashrc or bash_profile with the telegram bot env variables.


```bash
export TELEGRAM_BOT_TOKEN='XXX'
export TELEGRAM_CHAT_ID='XXX'
```

See [examples/health/telegram_health_reporter_example.py](examples/health/telegram_health_reporter_example.py)
for a runnable heartbeat + alert demo (requires the env vars above):

```bash
python examples/health/telegram_health_reporter_example.py --name "My Exhibit" --interval 2 --count 5
```

# Devinfos
## Testing
pip install pytest

make sure you are in base folder
```python
python -m pytest lunar_tools/tests/
```

## Get requirements
```python
pipreqs . --force
```
