# Introduction
Welcome to Lunar Tools, a comprehensive toolkit designed to fascilitate the programming of interactive exhibitions. Our suite of simple, modular tools is crafted to offer a seamless and hopefully bug-free experience for both exhibitors and visitors.

# Installation
Make sure you have python>=3.10.
```bash
python -m pip install git+https://github.com/lunarring/lunar_tools
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

## Audio Recorder
[examples/inputs/audio_recorder_example.py](examples/inputs/audio_recorder_example.py) exposes `lt.AudioRecorder` through two
CLI flags so you can verify your microphone pipeline without touching code.

```bash
python examples/inputs/audio_recorder_example.py --seconds 5 --output myvoice.mp3
```

## Webcam + Renderer
[examples/inputs/webcam_live_renderer.py](examples/inputs/webcam_live_renderer.py) pairs `lt.WebCam` with `lt.Renderer`
and displays a live preview window for whichever camera ID (or auto-probed
device) you pass in.

```bash
python examples/inputs/webcam_live_renderer.py --cam-id auto
```

## Meta Inputs
[examples/inputs/meta_input_inspector.py](examples/inputs/meta_input_inspector.py) uses `lt.MetaInput` to detect a MIDI
controller (or keyboard fallback) and continuously prints one slider + one
button so you can confirm your mappings on the spot.

```bash
python examples/inputs/meta_input_inspector.py
```

## Movie Reader
[examples/inputs/movie_reader_example.py](examples/inputs/movie_reader_example.py) wraps `lt.MovieReader`
with a CLI so you can inspect frame shapes, counts, and FPS before embedding
any mp4 into your pipeline.

```bash
python examples/inputs/movie_reader_example.py my_movie.mp4 --max-frames 10
```

# Outputs
Runnable output demos live in [examples/outputs](examples/outputs). Each script is a ready-to-run
showcase that you can copy into your own pipeline or execute as-is.

## Play Sounds
[examples/outputs/sound_playback_generated_sine.py](examples/outputs/sound_playback_generated_sine.py) demonstrates `lt.SoundPlayer`
by first writing a generated 440 Hz sine to disk, then streaming a 660 Hz tone
directly from memory via `play_audiosegment`.

```bash
python examples/outputs/sound_playback_generated_sine.py
```

## Real-time Display
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

## FPS Tracker
[examples/outputs/fps_tracker_example.py](examples/outputs/fps_tracker_example.py) uses `lt.FPSTracker` to emit color-coded
segment timings so you can profile render + process loops in real time.

```bash
python examples/outputs/fps_tracker_example.py
```

## Log Printer
[examples/outputs/logprint_example.py](examples/outputs/logprint_example.py) showcases `lt.LogPrint` formatting,
highlighting how to stream colored, timestamped console output.

```bash
python examples/outputs/logprint_example.py
```

## Movie Saver
[examples/outputs/movie_saver_example.py](examples/outputs/movie_saver_example.py) creates a short mp4 using
random RGB frames so you can validate codec support and file permissions.

```bash
python examples/outputs/movie_saver_example.py --output my_movie.mp4 --frames 10 --fps 24
```



# Language

## RealtimeVoice OpenAI

### Simple Example
```python
instructions = "Respond briefly and with a sarcastic attitude."
rtv = RealTimeVoice(instructions=instructions)
rtv.start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    rtv.stop()
    print("\nExiting.")
```

### Advanced Example
```python
instructions = "Respond briefly and with a sarcastic attitude."
temperature = 0.6
voice = "echo"
mute_mic_while_ai_speaking = True # that's the default already, just FYI

# Optional: callback for when the whisper transcription is done
async def on_user_transcript(transcript: str):
    print(f"(on_user_transcript) User said: {transcript}")

# Optional: callback for when the transcript of the voice response is there
async def on_ai_transcript(transcript: str):
    print(f"(on_ai_transcript) AI replied: {transcript}")

# Optional: callback for when the audio has been completely played
async def on_audio_complete():
    print("(on_audio_complete) AI audio has been completely played.")

rtv = RealTimeVoice(
    instructions=instructions,
    on_user_transcript=on_user_transcript,
    on_ai_transcript=on_ai_transcript,
    on_audio_complete=on_audio_complete,
    model="gpt-4o-mini-realtime-preview-2024-12-17",
    temperature=temperature,
    voice=voice,
    mute_mic_while_ai_speaking=mute_mic_while_ai_speaking,
    max_response_output_tokens="inf",
)

rtv.start()

# Let's inject an initial message so we have a conversation started. We can do this at any point!
rtv.inject_message("Hello AI, what's up?")

try:
    while True:
        cmd = input("Commands: (p) pause, (r) resume, (s) stop, (i) inject <msg>, (u) update_instructions <text>, (t) print_transcript\n> ").strip()
        if cmd.lower() == "p":
            rtv.pause()
        elif cmd.lower() == "r":
            rtv.resume()
        elif cmd.lower() == "s":
            rtv.stop()
            break
        elif cmd.lower().startswith("i "):
            message = cmd[len("i "):].strip()
            rtv.inject_message(message)
        elif cmd.lower().startswith("u "):
            new_instructions = cmd[len("u "):].strip()
            rtv.update_instructions(new_instructions)
        elif cmd.lower() == "t":
            print("\n".join([f"{entry.timestamp} {entry.role}: {entry.message}" for entry in rtv.transcripts]))
        else:
            print("Unknown command.")
except KeyboardInterrupt:
    rtv.stop()
    print("\nExiting.")
```


## OpenAI
```python
import lunar_tools as lt
openai = lt.OpenAIWrapper()
msg = openai.generate("tell me about yourself")
```

## Speech2Text
```python
import lunar_tools as lt
import time
speech_detector = lt.Speech2Text()
speech_detector.start_recording()
time.sleep(3)
translation = speech_detector.stop_recording()
print(f"translation: {translation}")
```

The playback is threaded and does not block the main application. You can stop the playback via: 
```python
player.stop_sound()
```

## Text2Speech OpenAI
```python
import lunar_tools as lt
text2speech = lt.Text2SpeechOpenAI()
text2speech.change_voice("nova")
text2speech.generate("hey there can you hear me?", "hervoice.mp3")
```

The Text2Speech can also directly generate and play back the sound via: 
```python
text2speech.play("hey there can you hear me?")
```

## Text2Speech elevenlabs
```python
text2speech = Text2SpeechElevenlabs()
text2speech.change_voice("FU5JW1L0DwfWILWkNpW6")
text2speech.play("hey there can you hear me?")
```

# Image generation APIs
## Generate Images with Dall-e-3
```python
import lunar_tools as lt
dalle3 = lt.Dalle3ImageGenerator()
image, revised_prompt = dalle3.generate("a beautiful red house with snow on the roof, a chimney with smoke")
```

## Generate Images with SDXL Turbo
```python
import lunar_tools as lt
sdxl_turbo = lt.SDXL_TURBO()
image, img_url = sdxl_turbo.generate("An astronaut riding a rainbow unicorn", "cartoon")
```


# Communication
## WebRTC Data Channels
Low-latency data channel built on WebRTC for streaming numpy arrays, JSON blobs, PNG previews, and log text. Requires the optional `aiortc` extra (`python -m pip install "lunar_tools[webrtc]"`).

Sender (hosts an embedded signaling server and streams mixed payloads):

```bash
python examples/comms/webrtc_sender.py --session demo
```

Receiver (auto-discovers the sender session via the cached signaling endpoint):

```bash
python examples/comms/webrtc_receiver.py --session demo
```

- `--sender-ip` defaults to the detected local address (via `lunar_tools.comms.utils.get_local_ip`).
- When the sender hosts the embedded signaling server it stores the endpoint details per session in `~/.lunar_tools/webrtc_sessions.json`. Receivers can omit `--sender-ip` to reuse the most recent entry for the requested session, which keeps the bootstrap process simple.
- If you prefer using your own signaling server, start it separately (or pass `--no-server` in the sender example) and point both peers to the same `http://<sender-ip>:<port>` URL.

## OSC
High-level OSC helper built on python-osc. The receiver example spawns the live grid visualizer, and the sender emits demo sine/triangle waves.

Receiver:

```bash
python examples/comms/osc_receiver.py --ip 0.0.0.0 --port 8003
```

Sender:

```bash
python examples/comms/osc_sender.py --ip 127.0.0.1 --port 8003 --channels /env1 /env2 /env3
```

## ZMQ Pair Endpoint
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

# Logging and terminal printing
```python
import lunar_tools as lt
logger = lt.LogPrint()  # No filename provided, will use default current_dir/logs/%y%m%d_%H%M
logger.print("white")
logger.print("red", "red")
logger.print("green", "green")
```

# Health status reporting via telegram
Obtain a bot here: https://docs.tracardi.com/qa/how_can_i_get_telegram_bot/
Next you will need to update your bashrc or bash_profile with the telegram bot env variables.


```bash
export TELEGRAM_BOT_TOKEN='XXX'
export TELEGRAM_CHAT_ID='XXX'
```

See the below example for the basic supported features
```python
health_reporter = lt.HealthReporter("Name of exhibit")
    
# in while loop, always report that exhibit is alive. This happens in a thread already.
for i in range(100): #while lopp
        health_reporter.report_alive()
    
# we can also just throw exceptions in there!
try: 
    x = y
except Exception as e:
    health_reporter.report_exception(e)
    
# or send something friendly!
health_reporter.report_message("friendly manual message")
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
