# Audio and Voice

Everything in this section requires the `audio` extra (`python -m pip install lunar_tools[audio]`). Optional LLM integrations also need the `llm` extra.

## Recording and transcription

### Speech2Text

`Speech2Text` wraps OpenAI’s Whisper models. Start and stop recording around the section you want to transcribe.

```python
import time
import lunar_tools as lt

speech = lt.Speech2Text()
speech.start_recording()
time.sleep(4)
print("Transcript:", speech.stop_recording())
```

Notes:
- Whisper runs against the `OPENAI_API_KEY` you expose via the environment.
- If you need offline transcription, instantiate with `offline_model_type="small"` (requires the `whisper` package).
- For live captions use `lt.RealTimeTranscribe`, shown below.

### RealTimeTranscribe

Deepgram real-time streaming wires into an async callback that fires for every partial result. Make sure you have the `DEEPGRAM_API_KEY` environment variable set.

```python
import time
import lunar_tools as lt

transcriber = lt.RealTimeTranscribe(auto_start=True, model="nova-3")
transcriber.wait_until_ready(timeout=5)

try:
    while True:
        print("Latest:", transcriber.get_text())
        time.sleep(2)
finally:
    transcriber.stop()
```

## Text-to-speech

### OpenAI voices

```python
import lunar_tools as lt

tts = lt.Text2SpeechOpenAI(voice="alloy")
tts.generate("Welcome to the gallery.", "output/opening.mp3")
```

Change the voice on the fly:

```python
tts.change_voice("nova")
tts.play("Lights dimming now.")  # Direct playback via simpleaudio
```

### ElevenLabs voices

```python
import lunar_tools as lt

voice_id = "FU5JW1L0DwfWILWkNpW6"
tts = lt.Text2SpeechElevenlabs(voice=voice_id)
tts.play("This kiosk is now online.")
```

Set the `ELEVEN_API_KEY` environment variable before running.

## Playback helpers

`SoundPlayer` uses `simpleaudio` for lightweight playback. It is ideal for short cues or verifying generated speech.

```python
import lunar_tools as lt

player = lt.SoundPlayer(blocking_playback=True)
player.play_sound("output/opening.mp3")
```

Pass `pan_value` between -1 (left) and +1 (right) to position the sound.

## Realtime voice conversations

`RealTimeVoice` is a turnkey loop that records the microphone, streams audio to OpenAI’s realtime endpoint, and plays responses back. It exposes optional async hooks so you can mirror transcripts or trigger side effects.

```python
import time
import lunar_tools as lt

async def on_user(text: str):
    print("[user]", text)

async def on_ai(text: str):
    print("[ai]", text)

voice = lt.RealTimeVoice(
    instructions="Keep replies short and playful.",
    on_user_transcript=on_user,
    on_ai_transcript=on_ai,
    voice="alloy",
    temperature=0.7,
)

voice.start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    voice.stop()

Callbacks are awaited internally, so you can update UI state, trigger lighting cues, or store transcripts asynchronously.
```

Quick controls:
- `voice.pause()` / `voice.resume()` toggle microphone streaming.
- `voice.inject_message("Stage lights triggered.")` primes the assistant with extra context.
- `voice.update_instructions("Switch to concierge mode.")` changes behaviour mid-show.

For an interactive CLI walkthrough see [`examples/realtime_voice_example.py`](../examples/realtime_voice_example.py).

## LLM convenience wrapper (`llm` extra)

The `OpenAIWrapper` exposes a minimal `.generate(prompt)` helper that picks sensible defaults. It is handy for scripting without pulling in the realtime stack.

```python
import lunar_tools as lt

client = lt.OpenAIWrapper(model="gpt-4o-mini")
print(client.generate("List three ambient lighting moods."))
```

To handle multi-turn chat use `.chat(messages=[...])` or drop down to the official SDK.
