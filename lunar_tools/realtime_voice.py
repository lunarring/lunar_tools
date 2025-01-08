import asyncio
import base64
import json
from typing import Any, Dict, Optional, Callable, Awaitable, List
import sounddevice as sd
import numpy as np
import threading
from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
import time

from dataclasses import dataclass
from datetime import datetime

CHUNK_LENGTH_S = 0.05  # 50ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1

@dataclass
class TranscriptEntry:
    role: str  # 'user' or 'assistant'
    message: str
    timestamp: datetime = datetime.now()

class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=FORMAT,
            blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
        )
        self.playing = False
        self._frame_count = 0
        self.is_currently_speaking = False
        self._last_nonzero_timestamp = 0.0
        self.timeout_ai_talking = 0.6

    def callback(self, outdata, frames, time_info, status):  # noqa
        with self.lock:
            data = np.empty(0, dtype=FORMAT)
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                needed = frames - len(data)
                data = np.concatenate((data, item[:needed]))
                if len(item) > needed:
                    self.queue.insert(0, item[needed:])

            self._frame_count += len(data)
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=FORMAT)))

        # Track speaking state
        if np.any(data != 0):
            self._last_nonzero_timestamp = time.time()
        if time.time() - self._last_nonzero_timestamp < self.timeout_ai_talking:
            self.is_currently_speaking = True
        else:
            self.is_currently_speaking = False

        outdata[:] = data.reshape(-1, 1)

    def start(self):
        if not self.playing:
            self.playing = True
            self.stream.start()

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []

    def add_data(self, data: bytes):
        with self.lock:
            self.queue.append(np.frombuffer(data, dtype=FORMAT))
            if not self.playing:
                self.start()

class RealTimeVoice:
    def __init__(
        self,
        instructions: str,
        on_user_message: Optional[Callable[[str], Awaitable[None]]] = None,
        on_ai_message: Optional[Callable[[str], Awaitable[None]]] = None,
        model="gpt-4o-mini-realtime-preview-2024-12-17",
        temperature=0.6,
        max_response_output_tokens="inf",
        voice="alloy",
        mute_mic_while_ai_speaking=True,
    ):
        self.instructions = instructions
        self.on_user_message = on_user_message
        self.on_ai_message = on_ai_message
        self.model = model
        self.temperature = temperature
        self.max_response_output_tokens = max_response_output_tokens
        self.mute_mic_while_ai_speaking = mute_mic_while_ai_speaking

        # Transcripts
        self.transcripts: List[TranscriptEntry] = []

        # OpenAI client and audio player
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()

        # Realtime config
        self.REALTIME_API_CONFIG = {
            "modalities": ["text", "audio"],
            "instructions": self.instructions,
            "voice": voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 100,
                "silence_duration_ms": 1000,
            },
            "tools": [],
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
        }

        # Control flags
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially

        # Thread for running the async loop
        self._thread: Optional[threading.Thread] = None

        # Connection and event loop
        self.conn: Optional[AsyncRealtimeConnection] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Event to signal that connection is ready
        self._connection_ready = threading.Event()

    async def _send_mic_audio(self, connection: AsyncRealtimeConnection) -> None:
        read_size = int(SAMPLE_RATE * 0.02)
        stream = sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype="int16")
        stream.start()
        print("Microphone recording started.")

        try:
            while not self._stop_event.is_set():
                if not self._pause_event.is_set():
                    await asyncio.sleep(0.1)
                    continue

                if stream.read_available < read_size:
                    await asyncio.sleep(0.01)
                    continue

                data, _ = stream.read(read_size)

                # Mute mic while AI is speaking, if requested
                if self.mute_mic_while_ai_speaking and self.audio_player.is_currently_speaking:
                    data = np.zeros_like(data)

                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(data).decode("utf-8")
                )
                await asyncio.sleep(0)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Stopping microphone stream.")
        finally:
            stream.stop()
            stream.close()
            if not self._stop_event.is_set():
                print("Committing final audio buffer and requesting response.")
                await connection.input_audio_buffer.commit()
                await connection.response.create()
                print("Done.")

    async def _main_loop(self):
        print("Connecting to the Realtime API...")
        async with self.client.beta.realtime.connect(model=self.model) as conn:
            self.conn = conn
            self._connection_ready.set()
            print("Real-time session established.")

            # Update session parameters
            await conn.session.update(session=self.REALTIME_API_CONFIG)

            # Start sending mic audio
            mic_task = asyncio.create_task(self._send_mic_audio(conn))
            acc_items: Dict[str, str] = {}

            try:
                async for event in conn:
                    if self._stop_event.is_set():
                        break
                    if not self._pause_event.is_set():
                        continue

                    if event.type == "session.created":
                        print(f"Session created with ID: {event.session.id}")

                    elif event.type == "session.updated":
                        print("Session updated.")

                    elif event.type == "response.audio.delta":
                        chunk_bytes = base64.b64decode(event.delta)
                        self.audio_player.add_data(chunk_bytes)

                    elif event.type == "response.audio_transcript.delta":
                        item_id = event.item_id
                        if item_id not in acc_items:
                            acc_items[item_id] = event.delta
                        else:
                            acc_items[item_id] += event.delta

                    elif event.type == "conversation.item.input_audio_transcription.completed":
                        user_message = event.transcript
                        self.transcripts.append(TranscriptEntry(role="user", message=user_message))
                        if self.on_user_message:
                            asyncio.create_task(self.on_user_message(user_message))

                    elif event.type == "response.done":
                        if event.response and event.response.output:
                            for item in event.response.output:
                                if item.type == "message":
                                    for content in item.content:
                                        if content.type == "audio":
                                            ai_message = content.transcript
                                            self.transcripts.append(
                                                TranscriptEntry(role="ai", message=ai_message)
                                            )
                                            if self.on_ai_message:
                                                asyncio.create_task(self.on_ai_message(ai_message))

            except asyncio.CancelledError:
                print("Main loop cancelled.")
            finally:
                mic_task.cancel()
                await mic_task
                print("Connection closed. Exiting.")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main_loop())
        self._loop.close()

    def start(self):
        if self._thread and self._thread.is_alive():
            print("RealTimeVoice is already running.")
            return
        print("Starting RealTimeVoice...")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        print("Stopping RealTimeVoice...")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self.audio_player.stop()
        print("RealTimeVoice stopped.")

    def pause(self):
        print("Pausing RealTimeVoice...")
        self._pause_event.clear()

    def resume(self):
        print("Resuming RealTimeVoice...")
        self._pause_event.set()

    def inject_message(self, message: str):
        async def _inject():
            if not self.conn:
                print("No connection available for injecting message.")
                return
            trigger = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            }
            await self.conn.send(trigger)
            await self.conn.send({"type": "response.create", "response": {"modalities": ["audio", "text"]}})
            print(f"Injected message: {message}")

        if not self._connection_ready.is_set():
            print("Waiting for connection to be ready to inject message...")
            self._connection_ready.wait()

        if not self._loop or not self._loop.is_running():
            print("Event loop not running. Cannot inject message.")
            return

        asyncio.run_coroutine_threadsafe(_inject(), self._loop)

    def update_instructions(self, new_instructions: str):
        """
        Dynamically updates the session instructions.
        Remember: changes take effect on the *next* conversation turn.
        """
        async def _update():
            if not self.conn:
                print("No connection available for updating instructions.")
                return
            self.instructions = new_instructions
            updated_config = dict(self.REALTIME_API_CONFIG)
            updated_config["instructions"] = self.instructions
            await self.conn.session.update(session=updated_config)
            print(f"Session instructions updated to: {new_instructions}")

        if not self._connection_ready.is_set():
            print("Waiting for connection to be ready to update instructions...")
            self._connection_ready.wait()

        if not self._loop or not self._loop.is_running():
            print("Event loop not running. Cannot update instructions.")
            return

        asyncio.run_coroutine_threadsafe(_update(), self._loop)

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    instructions = "Respond briefly and with a sarcastic attitude."
    temperature = 0.6
    voice = "echo"
    mute_mic_while_ai_speaking = True # that's the default already, just FYI

    # Example callback
    async def on_user_message(transcript: str):
        print(f"(on_user_message) User said: {transcript}")

    async def on_ai_message(transcript: str):
        print(f"(on_ai_message) AI replied: {transcript}")

    rtv = RealTimeVoice(
        instructions=instructions,
        on_user_message=on_user_message,
        on_ai_message=on_ai_message,
        model="gpt-4o-mini-realtime-preview-2024-12-17",
        temperature=temperature,
        voice=voice,
        mute_mic_while_ai_speaking=mute_mic_while_ai_speaking,
    )

    rtv.start()

    # Let's inject an initial message so we have a conversation started
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