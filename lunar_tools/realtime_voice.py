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

# --------------------------
# Audio Configuration
# --------------------------
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

        # For tracking if AI is currently speaking
        self.is_currently_speaking = False
        self._last_nonzero_timestamp = 0.0
        self.timeout_ai_talking = 0.6

    def callback(self, outdata, frames, time_info, status):  # noqa
        with self.lock:
            data = np.empty(0, dtype=FORMAT)
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=FORMAT)))

        # Update speaking state
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
            print("AudioPlayerAsync: Stream started.")

    def stop(self):
        if self.playing:
            self.playing = False
            self.stream.stop()
            with self.lock:
                self.queue = []
            print("AudioPlayerAsync: Stream stopped.")

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
        """
        Initialize the RealTimeVoice manager.
        """
        self.instructions = instructions
        self.on_user_message = on_user_message
        self.on_ai_message = on_ai_message
        self.model = model
        self.temperature = temperature
        self.max_response_output_tokens = max_response_output_tokens
        self.mute_mic_while_ai_speaking = mute_mic_while_ai_speaking

        # List to hold transcripts
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
        """
        Sends microphone audio to the model in real time.
        """
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

                # Read a small chunk from the mic
                data, _ = stream.read(read_size)

                # Mute mic while AI is speaking, if requested
                if self.mute_mic_while_ai_speaking and self.audio_player.is_currently_speaking:
                    data = np.zeros_like(data)

                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(data).decode("utf-8")
                )
                await asyncio.sleep(0)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Microphone stream interrupted.")
        finally:
            stream.stop()
            stream.close()
            if not self._stop_event.is_set():
                print("Committing final audio buffer and requesting response.")
                await connection.input_audio_buffer.commit()
                await connection.response.create()
                print("Microphone stream stopped.")

    async def _main_loop(self):
        """
        Main async logic for real-time voice interaction.
        """
        print("Connecting to the Realtime API...")
        try:
            async with self.client.beta.realtime.connect(model=self.model) as conn:
                self.conn = conn
                self._connection_ready.set()
                print("Real-time session established.")

                # Update session parameters
                await conn.session.update(session=self.REALTIME_API_CONFIG)
                print("Session parameters updated.")

                # Start sending mic audio
                mic_task = asyncio.create_task(self._send_mic_audio(conn))
                acc_items: Dict[str, str] = {}

                try:
                    async for event in conn:
                        if self._stop_event.is_set():
                            print("Stop event detected. Exiting event loop.")
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
                        # You can handle other event types here if needed

                except asyncio.CancelledError:
                    print("Main loop cancelled.")
                finally:
                    mic_task.cancel()
                    try:
                        await mic_task
                    except asyncio.CancelledError:
                        pass
                    print("Main loop has been cancelled.")
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
        finally:
            print("Connection closed. Exiting.")

    def _run_loop(self):
        """
        Runs the asyncio event loop in a separate thread.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_loop())
        except Exception as e:
            print(f"Exception in run loop: {e}")
        finally:
            self._loop.close()
            print("Asyncio event loop closed.")

    def start(self):
        """
        Starts the RealTimeVoice in a separate thread.
        """
        if self._thread and self._thread.is_alive():
            print("RealTimeVoice is already running.")
            return

        print("Starting RealTimeVoice...")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("RealTimeVoice started.")

    def stop(self):
        """
        Stops the RealTimeVoice gracefully.
        """
        print("Stopping RealTimeVoice...")
        self._stop_event.set()

        if self.conn:
            print("Closing the connection...")
            future = asyncio.run_coroutine_threadsafe(self.conn.close(), self._loop)
            try:
                future.result(timeout=5)
                print("Connection closed successfully.")
            except Exception as e:
                print(f"Error closing connection: {e}")

        if self._thread and self._thread.is_alive():
            print("Waiting for the thread to finish...")
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                print("Thread did not finish in time.")
            else:
                print("Thread has been joined successfully.")

        self.audio_player.stop()
        print("RealTimeVoice stopped.")

    def pause(self):
        """
        Temporarily pause sending audio (and processing new events).
        """
        print("Pausing RealTimeVoice...")
        self._pause_event.clear()

    def resume(self):
        """
        Resumes sending audio after a pause.
        """
        print("Resuming RealTimeVoice...")
        self._pause_event.set()

    def inject_message(self, message: str):
        """
        Injects a user message dynamically into the conversation.
        """
        async def _inject():
            if not self.conn:
                print("No connection available for injecting message.")
                return
            try:
                # Create and send the user message
                trigger = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": message}],
                    },
                }
                await self.conn.send(trigger)

                # Request model response
                await self.conn.send({"type": "response.create", "response": {"modalities": ["audio", "text"]}})
                print(f"Injected message: {message}")
            except Exception as e:
                print(f"Error injecting message: {e}")

        # Wait until the connection is ready
        if not self._connection_ready.is_set():
            print("Waiting for connection to be ready to inject message...")
            self._connection_ready.wait()

        if not self._loop or not self._loop.is_running():
            print("Event loop not running. Cannot inject message.")
            return

        # Schedule the coroutine on our loop
        future = asyncio.run_coroutine_threadsafe(_inject(), self._loop)
        try:
            future.result(timeout=5)
        except Exception as e:
            print(f"Error scheduling inject_message coroutine: {e}")

    def update_instructions(self, new_instructions: str):
        """
        Dynamically updates the session instructions.
        Remember: changes take effect on the *next* conversation turn.
        """
        async def _update():
            if not self.conn:
                print("No connection available for updating instructions.")
                return
            try:
                self.instructions = new_instructions
                updated_config = dict(self.REALTIME_API_CONFIG)
                updated_config["instructions"] = self.instructions
                await self.conn.session.update(session=updated_config)
                print(f"Session instructions updated to: {new_instructions}")
            except Exception as e:
                print(f"Error updating instructions: {e}")

        # Wait until the connection is ready
        if not self._connection_ready.is_set():
            print("Waiting for connection to be ready to update instructions...")
            self._connection_ready.wait()

        if not self._loop or not self._loop.is_running():
            print("Event loop not running. Cannot update instructions.")
            return

        # Schedule the coroutine on our loop
        future = asyncio.run_coroutine_threadsafe(_update(), self._loop)
        try:
            future.result(timeout=5)
        except Exception as e:
            print(f"Error scheduling update_instructions coroutine: {e}")


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