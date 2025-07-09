import asyncio
import base64
import json
from typing import Any, Dict, Optional, Callable, Awaitable, List
import sounddevice as sd
import numpy as np
import threading
import time

# Optional OpenAI realtime imports
try:
    from openai import AsyncOpenAI
    from openai.types.beta.realtime.session import Session
    from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
    OPENAI_REALTIME_AVAILABLE = True
except ImportError as e:
    OPENAI_REALTIME_AVAILABLE = False
    print(f"OpenAI realtime not available: {e}")
    # Create dummy classes for type hints
    AsyncOpenAI = None
    Session = None
    AsyncRealtimeConnection = None

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
    def __init__(self, on_playback_complete: Optional[Callable[[], None]] = None, verbose: bool = False):
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

        # AI speech tracking
        self.is_currently_speaking = False
        self._last_nonzero_timestamp = 0.0
        self.timeout_ai_talking = 0.6

        # Playback-complete callback
        self.on_playback_complete = on_playback_complete
        self._playback_complete_triggered = False

        self.verbose = verbose

    def callback(self, outdata, frames, time_info, status):
        with self.lock:
            data = np.empty(0, dtype=FORMAT)

            # Pull frames from the queue
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                needed = frames - len(data)
                data = np.concatenate((data, item[:needed]))
                if len(item) > needed:
                    self.queue.insert(0, item[needed:])

            self._frame_count += len(data)

            # If not enough data, fill with zeros
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=FORMAT)))

        # Determine if there's non-silent audio
        if np.any(data != 0):
            self._last_nonzero_timestamp = time.time()

        if time.time() - self._last_nonzero_timestamp < self.timeout_ai_talking:
            self.is_currently_speaking = True
        else:
            self.is_currently_speaking = False

        outdata[:] = data.reshape(-1, 1)

        # Check if the queue is now empty => possible playback completion
        with self.lock:
            # We no longer require 'not self.playing'. We only check empty queue.
            if not self.queue and not self._playback_complete_triggered and self.on_playback_complete:
                # Trigger only once
                self._playback_complete_triggered = True
                self.on_playback_complete()

    def add_data(self, data: bytes):
        """
        Add a chunk of AI audio to the queue.
        Reset the _playback_complete_triggered so that a new callback can fire.
        """
        with self.lock:
            audio_array = np.frombuffer(data, dtype=FORMAT)
            self.queue.append(audio_array)
            if not self.playing:
                self.start()
            # Because we're adding new audio, reset the triggered flag
            self._playback_complete_triggered = False

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

    def terminate(self):
        self.stream.close()

class RealTimeVoice:
    def __init__(
        self,
        instructions: str,
        on_user_transcript: Optional[Callable[[str], Awaitable[None]]] = None,
        on_ai_transcript: Optional[Callable[[str], Awaitable[None]]] = None,
        on_ai_audio_complete: Optional[Callable[[], Awaitable[None]]] = None,
        model="gpt-4o-mini-realtime-preview-2024-12-17",
        temperature=0.6,
        max_response_output_tokens="inf",
        voice="alloy",
        mute_mic_while_ai_speaking=True,
        verbose: bool = False,
    ):
        if not OPENAI_REALTIME_AVAILABLE:
            raise ImportError(
                "OpenAI realtime dependencies are not available. "
                "Please install with: pip install openai[realtime] or check your OpenAI library version."
            )
        self.instructions = instructions
        self.on_user_transcript = on_user_transcript
        self.on_ai_transcript = on_ai_transcript
        self.on_ai_audio_complete = on_ai_audio_complete

        self.model = model
        self.temperature = temperature
        self.max_response_output_tokens = max_response_output_tokens
        self.mute_mic_while_ai_speaking = mute_mic_while_ai_speaking
        self.verbose = verbose
        self._mic_muted = False

        # Keep track of transcripts
        self.transcripts: List[TranscriptEntry] = []

        # Audio player with custom callback
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync(
            on_playback_complete=self.onAIAudioComplete,
            verbose=self.verbose
        )

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
                "prefix_padding_ms": 500,
                "silence_duration_ms": 1000,
            },
            "tools": [],
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
        }

        # Threading & control
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._thread: Optional[threading.Thread] = None
        self.conn: Optional[AsyncRealtimeConnection] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connection_ready = threading.Event()

        # We'll set this True whenever we see 'response.done' that includes audio
        self._audio_complete_pending = False
        self._audio_complete_lock = threading.Lock()

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

                # Mute if AI is speaking or mic is muted
                if self._mic_muted or (self.mute_mic_while_ai_speaking and self.audio_player.is_currently_speaking):
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
        print("Connecting to the Realtime API...")
        try:
            async with self.client.beta.realtime.connect(model=self.model) as conn:
                self.conn = conn
                self._connection_ready.set()
                if self.verbose:
                    print("Real-time session established.")

                # Update session
                await conn.session.update(session=self.REALTIME_API_CONFIG)
                if self.verbose:
                    print("Session parameters updated.")

                mic_task = asyncio.create_task(self._send_mic_audio(conn))
                acc_items: Dict[str, str] = {}

                try:
                    async for event in conn:
                        if self._stop_event.is_set():
                            if self.verbose:
                                print("Stop event detected. Exiting event loop.")
                            break
                        if not self._pause_event.is_set():
                            continue

                        if event.type == "session.created":
                            if self.verbose:
                                print(f"Session created with ID: {event.session.id}")

                        elif event.type == "session.updated":
                            if self.verbose:
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
                            if self.on_user_transcript:
                                asyncio.create_task(self.on_user_transcript(user_message))

                        elif event.type == "response.done":
                            # The model finished an output turn
                            # If there's audio in that response, we want the playback-complete callback
                            if event.response and event.response.output:
                                for item in event.response.output:
                                    if item.type == "message":
                                        for content in item.content:
                                            if content.type == "audio":
                                                # We got AI audio
                                                ai_message = content.transcript
                                                self.transcripts.append(
                                                    TranscriptEntry(role="ai", message=ai_message)
                                                )
                                                if self.on_ai_transcript:
                                                    asyncio.create_task(self.on_ai_transcript(ai_message))
                                                # Mark that we expect an audio-complete event
                                                with self._audio_complete_lock:
                                                    self._audio_complete_pending = True
                        else:
                            if self.verbose:
                                print(f"Unhandled event type: {event.type}")  # Print unhandled events
                except asyncio.CancelledError:
                    if self.verbose:
                        print("Main loop cancelled.")
                    print("Main loop has been cancelled.")
        except Exception as e:
            if self.verbose:
                print(f"An error occurred in the main loop: {e}")
        finally:
            print("Connection closed. Exiting.")

    def _run_loop(self):
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
        if self._thread and self._thread.is_alive():
            if self.verbose:
                print("RealTimeVoice is already running.")
            return
        if self.verbose:
            print("Starting RealTimeVoice...")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print("RealTimeVoice started.")

    def stop(self):
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
        print("Pausing RealTimeVoice...")
        self._pause_event.clear()

    def resume(self):
        print("Resuming RealTimeVoice...")
        self._pause_event.set()

    def mute_mic(self):
        self._mic_muted = True
        if self.verbose:
            print("Microphone muted.")

    def unmute_mic(self):
        self._mic_muted = False
        if self.verbose:
            print("Microphone unmuted.")

    def inject_message(self, message: str):
        async def _inject():
            if not self.conn:
                print("No connection available for injecting message.")
                return
            try:
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
            except Exception as e:
                print(f"Error injecting message: {e}")

        if not self._connection_ready.is_set():
            print("Waiting for connection to be ready to inject message...")
            self._connection_ready.wait()

        if not self._loop or not self._loop.is_running():
            print("Event loop not running. Cannot inject message.")
            return

        future = asyncio.run_coroutine_threadsafe(_inject(), self._loop)
        try:
            future.result(timeout=5)
        except Exception as e:
            print(f"Error scheduling inject_message coroutine: {e}")

    def update_instructions(self, new_instructions: str):
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

        if not self._connection_ready.is_set():
            print("Waiting for connection to be ready to update instructions...")
            self._connection_ready.wait()

        if not self._loop or not self._loop.is_running():
            print("Event loop not running. Cannot update instructions.")
            return

        future = asyncio.run_coroutine_threadsafe(_update(), self._loop)
        try:
            future.result(timeout=5)
        except Exception as e:
            print(f"Error scheduling update_instructions coroutine: {e}")

    def onAIAudioComplete(self):
        """
        Called by AudioPlayerAsync once the queue is empty.
        We'll fire our on_ai_audio_complete callback only if we know
        the AI actually produced audio (i.e., _audio_complete_pending).
        """
        with self._audio_complete_lock:
            if self._audio_complete_pending and self.on_ai_audio_complete:
                if self.verbose:
                    print("Playback complete and AI audio complete callback triggered.")
                self._audio_complete_pending = False  # consume the flag
                if self._loop and self._loop.is_running():
                    async def _callback():
                        await self.on_ai_audio_complete()

                    asyncio.run_coroutine_threadsafe(_callback(), self._loop)
                else:
                    print("Event loop not running. Cannot execute on_ai_audio_complete callback.")

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    if not OPENAI_REALTIME_AVAILABLE:
        print("OpenAI realtime dependencies are not available. Cannot run example.")
        exit(1)
        
    instructions = "Respond briefly and with a sarcastic attitude."
    temperature = 0.9
    voice = "echo"
    mute_mic_while_ai_speaking = True
    verbose = False  # Enable verbose prints

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
        verbose=verbose,  # Pass verbose flag
    )

    rtv.start()
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
