import asyncio
import base64
import json
from typing import Any, Dict, Optional, Callable, Awaitable
import sounddevice as sd
import numpy as np
import threading
from openai import OpenAI
from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
import numpy as np
import time

CHUNK_LENGTH_S = 0.05  # 50ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1

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

        # Added variable to track speaking state
        self.is_currently_speaking = False
        self._last_nonzero_timestamp = 0.0
        self.timeout_ai_talking = 0.6

    def callback(self, outdata, frames, time_info, status):  # noqa
        with self.lock:
            data = np.empty(0, dtype=FORMAT)

            # Grab next items from the queue if there's space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # Fill the rest of the frames with zeros if no more data
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=FORMAT)))

        # Check if we have non-zero data
        if np.any(data != 0):
            self._last_nonzero_timestamp = time.time()

        # Update speaking state based on time since last non-zero data
        if time.time() - self._last_nonzero_timestamp < self.timeout_ai_talking:
            self.is_currently_speaking = True
        else:
            self.is_currently_speaking = False

        outdata[:] = data.reshape(-1, 1)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        with self.lock:
            np_data = np.frombuffer(data, dtype=FORMAT)
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        self.playing = True
        self.stream.start()

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []

    def terminate(self):
        self.stream.close()


async def send_audio_worker_sounddevice(
    connection: AsyncRealtimeConnection,
    should_send: Callable[[], bool] | None = None,
    start_send: Callable[[], Awaitable[None]] | None = None,
):
    sent_audio = False

    device_info = sd.query_devices()
    print(device_info)

    read_size = int(SAMPLE_RATE * 0.02)
    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
    )
    stream.start()

    try:
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue

            data, _ = stream.read(read_size)
            if should_send() if should_send else True:
                if not sent_audio and start_send:
                    await start_send()
                await connection.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode("utf-8"),
                    }
                )
                sent_audio = True
            elif sent_audio:
                print("Done, triggering inference")
                await connection.send({"type": "input_audio_buffer.commit"})
                await connection.send({"type": "response.create", "response": {}})
                sent_audio = False

            await asyncio.sleep(0)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

class RealTimeVoice:
    def __init__(
        self,
        instructions: str, 
        on_user_message: Optional[Callable[[str], None]] = None,
        model="gpt-4o-mini-realtime-preview-2024-12-17", 
        temperature=0.6,
        max_response_output_tokens="inf",
        trigger_message=None,
        voice="alloy",
    ):
        """
        Initialize the RealTimeVoice manager.
        instructions: The instructions for the model.
        on_user_message: An optional async callback function that will be invoked with the text
            of the user's completed transcription.
        model: The model to be used. Default is "gpt-4o-mini-realtime-preview-2024-12-17".
        temperature: The temperature for the model's output. Default is 0.6.
        max_response_output_tokens: The maximum number of output tokens for the model's response. Default is "inf".
        trigger_message: The trigger message for the model.
        voice: The voice for the model. Default is "alloy". Supported voices are alloy, ash, coral, echo, fable, onyx, nova, sage and shimmer.
        """
        self.on_user_message = on_user_message
        self.model = model
        self.temperature = temperature
        self.max_response_output_tokens = max_response_output_tokens

        # Spawn our own client and audio player here.
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.trigger_message = trigger_message

        self.REALTIME_API_CONFIG = dict(
            modalities=["text", "audio"],
            instructions=instructions,
            voice=voice,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            input_audio_transcription=dict(model="whisper-1"),
            turn_detection=dict(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=100,
                silence_duration_ms=1000,
            ),
            tools=[],
            tool_choice="auto",
            temperature=self.temperature,
            max_response_output_tokens=self.max_response_output_tokens,
        )

    async def _send_mic_audio(
        self,
        connection: AsyncRealtimeConnection
    ) -> None:
        """
        Sends microphone audio to the model in real time.
        """
        sent_audio_cancel = False
        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype="int16")
        stream.start()

        print("Microphone recording started. Press Ctrl+C to quit.")

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                # Read a small chunk from the mic.
                data, _ = stream.read(read_size)

                if self.audio_player.is_currently_speaking:
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
            print("Committing final audio buffer and requesting response.")
            await connection.input_audio_buffer.commit()
            await connection.response.create()
            print("Done.")

    async def _main_loop(self) -> None:
        """
        The main async logic that:
          - Connects to Realtime,
          - Updates session params (including the tool definitions),
          - Spawns mic audio streaming,
          - Handles model events, and
          - Invokes the `on_user_message` callback for completed transcripts.
        """
        print("Connecting to the Realtime API...")

        async with self.client.beta.realtime.connect(model=self.model) as conn:
            print("Connection established. Waiting for events...")

            # Override session parameters with our function definitions etc.
            await conn.session.update(session=self.REALTIME_API_CONFIG)

            # Start sending mic audio
            mic_task = asyncio.create_task(self._send_mic_audio(conn))

            # Accumulate transcript items by item_id
            acc_items: Dict[str, str] = {}

            # Send an initial user message to get the conversation going
            if self.trigger_message is not None:
                print("Sending initial assistant message (trigger_message) to start the conversation...")
                trigger = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": self.trigger_message,
                            }
                        ]
                    }
                }
                await conn.send(trigger)
                trigger = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"]
                    }
                }
                await conn.send(trigger)

            async for event in conn:
                if event.type == "session.created":
                    session = event.session
                    print(f"Session created with ID: {session.id}")

                elif event.type == "session.updated":
                    print("Session updated.")

                elif event.type == "response.audio.delta":
                    # Model is sending a chunk of audio to play
                    chunk_bytes = base64.b64decode(event.delta)
                    self.audio_player.add_data(chunk_bytes)

                elif event.type == "response.audio_transcript.delta":
                    # Model partial or final transcript text
                    item_id = event.item_id
                    if item_id not in acc_items:
                        acc_items[item_id] = event.delta
                    else:
                        acc_items[item_id] += event.delta

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    user_message = event.transcript
                    if self.on_user_message is not None:
                        async def do_callback():
                            await self.on_user_message(user_message)
                        asyncio.create_task(do_callback())

                # # Check if the model decided to call a function
                # elif event.type == "response.done":
                #     # Go through output items; if any are function calls, we handle them
                #     if event.response and event.response.output:
                #         for item in event.response.output:
                #             if item.type == "function_call":
                #                 # Parse arguments
                #                 func_name = item.name
                #                 call_id = item.call_id
                #                 args = json.loads(item.arguments)
                #                 # If it's our known function, handle it
                #                 if func_name == "redraw_image":
                #                     prompt = args["prompt"]
                #                     # Simulate the "redraw" with flux/renderer
                #                     # Provide the result back to the model:
                #                     # conversation.item.create with type=function_call_output
                #                     redraw_output = {
                #                         "type": "conversation.item.create",
                #                         "item": {
                #                             "type": "function_call_output",
                #                             "call_id": call_id,
                #                             "output": json.dumps({"result": f"Redrew with prompt: {prompt}"})
                #                         }
                #                     }
                #                     await conn.send(redraw_output)
                #                     # Now ask model for a new response with the function result
                #                     await conn.send({"type": "response.create", "response": {}})

                # else:
                #   print(f"other event: {event}")

            mic_task.cancel()

        print("Connection closed. Exiting.")

    def start(self):
        """
        Public method to start the real-time voice functionality in a normal Python setting.
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._main_loop())





if __name__ == "__main__":

    instructions = "Respond in a sassy and short way."
    trigger_message = "Ask me what is my favorite thing in life!"

    # Optional: Set up the callback to handle user messages.
    async def on_user_message(transcript: str):
        print(f"on_user_message called, transcript: {transcript}")

    rtv = RealTimeVoice(instructions, on_user_message=on_user_message, trigger_message=trigger_message)
    rtv.start()
    print("Realtime voice session finished.")
