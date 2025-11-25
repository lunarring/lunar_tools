#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
from pydub import AudioSegment
import threading
import os
import time
from openai import OpenAI
from lunar_tools.logprint import LogPrint
import simpleaudio
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play, save
import sounddevice as sd
import numpy as np
import wave
from pydub import AudioSegment
from lunar_tools.utils import read_api_key
import asyncio
from datetime import datetime
import logging
from contextlib import suppress

# Deepgram SDK imports are optional; guard at runtime if not installed
try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
        Microphone,
    )
    _HAS_DEEPGRAM = True
except Exception:
    _HAS_DEEPGRAM = False

class AudioRecorder:
    """
    A class to handle audio recording using sounddevice instead of pyaudio.

    Attributes:
        channels (int): Number of audio channels.
        rate (int): Sampling rate.
        chunk (int): Number of frames per buffer.
        frames (list): List to hold audio frames.
        is_recording (bool): Flag to check if recording is in progress.
        stream (sd.InputStream): Audio stream.
        output_filename (str): Output file name.
        logger: A logging instance. If None, a default logger will be used.
    """

    def __init__(
        self,
        channels=1,
        rate=44100,
        chunk=1024,
        logger=None
    ):
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.frames = []
        self.is_recording = False
        self.stream = None
        self.output_filename = None
        self.logger = logger if logger else LogPrint()

    def _record(self, max_time=None):
        self.stream = sd.InputStream(
            samplerate=self.rate,
            channels=self.channels,
            blocksize=self.chunk,
            dtype='float32'
        )
        self.logger.print("Recording...")
        self.frames = []
        start_time = time.time()
        with self.stream:
            while self.is_recording:
                if max_time and (time.time() - start_time) >= max_time:
                    break
                data, overflowed = self.stream.read(self.chunk)
                self.frames.append(data.flatten())

        self.logger.print("Finished recording.")
        
        # Convert to WAV and then to MP3
        wav_filename = tempfile.mktemp(suffix='.wav')
        wf = wave.open(wav_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)  # 2 bytes for 16-bit audio
        wf.setframerate(self.rate)
        self.frames = np.clip(self.frames, -1, +1)
        wf.writeframes(np.array(self.frames*32767).astype(np.int16).tobytes())
        wf.close()

        wav_audio = AudioSegment.from_wav(wav_filename)
        wav_audio.export(self.output_filename, format="mp3")

    def start_recording(self, output_filename=None, max_time=None):
        if not self.is_recording:
            self.is_recording = True
            if output_filename is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                self.output_filename = temp_file.name
                temp_file.close()
            else:
                output_filename = str(output_filename)
                if not output_filename.endswith('.mp3'):
                    raise ValueError("Output filename must have a .mp3 extension")
                self.output_filename = output_filename
            self.thread = threading.Thread(target=self._record, args=(max_time,))
            self.thread.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.thread.join()
            

class Speech2Text:
    def __init__(self, client=None, logger=None, audio_recorder=None, offline_model_type=None):
        """
        Initialize the Speech2Text with an OpenAI client, a logger, and an audio recorder.

        Args:
            client: An instance of OpenAI client. If None, it will be created using the OPENAI_API_KEY.
            logger: A logging instance. If None, a default logger will be used.
            audio_recorder: An instance of an audio recorder. If None, recording functionalities will be disabled.
            offline_model_type: An instance of an offline model for speech recognition. If None, it will use the API for transcription.

        Raises:
            ValueError: If no OpenAI API key is found in the environment variables.
        """
        self.transcript = None
        if client is None:
            api_key = read_api_key("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client

        if offline_model_type is not None:
            import whisper
            self.whisper_model = whisper.load_model(offline_model_type)
            self.offline_mode = True
        else:
            self.offline_mode = False
        
        self.logger = logger if logger else LogPrint()

        if audio_recorder is None:
            self.audio_recorder = AudioRecorder(logger=logger)
        else:
            self.audio_recorder = audio_recorder

    def start_recording(self, output_filename=None, max_time=None):
        """
        Start the audio recording.
        Args:
            output_filename (str): The filename for the output file. If None, a temporary file is created.
            max_time (int, optional): Maximum recording time in seconds.
        Raises:
            ValueError: If the audio recorder is not available.
        """
        if self.audio_recorder is None:
            raise ValueError("Audio recorder is not available")
        self.audio_recorder.start_recording(output_filename, max_time)


    def stop_recording(self, minimum_duration=0.4):
        """
Stop the audio recording and check if the recording meets the minimum duration.

Args:
    minimum_duration (float, optional): The minimum duration in seconds for a recording to be valid.
                                      Default is 1 second.

Returns:
    str: The transcribed text if the recording meets the minimum duration requirement, otherwise None.

Raises:
    ValueError: If the audio recorder is not available.
"""
        if self.audio_recorder is None:
            raise ValueError("Audio recorder is not available")
        self.audio_recorder.stop_recording()

        audio_duration = AudioSegment.from_mp3(self.audio_recorder.output_filename).duration_seconds
        if audio_duration < minimum_duration:
            self.logger.print(f"Recording is too short, only {audio_duration:.2f} seconds. Minimum required is {minimum_duration} seconds.")
            return None
        return self.translate(self.audio_recorder.output_filename)
    
    def translate(self, audio_filepath):
        """
        Translate the audio file to text using OpenAI's translation model.

        Args:
            audio_filepath: The file path of the audio file to be translated.

        Returns:
            str: The transcribed text.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")
        if self.offline_mode:
            audio_segment = AudioSegment.from_file(audio_filepath)
            # Convert audio_segment to a numpy array
            # Note: Pydub's samples are interleaved, so for multi-channel audio, every Nth sample is a sample from a different channel
            numpydata = np.array(audio_segment.get_array_of_samples()).astype(np.int16)
            numpydata = np.hstack(numpydata).astype(np.float32)
            numpydata = numpydata.astype(np.float32) / 32768.0  # Convert to float32 in range [-1, 1]
            options = dict(language="english", beam_size=5, best_of=5)
            translate_options = dict(task="translate", **options)
            result = self.whisper_model.transcribe(numpydata, **translate_options)
            return result["text"].strip()
        else:
            with open(audio_filepath, "rb") as audio_file:
                transcript = self.client.audio.translations.create(
                    model="whisper-1", 
                    file=audio_file
                )
                return transcript.text
            
    def handle_unmute_button(self, mic_button_state: bool):
        if mic_button_state:
            if not self.audio_recorder.is_recording:
                self.start_recording()
        else:
            if self.audio_recorder.is_recording:
                try:
                    transcript = self.stop_recording()
                    transcript = transcript.strip().lower()
                    self.transcript = transcript
                    return True
                except Exception as e:
                    print(f"Error stopping recording: {e}")
        return False


class Text2SpeechOpenAI:
    def __init__(
        self, 
        client=None, 
        logger=None, 
        voice_model="nova", 
        sound_player=None,
        blocking_playback=False
    ):
        """
        Initialize the Text2Speech with an OpenAI client, a logger, a text source, a default voice model, and optionally a sound player.
        """
        if client is None:
            api_key = read_api_key("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client
        self.logger = logger if logger else LogPrint()
        # Initialize the sound player only if provided
        self.sound_player = sound_player
        self.output_filename = None  # Initialize output filename
        self.voice_model = voice_model
        self.blocking_playback = blocking_playback

    def play(self, text=None):
        """
        Play a generated speech file. Instantiates SoundPlayer if it's not already.
        """
        self.generate(text)
        if self.sound_player is None:
            self.sound_player = SoundPlayer(blocking_playback=self.blocking_playback)
        self.sound_player.play_sound(self.output_filename)

    def stop(self):
        """
        Stop the currently playing speech file.
        """
        if self.sound_player:
            self.sound_player.stop_sound()


    def generate(self, text, output_filename=None):
        """
        Generate speech from text.
    
        Args:
            text (str): The text to be converted into speech. 
            output_filename (str): The filename for the output file. If None, a default filename is used.
    
        Raises:
            ValueError: If the text source is not available.
        """
        if text is None or len(text)==0:
            raise ValueError("text is invalid!")
    
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice_model,
            input=text
        )
        
        self.output_filename = output_filename if output_filename else "output_speech.mp3"
        response.stream_to_file(self.output_filename)
        self.logger.print(f"Generated speech saved to {self.output_filename}")


    def change_voice(self, new_voice):
        """
        Change the voice model for speech generation.

        Args:
            new_voice (str): The new voice model to be used.
        """
        if new_voice in self.list_available_voices():
            self.voice_model = new_voice
            self.logger.print(f"Voice model changed to {new_voice}")
        else:
            raise ValueError(f"Voice '{new_voice}' is not a valid voice model.")

    @staticmethod
    def list_available_voices():
        """
        List the available voice models.

        Returns:
            list: A list of available voice models.
        """
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    
class Text2SpeechElevenlabs:
    def __init__(
        self,
        logger=None,
        sound_player=None,
        voice_id=None,
        blocking_playback=False
    ):
        """
        Initialize the Text2Speech for elevenlabs, a optional logger and optionally a sound player.
        """
        self.client = ElevenLabs(api_key=read_api_key("ELEVEN_API_KEY"))
        self.logger = logger if logger else LogPrint()
        # Initialize the sound player only if provided
        self.sound_player = sound_player
        self.output_filename = None  # Initialize output filename
        self.default_voice_id = "EXAVITQu4vr4xnSDxMaL"
        if voice_id is None:
            self.voice_id = self.default_voice_id
        else:
            self.voice_id = voice_id
        self.blocking_playback = blocking_playback

    def play(
            self,
            text,
            output_filename=None,
            stability=0.71,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True
            ):
        """
        Play a generated speech file. Instantiates SoundPlayer if it's not already.
        """
        self.generate(text, output_filename, self.voice_id, stability, similarity_boost, style, use_speaker_boost)
        if self.sound_player is None:
            self.sound_player = SoundPlayer(blocking_playback=self.blocking_playback)
        self.sound_player.play_sound(self.output_filename)

    def change_voice(self, voice_id):
        """
        Change the voice model for speech generation.

        Args:
            new_voice (str): The new voice model to be used.
        """
        self.voice_id = voice_id
        self.logger.print(f"Voice model changed to {voice_id}")    

    def stop(self):
        """
        Stop the currently playing speech file.
        """
        if self.sound_player:
            self.sound_player.stop_sound()


    def generate(
            self,
            text,
            output_filename=None,
            voice_id=None,
            stability=0.71,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True
            ):
        """
        Generate speech from text.
    
        Args:
            text (str): The text to be converted into speech. 
            output_filename (str): The filename for the output file. If None, a default filename is used.
            voice_id (str): The ID for the voice to be used. If None, a default voice ID is used.
            stability (float): Stability setting for voice generation.
            similarity_boost (float): Similarity boost setting for voice generation.
            style (float): Style setting for voice generation.
            use_speaker_boost (bool): Flag to use speaker boost in voice generation.
    
        Raises:
            ValueError: If the text source is not available.
        """
        if text is None or len(text) == 0:
            raise ValueError("text is invalid!")
    
        if voice_id is None:
            voice_id = self.default_voice_id
        
            
    
        audio = self.client.generate(
            text=text,
            voice=Voice(
                voice_id=self.voice_id,
                settings=VoiceSettings(stability=stability, similarity_boost=similarity_boost, style=style, use_speaker_boost=True)
            )
        )
    
        self.output_filename = output_filename if output_filename else "output_speech.mp3"
        save(audio, self.output_filename)
        self.logger.print(f"Generated speech saved to {self.output_filename}")





class SoundPlayer:
    def __init__(self, blocking_playback=False):
        self._play_thread = None
        self._playback_object = None
        self.blocking_playback = blocking_playback

    def _play_sound_threaded(self, sound):
        self._playback_object = simpleaudio.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        self._playback_object.wait_done()  # Wait until sound has finished playing

    def _start_playback(self, sound: AudioSegment, pan_value=0) -> None:
        # Stop any currently playing sound
        self.stop_sound()

        # play sound from left/right speaker
        if pan_value != 0 and -1 < pan_value < 1:
            sound = sound.pan(pan_value)

        # Start a new thread for playing the sound
        self._play_thread = threading.Thread(target=self._play_sound_threaded, args=(sound,))
        self._play_thread.start()
        if self.blocking_playback:
            self._play_thread.join()

    def play_sound(self, file_path, pan_value=0):
        """Play a sound file located on disk."""
        sound = AudioSegment.from_file(file_path)
        self._start_playback(sound, pan_value=pan_value)

    def play_audiosegment(self, sound: AudioSegment, pan_value=0):
        """Play an in-memory AudioSegment without saving it to disk."""
        if not isinstance(sound, AudioSegment):
            raise TypeError("sound must be an instance of pydub.AudioSegment")
        self._start_playback(sound, pan_value=pan_value)

    def stop_sound(self):
        if self._play_thread and self._play_thread.is_alive():
            if self._playback_object:
                self._playback_object.stop()
            self._play_thread.join()

class RealTimeTranscribe:
    """
    Real-time transcription using Deepgram, running inside a background thread.

    - Starts a microphone stream and Deepgram websocket in an asyncio loop scoped to a thread
    - Stores finalized utterance blocks with timestamps
    - Exposes easy accessors to retrieve concatenated transcript text or structured blocks
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-3",
        language: str = "multi",
        sample_rate: int = 16000,
        utterance_end_ms: int = 1000,
        endpointing_ms: int = 30,
        logger: LogPrint | None = None,
        auto_start: bool = False,
        ready_timeout: float | None = None,
    ) -> None:
        if not _HAS_DEEPGRAM:
            raise ImportError(
                "Deepgram SDK not installed. Please install 'deepgram-sdk' to use RealTimeTranscribe."
            )

        self.logger = logger if logger else LogPrint()
        self.api_key = api_key or read_api_key("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("No DEEPGRAM_API_KEY found (env or provided)")

        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.utterance_end_ms = str(utterance_end_ms)
        self.endpointing_ms = endpointing_ms

        # Runtime state
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._running = False

        # Deepgram objects
        self._deepgram: DeepgramClient | None = None
        self._dg_connection = None
        self._microphone: Microphone | None = None
        self._audio_queue: asyncio.Queue[bytes] | None = None
        self._audio_task: asyncio.Task | None = None
        self._need_restart = False

        # Transcript storage (thread-safe)
        self._blocks_lock = threading.Lock()
        self._blocks: list[dict] = []
        self._utterance_counter = 0

        # Internal buffering of interim finals
        self._is_finals: list[str] = []
        self._last_saved_utterance: str = ""

        # Module logger for additional detail
        self._py_logger = logging.getLogger(__name__)

        # Per-chunk event logging (interim and final) with timestamps
        self._chunk_events_lock = threading.Lock()
        self._chunk_events: list[dict] = []
        self._chunk_counter = 0

        # Readiness state (set when Deepgram connection opens)
        self._ready_event = threading.Event()
        self._ready = False

        # Optionally auto-start and wait for readiness
        if auto_start:
            self.start()
            self.wait_until_ready(timeout=ready_timeout)

    # ------------- Public API -------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, name="RealTimeTranscribeThread", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        self._need_restart = False
        if self._loop and self._stop_event:
            # Signal the asyncio loop to shutdown
            def _signal_stop() -> None:
                if not self._stop_event.is_set():
                    self._stop_event.set()
            try:
                self._loop.call_soon_threadsafe(_signal_stop)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=timeout)

    def get_text(self) -> str:
        """Return the concatenated transcript text of all finalized blocks."""
        with self._blocks_lock:
            return " ".join(block["text"] for block in self._blocks)

    def get_blocks(self) -> list[dict]:
        """Return a shallow copy of the structured blocks collected so far."""
        with self._blocks_lock:
            return list(self._blocks)

    def get_chunk_events(self) -> list[dict]:
        """Return a shallow copy of per-chunk events (interim and final)."""
        with self._chunk_events_lock:
            return list(self._chunk_events)

    def get_chunks(self, silence_duration: float = 10.0) -> list[str]:
        """
        Return utterance texts that occurred AFTER the last long silence gap.

        A "long silence" is defined as a gap between consecutive finalized
        utterances where (received_at[i] - received_at[i-1]) >= silence_duration.

        Args:
            silence_duration: Threshold in seconds to detect a long silence.

        Returns:
            List of utterance texts after the most recent long silence. If no
            such gap exists yet, returns all collected utterances.
        """
        with self._blocks_lock:
            blocks_snapshot = list(self._blocks)

        if len(blocks_snapshot) == 0:
            return []

        last_break_index = 0
        for i in range(1, len(blocks_snapshot)):
            try:
                prev_ts = datetime.fromisoformat(blocks_snapshot[i - 1]["received_at"])  # type: ignore[arg-type]
                curr_ts = datetime.fromisoformat(blocks_snapshot[i]["received_at"])      # type: ignore[arg-type]
            except Exception:
                # If timestamps are malformed, skip this comparison
                continue
            gap_seconds = (curr_ts - prev_ts).total_seconds()
            if gap_seconds >= silence_duration:
                last_break_index = i

        return [b["text"] for b in blocks_snapshot[last_break_index:]]

    def is_ready(self) -> bool:
        return self._ready

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._ready_event.wait(timeout=timeout)

    # ------------- Thread & asyncio orchestration -------------
    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_session_loop())
        finally:
            try:
                pending = asyncio.all_tasks(loop=self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    async def _run_session_loop(self) -> None:
        while self._running:
            self._stop_event = asyncio.Event()
            try:
                await self._async_main()
            except Exception as exc:  # noqa: BLE001
                self._py_logger.error(f"Deepgram session error: {exc}")
            if not self._running:
                break
            if self._need_restart:
                self._py_logger.info("Restarting Deepgram realtime session after disconnect.")
                self._need_restart = False
                await asyncio.sleep(1.0)
                continue
            break
        self._stop_event = None

    async def _async_main(self) -> None:
        self._need_restart = False
        self._ready = False
        self._ready_event.clear()

        # Configure and create Deepgram client
        config: DeepgramClientOptions = DeepgramClientOptions(options={"keepalive": "true"})
        self._deepgram = DeepgramClient(self.api_key, config)
        self._dg_connection = self._deepgram.listen.asyncwebsocket.v("1")

        # Event handlers
        async def on_open(_self, _open, **kwargs):
            self.logger.print("Deepgram connection open")
            self._ready = True
            self._ready_event.set()

        async def on_message(_self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not sentence:
                return
            # Log interim and final chunk events with arrival time
            if getattr(result, "is_final", False):
                self._log_chunk_event("final", sentence, is_speech_final=getattr(result, "speech_final", False))
                self._is_finals.append(sentence)
                if getattr(result, "speech_final", False):
                    utterance = " ".join(self._is_finals).strip()
                    if utterance and utterance != self._last_saved_utterance:
                        self._append_block(utterance)
                        self._last_saved_utterance = utterance
                    self._is_finals = []
            else:
                self._log_chunk_event("interim", sentence, is_speech_final=False)

        async def on_close(_self, _close=None, **kwargs):
            self.logger.print("Deepgram connection closed")
            self._ready = False
            self._ready_event.clear()
            if not self._running:
                return
            if self._need_restart:
                return
            code = getattr(_close, "code", None) if _close is not None else None
            reason = getattr(_close, "reason", None) if _close is not None else None
            if isinstance(_close, dict):
                code = _close.get("code", code)
                reason = _close.get("reason", reason)
            if code in (1000, 1001):
                return
            desc = f"code={code} reason={reason or _close!r}" if _close is not None else "code=unknown"
            self._request_restart(f"websocket closed unexpectedly ({desc})")

        async def on_error(_self, error, **kwargs):
            self._py_logger.error(f"Deepgram error: {error}")
            self._request_restart(f"error event: {error}")

        self._dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        self._dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        self._dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        self._dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # Connect to websocket
        options: LiveOptions = LiveOptions(
            model=self.model,
            language=self.language,
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=self.sample_rate,
            interim_results=True,
            utterance_end_ms=self.utterance_end_ms,
            vad_events=True,
            endpointing=self.endpointing_ms,
        )

        addons = {"no_delay": "true"}

        started = await self._dg_connection.start(options, addons=addons)
        if started is False:
            self.logger.print("Failed to connect to Deepgram")
            return

        # Prepare audio forwarding queue/task in the loop thread
        self._audio_queue = asyncio.Queue(maxsize=50)
        self._audio_task = asyncio.create_task(self._drain_audio_queue())

        # Start microphone capture and forward to Deepgram
        self._microphone = Microphone(self._forward_audio_to_deepgram, rate=self.sample_rate, channels=1)
        self._microphone.start()

        try:
            # Idle loop until stop is requested
            while self._running and self._stop_event and not self._stop_event.is_set():
                await asyncio.sleep(0.2)
        finally:
            try:
                self._ready = False
                self._ready_event.clear()
                if self._microphone:
                    self._microphone.finish()
                    self._microphone = None
                if self._audio_task:
                    self._audio_queue = None
                    self._audio_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._audio_task
                    self._audio_task = None
                self._audio_queue = None
                if self._dg_connection:
                    await self._dg_connection.finish()
                    self._dg_connection = None
                self._deepgram = None
            except Exception:
                pass

    def _request_restart(self, reason: str | None = None) -> None:
        if not self._running:
            return
        if reason:
            self._py_logger.warning(f"Deepgram restart requested: {reason}")
        self._need_restart = True

        def _signal_stop() -> None:
            if self._stop_event and not self._stop_event.is_set():
                self._stop_event.set()

        loop = self._loop
        if not loop or not loop.is_running():
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            _signal_stop()
        else:
            loop.call_soon_threadsafe(_signal_stop)

    def _forward_audio_to_deepgram(self, data: bytes) -> None:
        """Synchronously invoked by Microphone; schedule send on the loop thread."""
        loop = self._loop
        queue = self._audio_queue
        if not loop or not queue or loop.is_closed() or not loop.is_running():
            return
        if not data:
            return
        chunk = bytes(data)

        def _enqueue() -> None:
            if self._audio_queue is None:
                return
            try:
                self._audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Drop oldest chunk to avoid runaway backpressure
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._audio_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    self._py_logger.warning("Dropping Deepgram audio chunk: queue full")

        try:
            loop.call_soon_threadsafe(_enqueue)
        except Exception as exc:  # noqa: BLE001
            self._py_logger.error(f"Deepgram audio enqueue failed: {exc}")

    async def _drain_audio_queue(self) -> None:
        """Consume queued audio chunks and forward them over Deepgram connection."""
        idle_interval = 2.0
        silence_duration = 0.2  # seconds
        silence_frames = max(1, int(self.sample_rate * silence_duration))
        silence_bytes = b"\x00" * (silence_frames * 2)
        while self._running and self._dg_connection and self._stop_event and not self._stop_event.is_set():
            queue = self._audio_queue
            if queue is None:
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=idle_interval)
            except asyncio.TimeoutError:
                try:
                    await self._dg_connection.send(silence_bytes)
                except Exception as exc:  # noqa: BLE001
                    self._py_logger.error(f"Deepgram keepalive send failed: {exc}")
                    self._request_restart(f"keepalive failed: {exc}")
                    break
                continue

            try:
                await self._dg_connection.send(chunk)
            except Exception as exc:  # noqa: BLE001
                self._py_logger.error(f"Deepgram audio send failed: {exc}")
                self._request_restart(f"audio send failed: {exc}")
                break
            finally:
                queue.task_done()
        self._audio_queue = None

    def _append_block(self, utterance: str) -> None:
        block = {
            "index": self._utterance_counter,
            "text": utterance,
            # Wall-clock timestamps; Deepgram word/segment timings are available via words if needed
            "received_at": datetime.now().isoformat(),
            "source": "deepgram_realtime_api",
            "type": "transcription_complete",
        }
        with self._blocks_lock:
            self._blocks.append(block)
            self._utterance_counter += 1

    def _log_chunk_event(self, event_type: str, text: str, is_speech_final: bool | None = None) -> None:
        """Record a chunk-level event with arrival timestamp.

        Args:
            event_type: 'interim' or 'final'.
            text: transcript content for this event chunk.
            is_speech_final: For final chunks, indicates Deepgram speech_final; otherwise None/False.
        """
        event = {
            "index": self._chunk_counter,
            "type": event_type,
            "text": text,
            "received_at": datetime.now().isoformat(),
            "speech_final": bool(is_speech_final) if is_speech_final is not None else False,
            "source": "deepgram_realtime_api",
        }
        with self._chunk_events_lock:
            self._chunk_events.append(event)
            self._chunk_counter += 1

    
# Example usage moved to examples/voice/deepgram_realtime_transcribe_example.py
