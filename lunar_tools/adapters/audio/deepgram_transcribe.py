from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger

try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
        Microphone,
    )

    _HAS_DEEPGRAM = True
except Exception:  # pragma: no cover - optional dependency
    DeepgramClient = None  # type: ignore
    DeepgramClientOptions = None  # type: ignore
    LiveTranscriptionEvents = None  # type: ignore
    LiveOptions = None  # type: ignore
    Microphone = None  # type: ignore
    _HAS_DEEPGRAM = False


class RealTimeTranscribe:
    """
    Real-time transcription using Deepgram, running inside a background thread.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-3",
        language: str = "multi",
        sample_rate: int = 16000,
        utterance_end_ms: int = 1000,
        endpointing_ms: int = 30,
        logger: Optional[logging.Logger] = None,
        auto_start: bool = False,
        ready_timeout: Optional[float] = None,
    ) -> None:
        if not _HAS_DEEPGRAM:
            raise ImportError(
                "Deepgram SDK not installed. Install 'deepgram-sdk' to use RealTimeTranscribe."
            )

        self.logger = logger if logger else create_logger(__name__)
        self.api_key = api_key or read_api_key("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("No DEEPGRAM_API_KEY found (env or provided)")

        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.utterance_end_ms = str(utterance_end_ms)
        self.endpointing_ms = endpointing_ms

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._running = False

        self._deepgram = None
        self._dg_connection = None
        self._microphone: Optional[Microphone] = None

        self._blocks_lock = threading.Lock()
        self._blocks: list[dict] = []
        self._utterance_counter = 0

        self._is_finals: list[str] = []
        self._last_saved_utterance: str = ""

        self._py_logger = logging.getLogger(__name__)
        self._chunk_events_lock = threading.Lock()
        self._chunk_events: list[dict] = []
        self._chunk_counter = 0

        self._ready_event = threading.Event()
        self._ready = False

        if auto_start:
            self.start()
            self.wait_until_ready(timeout=ready_timeout)

    # Public API
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, name="RealTimeTranscribeThread", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop and self._stop_event:
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
        with self._blocks_lock:
            return " ".join(block["text"] for block in self._blocks)

    def get_blocks(self) -> list[dict]:
        with self._blocks_lock:
            return list(self._blocks)

    def get_chunk_events(self) -> list[dict]:
        with self._chunk_events_lock:
            return list(self._chunk_events)

    def get_chunks(self, silence_duration: float = 10.0) -> list[str]:
        with self._blocks_lock:
            blocks_snapshot = list(self._blocks)

        if not blocks_snapshot:
            return []

        last_break_index = 0
        for i in range(1, len(blocks_snapshot)):
            try:
                prev_ts = datetime.fromisoformat(blocks_snapshot[i - 1]["received_at"])
                curr_ts = datetime.fromisoformat(blocks_snapshot[i]["received_at"])
            except Exception:
                continue
            gap_seconds = (curr_ts - prev_ts).total_seconds()
            if gap_seconds >= silence_duration:
                last_break_index = i

        return [b["text"] for b in blocks_snapshot[last_break_index:]]

    def is_ready(self) -> bool:
        return self._ready

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready_event.wait(timeout=timeout)

    # Internal orchestration
    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._async_main())
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

    async def _async_main(self) -> None:
        config: DeepgramClientOptions = DeepgramClientOptions(options={"keepalive": "true"})
        self._deepgram = DeepgramClient(self.api_key, config)
        self._dg_connection = self._deepgram.listen.asyncwebsocket.v("1")

        async def on_open(_self, _open, **kwargs):
            self.logger.info("Deepgram connection open")
            self._ready = True
            self._ready_event.set()

        async def on_message(_self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not sentence:
                return
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

        async def on_close(_self, _close, **kwargs):
            self.logger.info("Deepgram connection closed")
            self._ready = False
            self._ready_event.clear()

        async def on_error(_self, error, **kwargs):
            self._py_logger.error("Deepgram error: %s", error)

        self._dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        self._dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        self._dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        self._dg_connection.on(LiveTranscriptionEvents.Error, on_error)

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
            self.logger.error("Failed to connect to Deepgram")
            return

        self._microphone = Microphone(self._dg_connection.send)
        self._microphone.start()

        try:
            while self._running and self._stop_event and not self._stop_event.is_set():
                await asyncio.sleep(0.2)
        finally:
            try:
                if self._microphone:
                    self._microphone.finish()
                if self._dg_connection:
                    await self._dg_connection.finish()
            except Exception:
                pass

    def _append_block(self, utterance: str) -> None:
        block = {
            "index": self._utterance_counter,
            "text": utterance,
            "received_at": datetime.now().isoformat(),
            "source": "deepgram_realtime_api",
            "type": "transcription_complete",
        }
        with self._blocks_lock:
            self._blocks.append(block)
            self._utterance_counter += 1

    def _log_chunk_event(self, event_type: str, text: str, is_speech_final: Optional[bool] = None) -> None:
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

