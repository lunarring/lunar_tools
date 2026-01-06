import asyncio
import logging
import threading
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, Iterable, Optional

from .webrtc_codec import EncodedMessage, decode_message, encode_message
from .webrtc_signaling import RestSignalingClient


def _load_aiortc():
    try:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "WebRTC support requires the 'aiortc' package. Install it via 'pip install aiortc' "
            "or 'pip install lunar_tools[webrtc]'."
        ) from exc
    return RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription


def _apply_opus_max_quality(pc, logger: logging.Logger) -> None:
    try:
        from aiortc import RTCRtpSender
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("Opus tuning skipped (aiortc not available): %s", exc)
        return

    try:
        transceivers = pc.getTransceivers()
    except Exception as exc:
        logger.debug("Opus tuning skipped (no transceivers): %s", exc)
        return

    max_bitrate = 510_000

    applied = 0
    for transceiver in transceivers:
        if getattr(transceiver, "kind", None) != "audio":
            continue
        codecs = None
        for attr in ("_codecs", "codecs"):
            codecs = getattr(transceiver, attr, None)
            if codecs:
                break
        if not codecs:
            try:
                capabilities = RTCRtpSender.getCapabilities("audio")
                codecs = capabilities.codecs
            except Exception as exc:
                logger.debug("Opus tuning skipped (capabilities unavailable): %s", exc)
                continue
        opus = None
        for codec in codecs:
            if codec.mimeType.lower() == "audio/opus":
                opus = codec
                break
        if opus is None:
            logger.debug("Opus tuning skipped (Opus codec not advertised).")
            continue
        params = dict(opus.parameters or {})
        params["maxaveragebitrate"] = int(max_bitrate)
        params["stereo"] = 1
        params["sprop-stereo"] = 1
        params["maxplaybackrate"] = 48_000
        params["useinbandfec"] = 0
        params["usedtx"] = 0
        params["cbr"] = 1
        opus.parameters = params
        setter = getattr(transceiver, "setCodecPreferences", None)
        if callable(setter):
            try:
                setter([opus])
                applied += 1
            except Exception as exc:
                logger.debug("Failed to set Opus codec preferences: %s", exc)
        sender = getattr(transceiver, "sender", None)
        if sender is not None:
            get_params = getattr(sender, "getParameters", None)
            set_params = getattr(sender, "setParameters", None)
            if callable(get_params) and callable(set_params):
                try:
                    enc_params = get_params()
                    if enc_params.encodings:
                        enc_params.encodings[0].maxBitrate = int(max_bitrate)
                        set_params(enc_params)
                except Exception as exc:
                    logger.debug("Failed to set Opus sender bitrate: %s", exc)
    if applied:
        logger.info("Applied Opus max-quality settings to %s audio transceiver(s).", applied)


def _patch_opus_sdp(sdp: str, logger: logging.Logger) -> str:
    lines = sdp.splitlines()
    opus_pt = None
    for line in lines:
        if line.startswith("a=rtpmap:") and "opus/48000" in line:
            try:
                opus_pt = line.split()[0].split(":", 1)[1]
            except Exception:
                opus_pt = None
            if opus_pt:
                break
    if not opus_pt:
        return sdp

    fmtp_line = (
        f"a=fmtp:{opus_pt} "
        "maxaveragebitrate=510000;stereo=1;sprop-stereo=1;maxplaybackrate=48000;"
        "useinbandfec=0;usedtx=0;cbr=1"
    )

    new_lines = []
    inserted = False
    for line in lines:
        if line.startswith(f"a=fmtp:{opus_pt}"):
            if not inserted:
                new_lines.append(fmtp_line)
                inserted = True
            continue
        new_lines.append(line)
        if not inserted and line.startswith(f"a=rtpmap:{opus_pt}"):
            new_lines.append(fmtp_line)
            inserted = True
    if not inserted:
        return sdp
    patched = "\r\n".join(new_lines)
    if not patched.endswith("\r\n"):
        patched += "\r\n"
    logger.info("Patched Opus fmtp params for payload %s.", opus_pt)
    return patched


class WebRTCDataChannel:
    """Minimal helper around a WebRTC data channel for numpy/JSON/text payloads.

    Parameters
    ----------
    max_pending_messages:
        Optional cap on the internal receive queue. Smaller values help keep latency
        low when the consumer cannot drain the queue fast enough.
    drop_oldest_on_overflow:
        When the receive queue is full, drop the oldest message (True) or the new one (False).
    """

    def __init__(
        self,
        *,
        role: str,
        session_id: str,
        signaling_url: str,
        channel_label: str = "lunar-tools",
        ice_servers: Optional[Iterable[Dict[str, Any]]] = None,
        connect_timeout: float = 30.0,
        request_timeout: float = 30.0,
        ice_gathering_timeout: float = 5.0,
        reconnect_delay: float = 2.0,
        max_pending_messages: Optional[int] = None,
        drop_oldest_on_overflow: bool = True,
        logger: Optional[logging.Logger] = None,
        ordered: bool = True,
        max_retransmits: Optional[int] = None,
        max_packet_life_time: Optional[int] = None,
    ) -> None:
        if role not in {"offer", "answer"}:
            raise ValueError("role must be 'offer' or 'answer'")
        self._role = role
        self._session_id = session_id
        self._channel_label = channel_label
        self._ice_servers = list(ice_servers) if ice_servers else []
        self._connect_timeout = max(1.0, connect_timeout)
        self._logger = logger or logging.getLogger(__name__)
        self._signaling = RestSignalingClient(
            base_url=signaling_url,
            session_id=session_id,
            role=role,
            request_timeout=request_timeout,
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._worker_future: Optional[asyncio.Future] = None
        self._pc = None
        self._channel = None
        queue_size = 0 if not max_pending_messages or max_pending_messages <= 0 else int(max_pending_messages)
        self._messages: "Queue[Dict[str, Any]]" = Queue(maxsize=queue_size)
        self._max_pending_messages = queue_size
        self._drop_oldest_on_overflow = drop_oldest_on_overflow
        self._ready_event = threading.Event()
        self._channel_open_event: Optional[asyncio.Event] = None
        self._ice_gathering_timeout = max(1.0, ice_gathering_timeout)
        self._reconnect_delay = max(0.0, reconnect_delay)
        self._stopped = threading.Event()
        self._disconnect_event: Optional[asyncio.Event] = None
        self._ordered = ordered
        self._max_retransmits = max_retransmits
        self._max_packet_life_time = max_packet_life_time

    # Lifecycle ------------------------------------------------------
    def connect(self, timeout: Optional[float] = None) -> None:
        """Ensure the WebRTC worker is running and wait for the channel to open."""
        wait_timeout = timeout or self._connect_timeout
        if self._loop is None:
            self._start_loop()
        if self._worker_future is not None and self._worker_future.done():
            self._worker_future = None
        if self._worker_future is None:
            self._stopped.clear()
            self._logger.info(
                "WebRTC connect start (role=%s session=%s channel=%s)",
                self._role,
                self._session_id,
                self._channel_label,
            )
            self._worker_future = asyncio.run_coroutine_threadsafe(self._connection_worker(), self._loop)
        if not self._ready_event.wait(wait_timeout):
            raise TimeoutError("Timed out waiting for WebRTC data channel to open")
        self._logger.info("WebRTC data channel ready (role=%s)", self._role)

    def close(self) -> None:
        """Tear down the peer connection and background loop."""
        if self._loop is None:
            return
        self._logger.info("Closing WebRTC connection for session %s", self._session_id)
        self._stopped.set()
        if self._disconnect_event is not None:
            self._loop.call_soon_threadsafe(self._disconnect_event.set)
        if self._worker_future is not None:
            self._worker_future.cancel()
            try:
                self._worker_future.result(timeout=5.0)
            except Exception:
                pass
            self._worker_future = None
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result(timeout=5.0)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._loop = None
            self._thread = None
            self._pc = None
            self._channel = None
            self._ready_event.clear()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # Public API -----------------------------------------------------
    def send(self, payload: Any, address: str = "data") -> None:
        """Send a payload over the data channel."""
        if not self._ready_event.wait(self._connect_timeout):
            raise RuntimeError("WebRTC data channel is not open")
        if self._loop is None or self._channel is None:
            raise RuntimeError("WebRTC data channel is not running")

        encoded = encode_message(address, payload)
        future = asyncio.run_coroutine_threadsafe(self._async_send(encoded), self._loop)
        future.result(timeout=self._connect_timeout)

    def receive(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Return the next decoded payload, or None if the timeout expires."""
        try:
            return self._messages.get(timeout=timeout)
        except Empty:
            return None

    # Internal helpers -----------------------------------------------
    def _start_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(target=self._run_loop, name="WebRTCDataChannel", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    async def _connection_worker(self) -> None:
        while not self._stopped.is_set():
            self._disconnect_event = asyncio.Event()
            try:
                await self._connect_once()
                await self._wait_for_disconnect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.warning("WebRTC connection loop error: %s", exc)
            finally:
                try:
                    await self._shutdown()
                except Exception:
                    self._logger.debug("WebRTC shutdown error", exc_info=True)
                self._pc = None
                self._channel = None
                self._ready_event.clear()
                self._channel_open_event = None
            if self._stopped.is_set():
                break
            if self._reconnect_delay > 0:
                try:
                    await asyncio.sleep(self._reconnect_delay)
                except asyncio.CancelledError:
                    break
        self._disconnect_event = None

    async def _wait_for_disconnect(self) -> None:
        event = self._disconnect_event
        if event is None:
            return
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
                return
            except asyncio.TimeoutError:
                continue

    async def _connect_once(self) -> None:
        RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription = _load_aiortc()
        configuration = RTCConfiguration(
            [RTCIceServer(**server) for server in self._ice_servers] if self._ice_servers else []
        )
        pc = RTCPeerConnection(configuration)
        self._pc = pc
        loop = asyncio.get_running_loop()
        self._channel_open_event = asyncio.Event()

        @pc.on("connectionstatechange")
        async def _on_connection_state_change():  # pragma: no cover - callback wiring
            self._logger.info("WebRTC connection state: %s", pc.connectionState)
            if pc.connectionState in {"failed", "closed"}:
                self._ready_event.clear()
                if self._disconnect_event is not None and not self._disconnect_event.is_set():
                    self._disconnect_event.set()

        if self._role == "offer":
            self._logger.info(
                "Creating offer and local data channel (ordered=%s, maxRetransmits=%s, maxPacketLifeTime=%s)",
                self._ordered,
                self._max_retransmits,
                self._max_packet_life_time,
            )
            channel = pc.createDataChannel(
                self._channel_label,
                ordered=self._ordered,
                maxRetransmits=self._max_retransmits,
                maxPacketLifeTime=self._max_packet_life_time,
            )
            self._setup_channel(channel, loop)
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await self._wait_for_ice_gathering(pc)
            local = pc.localDescription
            assert local is not None
            self._logger.info("Posting local offer for session %s", self._session_id)
            await self._signaling.publish_local_description({"type": local.type, "sdp": local.sdp})
            self._logger.info("Waiting for remote answer...")
            remote = await self._signaling.wait_for_remote_description(timeout=self._connect_timeout)
            await pc.setRemoteDescription(RTCSessionDescription(remote["sdp"], remote["type"]))
        else:
            channel_ready = asyncio.Event()

            @pc.on("datachannel")
            def _on_datachannel(channel):  # pragma: no cover - callback wiring
                self._setup_channel(channel, loop)
                channel_ready.set()

            self._logger.info("Waiting for remote offer for session %s", self._session_id)
            remote = await self._signaling.wait_for_remote_description(timeout=self._connect_timeout)
            await pc.setRemoteDescription(RTCSessionDescription(remote["sdp"], remote["type"]))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await self._wait_for_ice_gathering(pc)
            local = pc.localDescription
            assert local is not None
            self._logger.info("Posting local answer for session %s", self._session_id)
            await self._signaling.publish_local_description({"type": local.type, "sdp": local.sdp})
            self._logger.info("Waiting for data channel from offer peer...")
            await asyncio.wait_for(channel_ready.wait(), timeout=self._connect_timeout)

        assert self._channel_open_event is not None
        try:
            await asyncio.wait_for(self._channel_open_event.wait(), timeout=self._connect_timeout)
        except asyncio.TimeoutError as exc:
            self._logger.error("Timed out waiting for data channel to open (role=%s)", self._role)
            raise

    async def _shutdown(self) -> None:
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:  # pragma: no cover - defensive close
                pass
        if self._pc is not None:
            await self._pc.close()

    async def _async_send(self, message: EncodedMessage) -> None:
        if self._channel is None:
            raise RuntimeError("WebRTC data channel closed")
        self._logger.debug("Sending payload via WebRTC: %s bytes", len(message.data) if message.is_binary else "text")
        self._channel.send(message.data)

    async def _wait_for_ice_gathering(self, pc) -> None:
        if pc.iceGatheringState == "complete":
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ice_gathering_timeout
        while pc.iceGatheringState != "complete" and loop.time() < deadline:
            await asyncio.sleep(0.1)
 
    def _enqueue_message(self, envelope: Dict[str, Any]) -> None:
        if not self._max_pending_messages:
            self._messages.put(envelope)
            return
        try:
            self._messages.put_nowait(envelope)
            return
        except Full:
            if self._drop_oldest_on_overflow:
                try:
                    self._messages.get_nowait()
                except Empty:
                    pass
                try:
                    self._messages.put_nowait(envelope)
                    return
                except Full:
                    pass
            self._logger.warning(
                "Dropping incoming WebRTC payload: receive buffer full (max=%s)",
                self._max_pending_messages,
            )

    def _setup_channel(self, channel, loop: asyncio.AbstractEventLoop) -> None:
        self._channel = channel

        @channel.on("open")
        def _on_open():  # pragma: no cover - callback wiring
            self._logger.info("Data channel '%s' open", channel.label)
            self._ready_event.set()
            if self._channel_open_event is not None and not self._channel_open_event.is_set():
                loop.call_soon_threadsafe(self._channel_open_event.set)

        @channel.on("close")
        def _on_close():  # pragma: no cover - callback wiring
            self._logger.info("Data channel '%s' closed", channel.label)
            self._ready_event.clear()
            # signal disconnect loop
            if self._disconnect_event is not None and not self._disconnect_event.is_set():
                self._disconnect_event.set()

        @channel.on("message")
        def _on_message(message):  # pragma: no cover - callback wiring
            try:
                envelope = decode_message(message)
            except Exception as exc:
                envelope = {"address": None, "payload": message, "kind": "raw", "error": str(exc)}
            self._logger.debug(
                "Received WebRTC payload addr=%s kind=%s",
                envelope.get("address"),
                envelope.get("kind"),
            )
            self._enqueue_message(envelope)

        # If the channel already opened before handlers were registered, trigger the open path.
        if channel.readyState == "open":
            _on_open()


def _create_microphone_audio_track(
    *,
    sample_rate: int,
    channels: int,
    frame_duration: float,
    device: Optional[int | str],
    max_pending_frames: Optional[int],
    drop_oldest_on_overflow: bool,
    blocksize: Optional[int],
    log_cadence: bool,
    logger: logging.Logger,
):
    from fractions import Fraction
    from queue import Queue
    import time

    import numpy as np
    import sounddevice as sd
    from aiortc import MediaStreamTrack
    from av import AudioFrame

    samples_per_frame = max(1, int(sample_rate * frame_duration))
    effective_blocksize = int(blocksize) if blocksize and blocksize > 0 else samples_per_frame
    queue_size = 0 if not max_pending_frames or max_pending_frames <= 0 else int(max_pending_frames)

    class MicrophoneAudioStreamTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self._queue: "Queue[Optional[np.ndarray]]" = Queue(maxsize=queue_size)
            self._drop_oldest_on_overflow = drop_oldest_on_overflow
            self._timestamp = 0
            self._time_base = Fraction(1, sample_rate)
            self._closed = False
            self._cadence_last = None
            self._cadence_frames = 0
            self._cadence_samples = 0
            self._stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                blocksize=effective_blocksize,
                device=device,
                callback=self._on_audio,
            )
            self._stream.start()
            logger.info(
                "Microphone stream started (sample_rate=%s channels=%s blocksize=%s)",
                sample_rate,
                channels,
                effective_blocksize,
            )

        def _on_audio(self, indata, frames, time_info, status):
            if self._closed:
                return
            if status:
                logger.debug("Microphone stream status: %s", status)
            if log_cadence:
                now = time.monotonic()
                if self._cadence_last is None:
                    self._cadence_last = now
                self._cadence_frames += 1
                self._cadence_samples += frames
                if now - self._cadence_last >= 1.0:
                    elapsed = max(1e-6, now - self._cadence_last)
                    fps = self._cadence_frames / elapsed
                    effective_rate = self._cadence_samples / elapsed
                    logger.info(
                        "Mic cadence: frames=%s samples=%s fps=%.2f rate=%.1f",
                        self._cadence_frames,
                        self._cadence_samples,
                        fps,
                        effective_rate,
                    )
                    self._cadence_last = now
                    self._cadence_frames = 0
                    self._cadence_samples = 0
            frame = np.copy(indata)
            if frame.ndim == 1:
                frame = frame.reshape(1, -1)
            else:
                frame = frame.T
            try:
                self._queue.put_nowait(frame)
            except Full:
                if self._drop_oldest_on_overflow:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                        return
                    except Full:
                        pass
                logger.warning("Dropping microphone frame: buffer full (max=%s)", queue_size)

        async def recv(self):
            if self.readyState != "live":
                raise asyncio.CancelledError
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(None, self._queue.get)
            if frame is None:
                raise asyncio.CancelledError
            if frame.ndim == 1:
                frame = frame.reshape(1, -1)
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
            samples = frame.shape[1]
            layout = "mono" if channels == 1 else "stereo"
            audio_frame = AudioFrame(format="s16", layout=layout, samples=samples)
            audio_frame.planes[0].update(frame.T.reshape(-1).tobytes())
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = self._timestamp
            audio_frame.time_base = self._time_base
            self._timestamp += samples
            return audio_frame

        def stop(self):
            if self._closed:
                return
            self._closed = True
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.debug("Microphone stream close error", exc_info=True)
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
            super().stop()

    return MicrophoneAudioStreamTrack()


def _create_sine_audio_track(
    *,
    sample_rate: int,
    channels: int,
    frame_duration: float,
    frequency: float,
    amplitude: float,
    logger: logging.Logger,
):
    from fractions import Fraction

    import numpy as np
    import time
    from aiortc import MediaStreamTrack
    from av import AudioFrame

    samples_per_frame = max(1, int(sample_rate * frame_duration))
    phase_step = 2.0 * np.pi * frequency / sample_rate
    amplitude = float(max(0.0, min(1.0, amplitude)))

    class SineAudioStreamTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self._timestamp = 0
            self._time_base = Fraction(1, sample_rate)
            self._phase = 0.0
            self._frame_count = 0
            self._cadence_last = time.monotonic()
            self._cadence_frames = 0
            self._cadence_samples = 0
            self._start_time = self._cadence_last
            self._frame_index = 0

        async def recv(self):
            if self.readyState != "live":
                raise asyncio.CancelledError
            t = self._phase + phase_step * np.arange(samples_per_frame, dtype=np.float32)
            tone = np.sin(t) * amplitude
            self._phase = float(t[-1] + phase_step)
            data = np.clip(tone * 32767.0, -32768, 32767).astype(np.int16)
            if channels > 1:
                data = np.repeat(data[None, :], channels, axis=0)
            else:
                data = data.reshape(1, -1)
            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)
            samples = data.shape[1]
            layout = "mono" if channels == 1 else "stereo"
            audio_frame = AudioFrame(format="s16", layout=layout, samples=samples)
            audio_frame.planes[0].update(data.T.reshape(-1).tobytes())
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = self._timestamp
            audio_frame.time_base = self._time_base
            self._timestamp += samples_per_frame
            self._frame_count += 1
            self._cadence_frames += 1
            self._cadence_samples += samples_per_frame
            self._frame_index += 1
            now = time.monotonic()
            if now - self._cadence_last >= 1.0:
                elapsed = max(1e-6, now - self._cadence_last)
                fps = self._cadence_frames / elapsed
                effective_rate = self._cadence_samples / elapsed
                logger.info(
                    "Tone cadence: frames=%s samples=%s fps=%.2f rate=%.1f total_frames=%s",
                    self._cadence_frames,
                    self._cadence_samples,
                    fps,
                    effective_rate,
                    self._frame_count,
                )
                self._cadence_last = now
                self._cadence_frames = 0
                self._cadence_samples = 0
            target_time = self._start_time + (self._frame_index * frame_duration)
            sleep_for = target_time - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            return audio_frame

    return SineAudioStreamTrack()


class WebRTCAudioPeer:
    """WebRTC helper for streaming microphone audio tracks."""

    def __init__(
        self,
        *,
        role: str,
        session_id: str,
        signaling_url: str,
        ice_servers: Optional[Iterable[Dict[str, Any]]] = None,
        connect_timeout: float = 30.0,
        request_timeout: float = 30.0,
        ice_gathering_timeout: float = 5.0,
        reconnect_delay: float = 2.0,
        logger: Optional[logging.Logger] = None,
        send_audio: bool = True,
        receive_audio: bool = True,
        audio_source: str = "mic",
        sample_rate: int = 48000,
        channels: int = 1,
        frame_duration: float = 0.02,
        audio_device: Optional[int | str] = None,
        max_pending_frames: Optional[int] = None,
        drop_oldest_on_overflow: bool = True,
        tone_frequency: float = 440.0,
        tone_amplitude: float = 0.2,
        mic_blocksize: Optional[int] = None,
        mic_log_cadence: bool = False,
    ) -> None:
        if role not in {"offer", "answer"}:
            raise ValueError("role must be 'offer' or 'answer'")
        self._role = role
        self._session_id = session_id
        self._ice_servers = list(ice_servers) if ice_servers else []
        self._connect_timeout = max(1.0, connect_timeout)
        self._logger = logger or logging.getLogger(__name__)
        self._signaling = RestSignalingClient(
            base_url=signaling_url,
            session_id=session_id,
            role=role,
            request_timeout=request_timeout,
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._worker_future: Optional[asyncio.Future] = None
        self._pc = None
        self._ready_event = threading.Event()
        self._disconnect_event: Optional[asyncio.Event] = None
        self._ice_gathering_timeout = max(1.0, ice_gathering_timeout)
        self._reconnect_delay = max(0.0, reconnect_delay)
        self._stopped = threading.Event()
        self._send_audio = send_audio
        self._receive_audio = receive_audio
        self._audio_source = audio_source
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_duration = frame_duration
        self._audio_device = audio_device
        self._max_pending_frames = max_pending_frames
        self._drop_oldest_on_overflow = drop_oldest_on_overflow
        self._tone_frequency = tone_frequency
        self._tone_amplitude = tone_amplitude
        self._mic_blocksize = mic_blocksize
        self._mic_log_cadence = mic_log_cadence
        self._local_audio_track = None
        self._remote_audio_track = None
        self._audio_track_event = threading.Event()
        self._audio_track_async_event: Optional[asyncio.Event] = None
        self._playback_enabled = False
        self._playback_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._monitor_on_frame: Optional[Callable[[Any], None]] = None
        self._monitor_raw_frames = False
        self._monitor_interval = 1.0

    def connect(self, timeout: Optional[float] = None) -> None:
        wait_timeout = timeout or self._connect_timeout
        if self._loop is None:
            self._start_loop()
        if self._worker_future is not None and self._worker_future.done():
            self._worker_future = None
        if self._worker_future is None:
            self._stopped.clear()
            self._logger.info("WebRTC audio connect start (role=%s session=%s)", self._role, self._session_id)
            self._worker_future = asyncio.run_coroutine_threadsafe(self._connection_worker(), self._loop)
        if not self._ready_event.wait(wait_timeout):
            raise TimeoutError("Timed out waiting for WebRTC audio connection to be ready")
        if self._playback_enabled:
            self._schedule_playback()
        self._logger.info("WebRTC audio ready (role=%s)", self._role)

    def close(self) -> None:
        if self._loop is None:
            return
        self._logger.info("Closing WebRTC audio connection for session %s", self._session_id)
        self._stopped.set()
        if self._disconnect_event is not None:
            self._loop.call_soon_threadsafe(self._disconnect_event.set)
        if self._worker_future is not None:
            self._worker_future.cancel()
            try:
                self._worker_future.result(timeout=5.0)
            except Exception:
                pass
            self._worker_future = None
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result(timeout=5.0)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._loop = None
            self._thread = None
            self._pc = None
            self._ready_event.clear()
            self._remote_audio_track = None
            self._audio_track_event.clear()
            self._audio_track_async_event = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def wait_for_remote_audio(self, timeout: Optional[float] = None):
        if not self._audio_track_event.wait(timeout=timeout):
            return None
        return self._remote_audio_track

    def get_connection_state(self) -> Optional[str]:
        if self._pc is None:
            return None
        return getattr(self._pc, "connectionState", None)

    def start_playback(self) -> None:
        self._playback_enabled = True
        self._schedule_playback()

    def start_audio_monitor(
        self,
        *,
        on_stats: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_frame: Optional[Callable[[Any], None]] = None,
        raw_frames: bool = False,
        interval: float = 1.0,
    ) -> None:
        self._monitor_callback = on_stats
        self._monitor_on_frame = on_frame
        self._monitor_raw_frames = raw_frames
        self._monitor_interval = max(0.2, interval)
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._ensure_monitor_task(), self._loop)

    def _schedule_playback(self) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._ensure_playback_task(), self._loop)

    def _start_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(target=self._run_loop, name="WebRTCAudioPeer", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    async def _connection_worker(self) -> None:
        while not self._stopped.is_set():
            self._disconnect_event = asyncio.Event()
            try:
                await self._connect_once()
                await self._wait_for_disconnect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.warning("WebRTC audio connection loop error: %s", exc)
            finally:
                try:
                    await self._shutdown()
                except Exception:
                    self._logger.debug("WebRTC audio shutdown error", exc_info=True)
                self._pc = None
                self._ready_event.clear()
                self._remote_audio_track = None
                self._audio_track_event.clear()
                self._audio_track_async_event = None
            if self._stopped.is_set():
                break
            if self._reconnect_delay > 0:
                try:
                    await asyncio.sleep(self._reconnect_delay)
                except asyncio.CancelledError:
                    break
        self._disconnect_event = None

    async def _wait_for_disconnect(self) -> None:
        event = self._disconnect_event
        if event is None:
            return
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
                return
            except asyncio.TimeoutError:
                continue

    async def _connect_once(self) -> None:
        RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription = _load_aiortc()
        configuration = RTCConfiguration(
            [RTCIceServer(**server) for server in self._ice_servers] if self._ice_servers else []
        )
        pc = RTCPeerConnection(configuration)
        self._pc = pc
        loop = asyncio.get_running_loop()
        self._audio_track_async_event = asyncio.Event()

        @pc.on("connectionstatechange")
        async def _on_connection_state_change():  # pragma: no cover - callback wiring
            self._logger.info("WebRTC audio connection state: %s", pc.connectionState)
            if pc.connectionState in {"connected", "completed"}:
                self._ready_event.set()
            if pc.connectionState in {"failed", "closed"}:
                self._ready_event.clear()
                if self._disconnect_event is not None and not self._disconnect_event.is_set():
                    self._disconnect_event.set()

        if self._receive_audio:
            @pc.on("track")
            def _on_track(track):  # pragma: no cover - callback wiring
                if track.kind != "audio":
                    return
                self._remote_audio_track = track
                self._audio_track_event.set()
                if self._audio_track_async_event is not None and not self._audio_track_async_event.is_set():
                    loop.call_soon_threadsafe(self._audio_track_async_event.set)

        if self._send_audio:
            if self._audio_source == "tone":
                self._local_audio_track = _create_sine_audio_track(
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    frame_duration=self._frame_duration,
                    frequency=self._tone_frequency,
                    amplitude=self._tone_amplitude,
                    logger=self._logger,
                )
            else:
                self._local_audio_track = _create_microphone_audio_track(
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    frame_duration=self._frame_duration,
                    device=self._audio_device,
                    max_pending_frames=self._max_pending_frames,
                    drop_oldest_on_overflow=self._drop_oldest_on_overflow,
                    blocksize=self._mic_blocksize,
                    log_cadence=self._mic_log_cadence,
                    logger=self._logger,
                )
            pc.addTrack(self._local_audio_track)
            _apply_opus_max_quality(pc, self._logger)

        if self._role == "offer":
            offer = await pc.createOffer()
            offer_sdp = _patch_opus_sdp(offer.sdp, self._logger)
            if offer_sdp != offer.sdp:
                offer = RTCSessionDescription(offer_sdp, offer.type)
            await pc.setLocalDescription(offer)
            await self._wait_for_ice_gathering(pc)
            local = pc.localDescription
            assert local is not None
            self._logger.info("Posting local offer for session %s", self._session_id)
            await self._signaling.publish_local_description({"type": local.type, "sdp": local.sdp})
            self._logger.info("Waiting for remote answer...")
            remote = await self._signaling.wait_for_remote_description(timeout=self._connect_timeout)
            await pc.setRemoteDescription(RTCSessionDescription(remote["sdp"], remote["type"]))
        else:
            self._logger.info("Waiting for remote offer for session %s", self._session_id)
            remote = await self._signaling.wait_for_remote_description(timeout=self._connect_timeout)
            await pc.setRemoteDescription(RTCSessionDescription(remote["sdp"], remote["type"]))
            _apply_opus_max_quality(pc, self._logger)
            answer = await pc.createAnswer()
            answer_sdp = _patch_opus_sdp(answer.sdp, self._logger)
            if answer_sdp != answer.sdp:
                answer = RTCSessionDescription(answer_sdp, answer.type)
            await pc.setLocalDescription(answer)
            await self._wait_for_ice_gathering(pc)
            local = pc.localDescription
            assert local is not None
            self._logger.info("Posting local answer for session %s", self._session_id)
            await self._signaling.publish_local_description({"type": local.type, "sdp": local.sdp})

    async def _shutdown(self) -> None:
        if self._local_audio_track is not None:
            try:
                self._local_audio_track.stop()
            except Exception:  # pragma: no cover - defensive close
                pass
            self._local_audio_track = None
        if self._pc is not None:
            await self._pc.close()

    async def _wait_for_ice_gathering(self, pc) -> None:
        if pc.iceGatheringState == "complete":
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ice_gathering_timeout
        while pc.iceGatheringState != "complete" and loop.time() < deadline:
            await asyncio.sleep(0.1)

    async def _ensure_playback_task(self) -> None:
        if not self._receive_audio:
            return
        if self._playback_task is not None and not self._playback_task.done():
            return
        self._playback_task = asyncio.create_task(self._playback_loop())

    async def _ensure_monitor_task(self) -> None:
        if not self._receive_audio:
            return
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _playback_loop(self) -> None:
        try:
            await self._run_playback_loop()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._logger.warning("Audio playback error: %s", exc)

    async def _run_playback_loop(self) -> None:
        import numpy as np
        import sounddevice as sd

        logger = self._logger

        class PlaybackBuffer:
            def __init__(self, sample_rate: int, channels: int, blocksize: int):
                self._queue = []
                self._lock = threading.Lock()
                self._stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="int16",
                    blocksize=blocksize,
                    callback=self._callback,
                )
                self._stream.start()

            def _callback(self, outdata, frames, time_info, status):
                if status:
                    logger.debug("Playback stream status: %s", status)
                with self._lock:
                    data = np.empty((0, outdata.shape[1]), dtype=np.int16)
                    while data.shape[0] < frames and self._queue:
                        chunk = self._queue.pop(0)
                        needed = frames - data.shape[0]
                        data = np.vstack([data, chunk[:needed]])
                        if chunk.shape[0] > needed:
                            self._queue.insert(0, chunk[needed:])
                    if data.shape[0] < frames:
                        pad = np.zeros((frames - data.shape[0], outdata.shape[1]), dtype=np.int16)
                        data = np.vstack([data, pad])
                outdata[:] = data

            def enqueue(self, data: np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                with self._lock:
                    self._queue.append(data)

            def close(self):
                self._stream.stop()
                self._stream.close()

        event = self._audio_track_async_event
        if event is not None and not event.is_set():
            await event.wait()
        track = self._remote_audio_track
        if track is None:
            return
        playback = None
        try:
            while not self._stopped.is_set() and track.readyState == "live":
                try:
                    frame = await track.recv()
                except Exception:
                    break
                data = frame.to_ndarray()
                frame_channels = None
                try:
                    frame_channels = int(frame.layout.channels)
                except Exception:
                    frame_channels = None
                expected_channels = int(getattr(self, "_channels", 0) or 0)
                target_channels = frame_channels or expected_channels or 1
                if data.ndim == 1:
                    if target_channels > 1 and data.size % target_channels == 0:
                        data = data.reshape(-1, target_channels)
                    else:
                        data = data.reshape(-1, 1)
                elif data.ndim == 2:
                    if 1 in data.shape and target_channels > 1 and data.size % target_channels == 0:
                        data = data.reshape(-1, target_channels)
                    elif target_channels and data.shape[0] == target_channels and data.shape[1] != target_channels:
                        data = data.T
                    elif target_channels and data.shape[1] != target_channels and data.shape[0] != target_channels and data.size % target_channels == 0:
                        data = data.reshape(-1, target_channels)
                    elif data.shape[1] > data.shape[0]:
                        data = data.T
                elif data.ndim > 2:
                    if target_channels > 1 and data.size % target_channels == 0:
                        data = data.reshape(-1, target_channels)
                    else:
                        data = data.reshape(-1, 1)
                if data.dtype.kind == "f":
                    data = np.clip(data, -1.0, 1.0)
                    data = (data * 32767.0).astype(np.int16)
                elif data.dtype != np.int16:
                    data = data.astype(np.int16, copy=False)
                if playback is None:
                    sample_rate = frame.sample_rate or self._sample_rate
                    playback = PlaybackBuffer(sample_rate, data.shape[1], data.shape[0])
                playback.enqueue(data)
        finally:
            if playback is not None:
                playback.close()

    async def _monitor_loop(self) -> None:
        import time
        import numpy as np

        event = self._audio_track_async_event
        if event is not None and not event.is_set():
            await event.wait()
        track = self._remote_audio_track
        if track is None:
            return
        total_samples = 0
        total_frames = 0
        last_report = time.monotonic()
        last_samples = 0
        self._logger.debug("Audio monitor loop started (readyState=%s).", track.readyState)
        try:
            while not self._stopped.is_set() and track.readyState == "live":
                try:
                    frame = await track.recv()
                except Exception as exc:
                    self._logger.warning("Audio monitor recv failed: %s", exc, exc_info=True)
                    break
                frame_channels = None
                try:
                    frame_channels = int(frame.layout.channels)
                except Exception:
                    frame_channels = None
                expected_channels = int(getattr(self, "_channels", 0) or 0)
                target_channels = frame_channels or expected_channels or 1
                samples = int(getattr(frame, "samples", 0) or 0)
                data = None
                if not self._monitor_raw_frames:
                    data = frame.to_ndarray()
                    if data.ndim == 1:
                        if target_channels > 1 and data.size == samples * target_channels:
                            data = data.reshape(samples, target_channels)
                        elif target_channels > 1 and data.size % target_channels == 0:
                            data = data.reshape(-1, target_channels)
                        else:
                            data = data.reshape(-1, 1)
                    elif data.ndim == 2:
                        if data.shape == (target_channels, samples):
                            data = data.T
                        elif data.shape == (samples, target_channels):
                            pass
                        elif target_channels > 1 and data.size == samples * target_channels:
                            data = data.reshape(samples, target_channels)
                        elif data.shape[1] > data.shape[0]:
                            data = data.T
                    else:
                        if target_channels > 1 and data.size == samples * target_channels:
                            data = data.reshape(samples, target_channels)
                        elif target_channels > 1 and data.size % target_channels == 0:
                            data = data.reshape(-1, target_channels)
                        else:
                            data = data.reshape(-1, 1)
                if self._monitor_on_frame is not None:
                    try:
                        if self._monitor_raw_frames:
                            self._monitor_on_frame(frame)
                        else:
                            self._monitor_on_frame(data)
                    except Exception as exc:
                        self._logger.warning("Audio monitor frame callback error: %s", exc, exc_info=True)
                if samples <= 0:
                    if data is not None:
                        samples = data.shape[0]
                total_samples += samples
                total_frames += 1
                now = time.monotonic()
                if now - last_report >= self._monitor_interval:
                    elapsed = now - last_report
                    sample_rate = frame.sample_rate or self._sample_rate
                    rms = 0.0
                    if data is not None and data.size:
                        rms = float(np.sqrt(np.mean(np.square(data.astype(np.float32)))))
                    stats = {
                        "frames": total_frames,
                        "samples": total_samples,
                        "recent_samples": total_samples - last_samples,
                        "sample_rate": sample_rate,
                        "rms": rms,
                        "elapsed": elapsed,
                    }
                    if self._monitor_callback is not None:
                        try:
                            self._monitor_callback(stats)
                        except Exception as exc:
                            self._logger.warning("Audio monitor stats callback error: %s", exc, exc_info=True)
                    else:
                        self._logger.info(
                            "Audio stats: frames=%s samples=%s rate=%s rms=%.1f",
                            total_frames,
                            total_samples,
                            sample_rate,
                            rms,
                        )
                    last_report = now
                    last_samples = total_samples
        except asyncio.CancelledError:
            return


__all__ = ["WebRTCDataChannel", "WebRTCAudioPeer"]
