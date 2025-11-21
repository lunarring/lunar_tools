import asyncio
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Iterable, Optional

from .webrtc_codec import EncodedMessage, decode_message, encode_message
from .webrtc_signaling import RestSignalingClient


def _load_aiortc():
    try:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "WebRTC support requires the 'aiortc' package. Install it via 'pip install aiortc'."
        ) from exc
    return RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription


class WebRTCDataChannel:
    """Minimal helper around a WebRTC data channel for numpy/JSON/text payloads."""

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
        logger: Optional[logging.Logger] = None,
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
        self._pc = None
        self._channel = None
        self._messages: "Queue[Dict[str, Any]]" = Queue()
        self._ready_event = threading.Event()
        self._channel_open_event: Optional[asyncio.Event] = None
        self._ice_gathering_timeout = max(1.0, ice_gathering_timeout)

    # Lifecycle ------------------------------------------------------
    def connect(self, timeout: Optional[float] = None) -> None:
        """Establish the WebRTC connection and wait for the data channel to open."""
        if self._loop is not None:
            if not self._ready_event.wait(timeout or self._connect_timeout):
                raise TimeoutError("Existing WebRTC data channel has not opened yet")
            return
        self._start_loop()
        wait_timeout = timeout or self._connect_timeout
        self._logger.info(
            "WebRTC connect start (role=%s session=%s channel=%s)",
            self._role,
            self._session_id,
            self._channel_label,
        )
        future = asyncio.run_coroutine_threadsafe(self._connect_once(), self._loop)
        try:
            future.result(timeout=wait_timeout)
        except Exception:
            self.close()
            raise
        self._logger.info("WebRTC data channel ready (role=%s)", self._role)

    def close(self) -> None:
        """Tear down the peer connection and background loop."""
        if self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result(timeout=5.0)
        finally:
            self._logger.info("Closing WebRTC connection for session %s", self._session_id)
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
        if not self._ready_event.is_set():
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

        if self._role == "offer":
            self._logger.info("Creating offer and local data channel")
            channel = pc.createDataChannel(self._channel_label)
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
            self._messages.put(envelope)

        # If the channel already opened before handlers were registered, trigger the open path.
        if channel.readyState == "open":
            _on_open()


__all__ = ["WebRTCDataChannel"]
