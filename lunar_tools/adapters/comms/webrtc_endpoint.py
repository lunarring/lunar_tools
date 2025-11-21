from __future__ import annotations

import asyncio
from concurrent.futures import Future
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

from lunar_tools._optional import require_extra
from lunar_tools.adapters.comms.webrtc_codec import EncodedMessage, decode_message, encode_message
from lunar_tools.platform.logging import create_logger

try:  # pragma: no cover - optional dependency for typing only
    from typing import TypedDict
except ImportError:  # pragma: no cover - python fallback
    TypedDict = dict  # type: ignore


if TYPE_CHECKING:  # pragma: no cover
    from aiortc import RTCDataChannel, RTCPeerConnection, RTCSessionDescription


def _load_aiortc():
    try:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    except ImportError:  # pragma: no cover - optional dependency
        require_extra("WebRTC data channel", extras="comms")
    return RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription


class SignalingDescription(TypedDict):
    type: str
    sdp: str


class WebRTCSignalingClient(Protocol):
    async def publish_local_description(self, description: SignalingDescription) -> None:
        ...

    async def wait_for_remote_description(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> SignalingDescription:
        ...


@dataclass
class WebRTCIceServer:
    urls: list[str]
    username: Optional[str] = None
    credential: Optional[str] = None


class WebRTCDataChannelEndpoint:
    """Adapter that exposes a WebRTC data channel via the message bus contracts with auto-reconnect."""

    def __init__(
        self,
        *,
        role: str,
        signaling: WebRTCSignalingClient,
        channel_label: str = "lunar-data",
        ice_servers: Optional[list[WebRTCIceServer | Dict[str, Any]]] = None,
        connect_timeout: float = 30.0,
        send_timeout: float = 10.0,
        reconnect_delay: float = 2.0,
        logger=None,
    ) -> None:
        if role not in {"offer", "answer"}:
            raise ValueError("role must be 'offer' or 'answer'")
        self._role = role
        self._signaling = signaling
        self._channel_label = channel_label
        self._connect_timeout = connect_timeout
        self._send_timeout = send_timeout
        self._reconnect_delay = max(reconnect_delay, 0.0)
        self._ice_servers = list(ice_servers) if ice_servers is not None else []
        self._logger = logger if logger else create_logger(__name__ + ".webrtc")

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._worker_future: Optional[Future] = None
        self._pc: Optional["RTCPeerConnection"] = None
        self._channel: Optional["RTCDataChannel"] = None

        self._messages: list[Dict[str, Any]] = []
        self._condition = threading.Condition()
        self._ready_event = threading.Event()
        self._stopped = threading.Event()
        self._disconnect_event: Optional[asyncio.Event] = None

    # Lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._loop is not None:
            return
        self._stopped.clear()
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(target=self._run_loop, name="WebRTCDataChannel", daemon=True)
        self._thread.start()
        self._worker_future = asyncio.run_coroutine_threadsafe(self._connection_worker(), loop)

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    def stop(self) -> None:
        if self._loop is None:
            return
        self._stopped.set()
        if self._disconnect_event is not None:
            self._loop.call_soon_threadsafe(self._disconnect_event.set)
        if self._worker_future is not None:
            self._worker_future.cancel()
            try:
                self._worker_future.result(timeout=5.0)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.debug("Error while waiting for WebRTC worker to stop: %s", exc)
            self._worker_future = None
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result(timeout=5.0)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Error while shutting down WebRTC endpoint: %s", exc)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._loop = None
        self._thread = None
        self._pc = None
        self._channel = None
        self._ready_event.clear()
        self._disconnect_event = None
        with self._condition:
            self._messages.clear()

    async def _shutdown(self) -> None:
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:  # pragma: no cover - defensive close
                pass
        if self._pc is not None:
            await self._pc.close()

    async def _connection_worker(self) -> None:
        while not self._stopped.is_set():
            self._disconnect_event = asyncio.Event()
            try:
                await self._bootstrap()
                await self._wait_for_disconnect()
            except asyncio.CancelledError:  # pragma: no cover - shutdown path
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("WebRTC connection loop error: %s", exc)
            finally:
                try:
                    await self._shutdown()
                except Exception:  # pragma: no cover - defensive close
                    self._logger.debug("WebRTC shutdown during reconnect failed", exc_info=True)
                self._pc = None
                self._channel = None
                self._ready_event.clear()
            if self._stopped.is_set():
                break
            if self._reconnect_delay > 0:
                try:
                    await asyncio.sleep(self._reconnect_delay)
                except asyncio.CancelledError:  # pragma: no cover - shutdown path
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
    # Bootstrap ------------------------------------------------------
    async def _bootstrap(self) -> None:
        RTCConfiguration, RTCIceServer, RTCPeerConnection, _RTCSessionDescription = _load_aiortc()
        configuration = RTCConfiguration(
            [
                RTCIceServer(**server.__dict__) if isinstance(server, WebRTCIceServer) else RTCIceServer(**server)
                for server in self._ice_servers
            ]
        )
        pc = RTCPeerConnection(configuration)
        self._pc = pc

        @pc.on("connectionstatechange")
        def _on_state_change() -> None:  # noqa: WPS430 - nested for aiortc callback style
            self._on_connection_state_change()

        @pc.on("iceconnectionstatechange")
        def _on_ice_state_change() -> None:  # noqa: WPS430
            state = pc.iceConnectionState
            self._logger.info("WebRTC ICE state: %s", state)

        @pc.on("icecandidate")
        def _on_ice_candidate(candidate) -> None:  # noqa: WPS430
            if candidate is None:
                self._logger.debug("Local ICE gathering complete (null candidate)")
            else:
                self._logger.debug("Local ICE candidate: %s", candidate)

        gathering_complete = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def _on_ice_gathering_state_change() -> None:  # noqa: WPS430
            if pc.iceGatheringState == "complete":
                gathering_complete.set()

        async def _await_ice_gathering_complete() -> None:
            if pc.iceGatheringState == "complete":
                return
            try:
                await asyncio.wait_for(gathering_complete.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._logger.debug("ICE gathering did not reach 'complete' before timeout")

        if self._role == "offer":
            channel = pc.createDataChannel(self._channel_label)
            self._configure_channel(channel)
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await _await_ice_gathering_complete()
            local_description = _description_to_dict(pc.localDescription)
            self._logger.info("Publishing offer SDP snippet: %s", local_description.get("sdp", "")[:200])
            await self._signaling.publish_local_description(local_description)

            remote: Optional[SignalingDescription] = None
            while not self._stopped.is_set() and remote is None:
                try:
                    remote = await self._signaling.wait_for_remote_description(timeout=self._connect_timeout)
                except TimeoutError:
                    self._logger.info("Waiting for remote WebRTC answer...")
            if remote is None:
                self._logger.info("Stopping WebRTC endpoint before remote answer arrived")
                return
            self._logger.info("Applying remote answer SDP snippet: %s", remote.get("sdp", "")[:200])
            await pc.setRemoteDescription(_dict_to_description(remote))
        else:

            @pc.on("datachannel")
            def _on_datachannel(channel):  # noqa: WPS430
                self._on_datachannel(channel)
            offer: Optional[SignalingDescription] = None
            while not self._stopped.is_set() and offer is None:
                try:
                    offer = await self._signaling.wait_for_remote_description(timeout=self._connect_timeout)
                except TimeoutError:
                    self._logger.info("Waiting for remote WebRTC offer...")
            if offer is None:
                self._logger.info("Stopping WebRTC endpoint before remote offer arrived")
                return
            self._logger.info("Applying remote offer SDP snippet: %s", offer.get("sdp", "")[:200])
            await pc.setRemoteDescription(_dict_to_description(offer))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await _await_ice_gathering_complete()
            local_description = _description_to_dict(pc.localDescription)
            self._logger.info("Publishing answer SDP snippet: %s", local_description.get("sdp", "")[:200])
            await self._signaling.publish_local_description(local_description)

    # Data channel handlers ------------------------------------------
    def _configure_channel(self, channel: "RTCDataChannel") -> None:
        self._channel = channel

        @channel.on("open")
        def _on_open() -> None:  # noqa: WPS430 - nested for aiortc callback style
            self._logger.info("WebRTC data channel '%s' open", channel.label)
            self._ready_event.set()

        @channel.on("close")
        def _on_close() -> None:  # noqa: WPS430
            self._logger.info("WebRTC data channel '%s' closed", channel.label)
            self._ready_event.clear()
            event = self._disconnect_event
            if event is not None and not event.is_set():
                event.set()

        @channel.on("message")
        def _on_message(message: Any) -> None:  # noqa: WPS430
            try:
                envelope = decode_message(message)
            except Exception as exc:  # pragma: no cover - logging side effect
                self._logger.error("Failed to decode WebRTC payload: %s", exc)
                return
            with self._condition:
                self._messages.append(envelope)
                self._condition.notify_all()

    def _on_datachannel(self, channel: "RTCDataChannel") -> None:
        self._logger.info("Received remote data channel '%s'", channel.label)
        self._configure_channel(channel)

    def _on_connection_state_change(self) -> None:
        state = self._pc.connectionState if self._pc else "unknown"
        self._logger.info("WebRTC connection state: %s", state)
        if state in {"failed", "closed", "disconnected"}:
            self._ready_event.clear()
            event = self._disconnect_event
            if event is not None and not event.is_set():
                event.set()

    # Message bus API ------------------------------------------------
    def send(self, address: str, payload: Any) -> None:
        if self._loop is None:
            raise RuntimeError("WebRTC endpoint is not started")
        if not self._ready_event.wait(timeout=self._connect_timeout):
            raise TimeoutError("WebRTC data channel is not open yet")
        encoded = encode_message(address, payload)
        future = asyncio.run_coroutine_threadsafe(self._send_encoded(encoded), self._loop)
        future.result(timeout=self._send_timeout)

    async def _send_encoded(self, encoded: EncodedMessage) -> None:
        if self._channel is None:
            raise RuntimeError("WebRTC data channel is unavailable")
        self._channel.send(encoded.data)

    def receive(
        self,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            message = self._pop_message(address)
            while message is None:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)
                message = self._pop_message(address)
            return message

    def _pop_message(self, address: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self._messages:
            return None
        if address is None:
            return self._messages.pop(0)
        for idx, envelope in enumerate(self._messages):
            if envelope.get("address") == address:
                return self._messages.pop(idx)
        return None


def _description_to_dict(description: "RTCSessionDescription") -> SignalingDescription:
    return {"sdp": description.sdp, "type": description.type}


def _dict_to_description(payload: SignalingDescription):
    _, _, _, RTCSessionDescription = _load_aiortc()
    return RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])


__all__ = [
    "WebRTCDataChannelEndpoint",
    "WebRTCIceServer",
    "WebRTCSignalingClient",
    "SignalingDescription",
]
