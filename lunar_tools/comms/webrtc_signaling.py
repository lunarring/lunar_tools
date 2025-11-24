import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class RestSignalingClient:
    """Polling REST client used to exchange SDP blobs between peers."""

    def __init__(self, *, base_url: str, session_id: str, role: str, request_timeout: float = 30.0) -> None:
        if role not in {"offer", "answer"}:
            raise ValueError("role must be 'offer' or 'answer'")
        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._role = role
        self._remote_role = "answer" if role == "offer" else "offer"
        self._request_timeout = max(1.0, request_timeout)

    async def publish_local_description(self, description: Dict[str, str]) -> None:
        await self._to_thread(self._post_description, self._role, description)

    async def wait_for_remote_description(self, *, timeout: Optional[float] = 30.0) -> Dict[str, str]:
        result = await self._to_thread(self._wait_for_description, self._remote_role, timeout)
        if result is None:
            raise TimeoutError("Timed out waiting for remote SDP")
        return result

    async def _to_thread(self, func, *args):
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    def _post_description(self, role: str, description: Dict[str, str]) -> None:
        url = f"{self._base_url}/session/{self._session_id}/{role}"
        response = requests.post(url, json=description, timeout=self._request_timeout)
        response.raise_for_status()

    def _wait_for_description(self, role: str, timeout: Optional[float]) -> Optional[Dict[str, str]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        url = f"{self._base_url}/session/{self._session_id}/{role}"
        max_wait = max(0.5, self._request_timeout - 0.5)
        while True:
            wait = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                wait_candidate = min(10.0, remaining)
                if wait_candidate > 0:
                    wait = min(wait_candidate, max_wait)
            params = {"wait": f"{wait:.2f}"} if wait is not None else None
            response = requests.get(url, params=params, timeout=self._request_timeout)
            if response.status_code == 200:
                return response.json()
            if response.status_code not in (404, 204):
                response.raise_for_status()
            if deadline is None:
                time.sleep(1.0)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                time.sleep(min(1.0, remaining))


class SimpleWebRTCSignalingServer:
    """Minimal HTTP server that stores SDP offers/answers in memory."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8787) -> None:
        self._host = host
        self._port = port
        self._sessions: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._condition = threading.Condition()
        self._handler_class = self._build_handler()
        self._server: Optional[_ReusableThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _build_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                server._handle_post(self)

            def do_GET(self):  # noqa: N802
                server._handle_get(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                logger.debug("HTTP: " + format, *args)

        return Handler

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._server = _ReusableThreadingHTTPServer((self._host, self._port), self._handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, name="WebRTCSignalingServer", daemon=True)
        self._thread.start()
        bound = self.address()
        logger.info(
            "WebRTC signaling server listening on http://%s:%s",
            bound[0] if bound else self._host,
            bound[1] if bound else self._port,
        )

    def stop(self) -> None:
        if not self._server:
            return
        self._server.shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._server.server_close()
        self._thread = None
        self._server = None

    def address(self) -> Optional[Tuple[str, int]]:
        if self._server is None:
            return None
        host, port = self._server.server_address
        # When bound to 0.0.0.0 expose the configured host which is usually what peers need.
        if host == "0.0.0.0":
            host = self._host
        return host, port

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        raw_body = handler.rfile.read(length) if length else b""
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_response(handler, 400, {"error": "invalid json"})
            return

        parsed = urlparse(handler.path)
        parts = [segment for segment in parsed.path.split("/") if segment]
        if len(parts) != 3 or parts[0] != "session":
            self._send_response(handler, 404, {"error": "unknown endpoint"})
            return

        session_id, role = parts[1], parts[2]
        if role not in {"offer", "answer"}:
            self._send_response(handler, 400, {"error": "role must be offer or answer"})
            return

        if "sdp" not in body or "type" not in body:
            self._send_response(handler, 400, {"error": "missing sdp or type"})
            return

        with self._condition:
            session = self._sessions.setdefault(session_id, {})
            session[role] = {"sdp": body["sdp"], "type": body["type"]}
            self._condition.notify_all()
        logger.info("Stored %s for session %s", role, session_id)
        self._send_response(handler, 204, None)

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        parts = [segment for segment in parsed.path.split("/") if segment]
        if len(parts) != 3 or parts[0] != "session":
            self._send_response(handler, 404, {"error": "unknown endpoint"})
            return

        session_id, role = parts[1], parts[2]
        if role not in {"offer", "answer"}:
            self._send_response(handler, 400, {"error": "role must be offer or answer"})
            return

        wait_param = parse_qs(parsed.query).get("wait", ["0"])[0]
        try:
            wait_timeout = max(0.0, float(wait_param))
        except ValueError:
            wait_timeout = 0.0

        description = self._wait_for(role, session_id, wait_timeout)
        if description is None:
            logger.debug("No %s yet for session %s (wait %.2fs)", role, session_id, wait_timeout)
            self._send_response(handler, 404, {"error": "not ready"})
        else:
            logger.info("Served %s for session %s", role, session_id)
            self._send_response(handler, 200, description)

    def _wait_for(self, role: str, session_id: str, timeout: float) -> Optional[Dict[str, str]]:
        deadline = time.monotonic() + timeout if timeout > 0 else None
        with self._condition:
            description = self._sessions.get(session_id, {}).get(role)
            while description is None:
                if deadline is None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
                description = self._sessions.get(session_id, {}).get(role)
            if description is not None:
                session = self._sessions.get(session_id)
                if session is not None:
                    session.pop(role, None)
                    if not session:
                        self._sessions.pop(session_id, None)
            return description

    @staticmethod
    def _send_response(handler: BaseHTTPRequestHandler, status: int, payload: Optional[Dict[str, Any]]) -> None:
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            if body:
                handler.wfile.write(body)
        except BrokenPipeError:
            logger.debug("Client closed connection before response could be sent")


__all__ = ["RestSignalingClient", "SimpleWebRTCSignalingServer"]
