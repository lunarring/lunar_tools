import os
import socket
import sys
import unittest

import numpy as np
import requests

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
sys.path.append(os.path.join(os.getcwd(), 'lunar_tools', 'comms'))

from webrtc_codec import decode_message, encode_message  # noqa: E402
from webrtc_signaling import SimpleWebRTCSignalingServer  # noqa: E402


class TestWebRTCCodec(unittest.TestCase):
    def test_json_roundtrip(self):
        payload = {"value": 123, "label": "demo"}
        encoded = encode_message("status", payload)
        self.assertFalse(encoded.is_binary)
        decoded = decode_message(encoded.data)
        self.assertEqual(decoded["address"], "status")
        self.assertEqual(decoded["kind"], "json")
        self.assertEqual(decoded["payload"], payload)

    def test_bytes_roundtrip(self):
        data = b"abc123"
        encoded = encode_message("blob", data)
        self.assertTrue(encoded.is_binary)
        decoded = decode_message(encoded.data)
        self.assertEqual(decoded["address"], "blob")
        self.assertEqual(decoded["kind"], "bytes")
        self.assertEqual(decoded["payload"], data)

    def test_ndarray_roundtrip(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        encoded = encode_message("frames", array)
        self.assertTrue(encoded.is_binary)
        decoded = decode_message(encoded.data)
        payload = decoded["payload"]
        self.assertEqual(decoded["address"], "frames")
        self.assertEqual(decoded["kind"], "ndarray")
        self.assertTrue(isinstance(payload, np.ndarray))
        np.testing.assert_array_equal(payload, array)


class TestWebRTCSignalingServer(unittest.TestCase):
    def _start_server(self):
        server = SimpleWebRTCSignalingServer(host="127.0.0.1", port=0)
        try:
            server.start()
        except OSError as exc:  # pragma: no cover - env restrictions
            if getattr(exc, "errno", None) == 1:
                self.skipTest("Socket binding is not permitted in this environment")
            raise
        address = server.address()
        self.assertIsNotNone(address)
        host, port = address
        base_url = f"http://{host}:{port}"
        return server, base_url, port

    def test_offer_answer_flow(self):
        server, base_url, _ = self._start_server()
        try:
            offer = {"sdp": "o", "type": "offer"}
            resp = requests.post(f"{base_url}/session/demo/offer", json=offer, timeout=5)
            self.assertEqual(resp.status_code, 204)

            resp = requests.get(f"{base_url}/session/demo/offer", timeout=5)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), offer)

            # After the offer is consumed, it should disappear.
            resp = requests.get(f"{base_url}/session/demo/offer", timeout=5)
            self.assertEqual(resp.status_code, 404)
        finally:
            server.stop()

    def test_stop_releases_port(self):
        server, _, port = self._start_server()
        server.stop()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
        except OSError as exc:  # pragma: no cover - env restrictions
            if getattr(exc, "errno", None) == 1:
                self.skipTest("Socket binding is not permitted in this environment")
            raise


if __name__ == "__main__":
    unittest.main()
