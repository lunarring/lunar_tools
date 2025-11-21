from __future__ import annotations

import numpy as np

from lunar_tools.adapters.comms.webrtc_codec import decode_message, encode_message


def test_json_round_trip():
    message = {"value": 42, "label": "test"}
    encoded = encode_message("control", message)
    assert not encoded.is_binary
    decoded = decode_message(encoded.data)
    assert decoded["address"] == "control"
    assert decoded["payload"] == message
    assert decoded["kind"] == "json"


def test_text_round_trip():
    encoded = encode_message("chat", "hello")
    assert not encoded.is_binary
    decoded = decode_message(encoded.data)
    assert decoded["payload"] == "hello"
    assert decoded["kind"] == "text"


def test_bytes_round_trip():
    payload = b"binary-data"
    encoded = encode_message("frames", payload)
    assert encoded.is_binary
    decoded = decode_message(encoded.data)
    assert decoded["payload"] == payload
    assert decoded["kind"] == "bytes"


def test_numpy_round_trip():
    frame = (np.random.rand(4, 4, 3) * 255).astype("uint8")
    encoded = encode_message("frames", frame)
    assert encoded.is_binary
    decoded = decode_message(encoded.data)
    restored = decoded["payload"]
    assert isinstance(restored, np.ndarray)
    assert np.array_equal(restored, frame)
    assert decoded["kind"] == "ndarray"
