import json
import struct
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

_HEADER_STRUCT = struct.Struct("!I")


@dataclass
class EncodedMessage:
    is_binary: bool
    data: bytes | str


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def encode_message(address: str, payload: Any) -> EncodedMessage:
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    elif isinstance(payload, bytearray):
        payload = bytes(payload)

    if isinstance(payload, np.ndarray):
        array = np.ascontiguousarray(payload)
        header = {
            "address": address,
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        body = array.tobytes()
        return EncodedMessage(True, _pack_binary(header, body))

    if isinstance(payload, bytes):
        header = {"address": address, "kind": "bytes"}
        return EncodedMessage(True, _pack_binary(header, payload))

    if isinstance(payload, str):
        envelope = {"address": address, "kind": "text", "payload": payload}
        return EncodedMessage(False, json.dumps(envelope, ensure_ascii=True))

    envelope = {"address": address, "kind": "json", "payload": payload}
    return EncodedMessage(False, json.dumps(envelope, default=_json_default, ensure_ascii=True))


def decode_message(message: bytes | str) -> Dict[str, Any]:
    if isinstance(message, str):
        envelope = json.loads(message)
        return {
            "address": envelope.get("address"),
            "payload": envelope.get("payload"),
            "kind": envelope.get("kind", "json"),
        }

    if len(message) < _HEADER_STRUCT.size:
        raise ValueError("Binary message too short to contain header length")

    header_len = _HEADER_STRUCT.unpack_from(message, 0)[0]
    header_end = _HEADER_STRUCT.size + header_len
    if header_end > len(message):
        raise ValueError("Binary header length exceeds message size")

    header = json.loads(message[_HEADER_STRUCT.size : header_end].decode("utf-8"))
    body = message[header_end:]
    kind = header.get("kind", "bytes")

    if kind == "ndarray":
        dtype = header.get("dtype")
        shape = header.get("shape")
        if not dtype or shape is None:
            raise ValueError("ndarray payload missing dtype or shape metadata")
        array = np.frombuffer(body, dtype=np.dtype(dtype))
        array = array.reshape(shape)
        payload = array
    else:
        payload = body

    return {
        "address": header.get("address"),
        "payload": payload,
        "kind": kind,
    }


def _pack_binary(header: Dict[str, Any], body: bytes) -> bytes:
    header_bytes = json.dumps(header, ensure_ascii=True).encode("utf-8")
    return _HEADER_STRUCT.pack(len(header_bytes)) + header_bytes + body


__all__ = ["EncodedMessage", "encode_message", "decode_message"]
