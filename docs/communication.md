# Communication

Install the `comms` extra (`python -m pip install lunar_tools[comms]`) to enable
OSC, ZeroMQ, and WebRTC adapters. Phase C introduces a service-layer message bus so
presentation code can stay ignorant of socket details.

## Message bus quickstart

```python
from lunar_tools import MessageBusConfig, create_message_bus

services = create_message_bus(
    MessageBusConfig(
        osc_host="127.0.0.1",
        osc_port=9000,
        osc_default_address="/lunar/state",
        zmq_bind=True,
        zmq_port=5556,
        zmq_default_address="control",
    )
)

bus = services.message_bus

# Send to a specific transport
bus.send("osc", {"led": 1})

# Broadcast across all registered transports
bus.broadcast({"mode": "ambient"})

# Poll for inbound messages (returns {"address": ..., "payload": ...})
message = bus.wait_for("zmq", timeout=5.0)
if message:
    print("Received:", message)
```

`create_message_bus` lazily imports the OSC/ZeroMQ adapters, so environments
without the extras can still call it (they just receive services without
endpoints). Receivers auto-start by default; use `bus.stop_all()` during
shutdown.

### Custom routing

When sharing transports across multiple addresses you can override each call:

```python
bus.send("osc", 0.42, address="/synth/fader1")
reading = bus.wait_for("osc", address="/sensors/theremin", timeout=1.0)
```

### Wiring by hand

If you need to swap the default adapters, construct them manually and register
them with the service:

```python
from lunar_tools.adapters.comms.osc_endpoints import OSCMessageReceiver, OSCMessageSender
from lunar_tools.services.comms.message_bus import MessageBusService

bus = MessageBusService()
sender = OSCMessageSender("192.168.1.22", 9100)
receiver = OSCMessageReceiver("0.0.0.0", 9100)

bus.register_sender("osc", sender)
bus.register_receiver("osc", receiver, auto_start=True)
```

## WebRTC data channels

`aiortc` powers the new WebRTC transport, giving you an encrypted, low-latency
way to ship JSON blobs, raw image tensors, `numpy` arrays, or pre-encoded bytes
directly to browsers or other peers. The sender example (and
`SimpleRestSignalingServer`) hosts a tiny REST endpoint so peers can trade SDP
offers/answers before the peer connection comes online. Point both peers at the
same URL by configuring `WebRTCConfig.signaling_url`:

```python
from lunar_tools import MessageBusConfig, WebRTCConfig, create_message_bus

services = create_message_bus(
    MessageBusConfig(
        webrtc=WebRTCConfig(
            session_id="demo-session",
            role="offer",  # sender side
            signaling_url="http://192.168.4.10:8787",
            channel_label="lunar-data",
            default_address="frames",
        )
    )
)

bus = services.message_bus
bus.send("webrtc", frame_array, address="frames")
bus.send("webrtc", {"scene": "sunrise"}, address="control")
message = bus.wait_for("webrtc", timeout=5.0)
if message:
    kind = message.get("kind")  # "json", "text", "bytes", or "ndarray"
    payload = message["payload"]
```

Receivers call the same bus APIs; the adapter tags each envelope with a `kind`
field so you can branch on payload type quickly. See
[`examples/webrtc_sender.py`](../examples/webrtc_sender.py) and
[`examples/webrtc_receiver.py`](../examples/webrtc_receiver.py) for end-to-end
scripts that stream numpy frames plus JPEG bytes and print what arrives on the
other side. Run the sender (role `offer`) first so its embedded signaling server
is listening, then launch receivers with `role="answer"` pointing to the same
host/port.

## ZeroMQ and OSC adapters (legacy API)

The legacy helpers remain available during the migration window:

- `lt.ZMQPairEndpoint` – bidirectional PAIR socket with helpers for JSON, image,
  and audio payloads.
- `lt.OSCSender` / `lt.OSCReceiver` – thin wrappers around python-osc with
  historical buffering semantics.

```python
import numpy as np
import time
import lunar_tools as lt

endpoint = lt.ZMQPairEndpoint(is_server=True, ip="0.0.0.0", port="5557")
try:
    while True:
        frame = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
        endpoint.send_img(frame)
        time.sleep(1 / 30)
finally:
    endpoint.stop()
```

## Utility: determining the local IP

For headless installs you may not know which IP to share with clients.
`lt.get_local_ip()` inspects network interfaces and falls back to the hostname.

```python
import lunar_tools as lt

print("Control endpoint reachable at:", lt.get_local_ip())
```
