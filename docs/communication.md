# Communication

Install the `comms` extra (`python -m pip install lunar_tools[comms]`) to enable OSC and ZeroMQ helpers. These adapters hide the boilerplate of sockets, polling loops, and JPEG encoding so you can focus on message payloads.

## ZeroMQ pair endpoints

`ZMQPairEndpoint` creates a bidirectional socket with background threads that flush send queues and collect inbound data. One side binds (`is_server=True`), the other connects.

```python
import time
import numpy as np
import lunar_tools as lt

server = lt.ZMQPairEndpoint(is_server=True, ip="127.0.0.1", port="5556")
client = lt.ZMQPairEndpoint(is_server=False, ip="127.0.0.1", port="5556")

client.send_json({"hello": "from client"})
time.sleep(0.01)
print("Server received:", server.get_messages())

server.send_json({"status": "ready"})
time.sleep(0.01)
print("Client received:", client.get_messages())

image = (np.random.rand(480, 640, 3) * 255).astype("uint8")
client.send_img(image)
time.sleep(0.01)
frame = server.get_img()
print("Image decoded:", frame.shape if frame is not None else None)
```

Key features:
- JPEG encoding uses OpenCV (`send_img`) with adjustable quality via `configure_image_encoding(format=".webp", webp_quality=80)`.
- `send_audio` streams numpy arrays, tagging sample rate and channel layout.
- All network IO runs on a single background thread; `stop()` shuts the socket down cleanly.

### Remote rendering pipeline

Pair the endpoint with [`Renderer`](vision_and_display.md) to push frames from a GPU node to a lightweight display client.

```python
# sender.py
import time
import numpy as np
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

```python
# receiver.py
import lunar_tools as lt

endpoint = lt.ZMQPairEndpoint(is_server=False, ip="192.168.1.10", port="5557")
renderer = lt.Renderer(width=1280, height=720)

try:
    while True:
        frame = endpoint.get_img()
        if frame is not None:
            renderer.render(frame)
finally:
    endpoint.stop()
```

## OSC endpoints

OSC is perfect for lightweight parameter updates and integrations with creative coding tools (TouchDesigner, Max/MSP, etc.).

```python
import time
import math
import lunar_tools as lt

sender = lt.OSCSender("127.0.0.1", port=9000)
receiver = lt.OSCReceiver("127.0.0.1", port=9000)

for _ in range(100):
    value = (math.sin(time.time()) + 1) * 0.5
    sender.send_message("/synth/fader1", value)
    time.sleep(0.05)

print("Latest values:", receiver.get_all_values("/synth/fader1"))
```

`OSCReceiver` caches the most recent values per address so you can sample them at your own cadence.

## Utility: determining the local IP

For headless installs you may not know which IP to share with clients. `lt.get_local_ip()` inspects network interfaces and falls back to the system hostname.

```python
import lunar_tools as lt

print("Control endpoint reachable at:", lt.get_local_ip())
```
