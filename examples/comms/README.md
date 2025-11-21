# Communications examples

Self-contained demos that exercise the Lunar Tools communications adapters.
Run every script from the repository root so relative imports resolve correctly.

| Script | Scenario | Command | Extras |
| --- | --- | --- | --- |
| `webrtc_sender.py` | Streams numpy frames, JPEG snaps, and JSON status updates over a WebRTC data channel with an embedded REST signaling server. | `python examples/comms/webrtc_sender.py --session demo-session` | `lunar_tools[comms]` |
| `webrtc_receiver.py` | Connects to the sender, accepts data-channel payloads, and logs inbound frames/bytes/messages. | `python examples/comms/webrtc_receiver.py --session demo-session --role answer` | `lunar_tools[comms]` |
| `zmq_remote_renderer_sender.py` | Publishes animated RGB frames via ZeroMQ for a remote renderer. | `python examples/comms/zmq_remote_renderer_sender.py --port 5557 --fps 30` | `lunar_tools[comms]` |
| `zmq_remote_renderer_receiver.py` | Subscribes to the sender, pipes frames through the display stack, and optionally shows FPS metrics. | `python examples/comms/zmq_remote_renderer_receiver.py --endpoint 127.0.0.1 --port 5557` | `lunar_tools[comms,display]` |
| `osc_sender.py` | Emits a sine-wave value to an OSC address to sanity-check downstream devices. | `python examples/comms/osc_sender.py --address /lunar/demo` | `lunar_tools[comms]` |
| `osc_receiver.py` | Binds an OSC server through the message bus and prints every inbound address/payload pair. | `python examples/comms/osc_receiver.py --address /lunar/demo` | `lunar_tools[comms]` |

Feel free to copy these scripts into your own projects and swap in production
configuration (e.g., different signaling URLs, OSC addresses, or display backends).

