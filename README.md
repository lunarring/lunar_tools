# Lunar Tools
Utilities for quickly wiring together interactive exhibition prototypes: WebRTC data channels, OSC, ZMQ streaming, audio/vision helpers, and more.

## Install
```bash
python -m pip install git+https://github.com/lunarring/lunar_tools
```

Optional extras:
```bash
# WebRTC (aiortc)
python -m pip install "lunar_tools[webrtc]"
```

## Comms Example Pairs
Each example lives under `examples/comms/`. Run the sender and receiver in separate terminals (or separate machines) using the copy-paste blocks below.

### WebRTC Data Channel
```bash
# Terminal 1 – sender + embedded signaling server
python examples/comms/webrtc_sender.py --session demo

# Terminal 2 – receiver (auto-discovers the sender session)
python examples/comms/webrtc_receiver.py --session demo
```

### OSC
```bash
# Terminal 1 – visualize incoming OSC values
python examples/comms/osc_receiver.py --ip 0.0.0.0 --port 8003

# Terminal 2 – emit demo sine/triangle waves
python examples/comms/osc_sender.py --ip 127.0.0.1 --port 8003 --channels /env1 /env2 /env3
```

### ZMQ Pair Streaming
```bash
# Terminal 1 – receive JSON/image/audio payloads
python examples/comms/zmq_receiver.py --ip 0.0.0.0 --port 5556

# Terminal 2 – send sample payload batches
python examples/comms/zmq_sender.py --ip 127.0.0.1 --port 5556
```

### Need Help?
Open an issue on GitHub or ping the Lunar team on Slack with the command you ran and any console output. We'll grow these examples as new transports land. Have fun!
