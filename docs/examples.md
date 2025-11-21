# Example scripts

Short descriptions of the runnable demos in [`examples/`](../examples). Use these as blueprints when wiring components together.

| Script | Required extras | What it shows |
| --- | --- | --- |
| [`realtime_voice_example.py`](../examples/realtime_voice_example.py) | `audio`, `llm` | Bootstraps the audio stack, reuses `AudioConversationController`, and drives `RealTimeVoice` with realtime transcript polling and TTS playback. Try the CLI with [`configs/realtime_voice.yaml`](../examples/configs/realtime_voice.yaml). |
| [`fps_tracker_example.py`](../examples/fps_tracker_example.py) | _none_ | Minimal `FPSTracker` usage with simulated workloads to visualise segment timings. |
| [`simple_webcam_renderer.py`](../examples/simple_webcam_renderer.py) | `camera`, `display` | Minimal wrapper around the webcam display CLI; pass [`configs/webcam_display.yaml`](../examples/configs/webcam_display.yaml). |
| [`webcam_renderer_example.py`](../examples/webcam_renderer_example.py) | `camera`, `display` | Same CLI with FPS logging enabled for quick benchmarking. |
| [`zmq_remote_renderer_sender.py`](../examples/comms/zmq_remote_renderer_sender.py) / [`zmq_remote_renderer_receiver.py`](../examples/comms/zmq_remote_renderer_receiver.py) | `comms`, `display` | Streams frames over the ZeroMQ message bus and boots the receiver via `DisplayStackConfig`. |
| [`webrtc_sender.py`](../examples/comms/webrtc_sender.py) / [`webrtc_receiver.py`](../examples/comms/webrtc_receiver.py) | `comms` | Sender streams numpy frames, JPEG bytes, JSON metadata, and text heartbeats over WebRTC and auto-hosts the signaling server; receiver attaches via the same host/port. |
| [`osc_sender.py`](../examples/comms/osc_sender.py) / [`osc_receiver.py`](../examples/comms/osc_receiver.py) | `comms` | Sender emits a sine-wave value to an OSC address; receiver listens via the message bus and prints each address/payload pair. |
| [`midi_meta_example.py`](../examples/midi_meta_example.py) | `inputs` | Demonstrates the control input stack auto-detecting Akai MIDImix / LPD8 mappings. Configurable via [`configs/midi_input.yaml`](../examples/configs/midi_input.yaml). |
| [`flux_kontext_gradio.py`](../examples/flux_kontext_gradio.py) | `imaging` + `gradio` | Gradio UI that pipes drawings/uploads through the Flux Kontext image-edit endpoint. |
| [`nano_banana_edit_gradio.py`](../examples/nano_banana_edit_gradio.py) | `imaging` + `gradio` | Advanced image editing workflow with masks, reference galleries, and Nano Banana edits. |
| [`movie_example.py`](../examples/movie_example.py) | `video` | Gallery video generator powered by `MovieStackConfig`; see [`configs/movie_writer.yaml`](../examples/configs/movie_writer.yaml). |

Tip: run scripts directly with `python examples/<name>.py` from the project root so relative paths resolve correctly.
