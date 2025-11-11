# Example scripts

Short descriptions of the runnable demos in [`examples/`](../examples). Use these as blueprints when wiring components together.

| Script | Required extras | What it shows |
| --- | --- | --- |
| [`realtime_voice_example.py`](../examples/realtime_voice_example.py) | `audio`, `llm` | Bootstraps the audio stack, reuses `AudioConversationController`, and drives `RealTimeVoice` with realtime transcript polling and TTS playback. |
| [`fps_tracker_example.py`](../examples/fps_tracker_example.py) | _none_ | Minimal `FPSTracker` usage with simulated workloads to visualise segment timings. |
| [`simple_webcam_renderer.py`](../examples/simple_webcam_renderer.py) | `camera`, `display` | Smallest possible webcam→renderer loop; great for hardware smoke tests. |
| [`webcam_renderer_example.py`](../examples/webcam_renderer_example.py) | `camera`, `display` | Adds error handling and FPS metrics to the webcam preview so you can benchmark capture latency. |
| [`zmq_remote_renderer_sender.py`](../examples/zmq_remote_renderer_sender.py) / [`zmq_remote_renderer_receiver.py`](../examples/zmq_remote_renderer_receiver.py) | `comms`, `display` | Streams frames over the ZeroMQ message bus and boots the receiver via `DisplayStackConfig`. |
| [`midi_meta_example.py`](../examples/midi_meta_example.py) | `inputs` | Demonstrates the control input stack auto-detecting Akai MIDImix / LPD8 mappings. |
| [`flux_kontext_gradio.py`](../examples/flux_kontext_gradio.py) | `imaging` + `gradio` | Gradio UI that pipes drawings/uploads through the Flux Kontext image-edit endpoint. |
| [`nano_banana_edit_gradio.py`](../examples/nano_banana_edit_gradio.py) | `imaging` + `gradio` | Advanced image editing workflow with masks, reference galleries, and Nano Banana edits. |
| [`movie_example.py`](../examples/movie_example.py) | `video` | Placeholder for a MovieSaver demo—use [`vision_and_display.md`](vision_and_display.md) snippets until this script is populated. |

Tip: run scripts directly with `python examples/<name>.py` from the project root so relative paths resolve correctly.
