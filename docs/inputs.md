# Inputs

Capture devices and controllers form the backbone of an interactive exhibit. Lunar Tools exposes simple wrappers that hide the streaming details while keeping latency low. Each subsection lists the required extra so you know which optional dependencies to install.

## AudioRecorder (`audio` extra)

`AudioRecorder` is an alias for the sounddevice-based recorder adapter. It records straight to disk and optionally streams PCM frames to your own buffers.

```python
import time
import lunar_tools as lt

recorder = lt.AudioRecorder(rate=48_000, channels=1)
recorder.start_recording("takes/line_check.mp3")
time.sleep(5)
recorder.stop_recording()
```

Tips:
- The current adapter exports MP3 files. If you need WAV you can transcode with `pydub.AudioSegment.from_file("takes/line_check.mp3").export("scene.wav", format="wav")`.
- Set `rate`, `channels`, and `chunk` in the constructor to balance latency and quality.
- Use `max_time` in `start_recording("clip.mp3", max_time=10)` to cap capture duration automatically.
- Combine with [`lt.FPSTracker`](logging_and_monitoring.md) to monitor capture cadence.

## WebCam (`camera` extra)

`WebCam` spins up a background thread that keeps the most recent frame ready for you. Auto-detection is built in—pass `cam_id="auto"` to scan connected devices.

```python
import lunar_tools as lt

cam = lt.WebCam(cam_id="auto", shape_hw=(720, 1280))

image = cam.get_img()
if image is not None:
    print("Frame size:", image.shape)
```

Extras:
- `do_mirror=True` mirrors the preview so installations feel natural.
- `do_digital_exposure_accumulation=True` will accumulate frames for low-light scenarios.

## MIDI controllers (`inputs` extra)

`MidiInput` treats controllers as a grid (columns A–H, rows 0–7). Button accessors let you choose toggle, pressed-once, released-once, or held modes.

```python
import time
import lunar_tools as lt

midi = lt.MidiInput(device_name="akai_lpd8")
while True:
    scene_trigger = midi.get("A0", button_mode="pressed_once")
    brightness = midi.get("D1", val_min=0.1, val_max=1.0)
    if scene_trigger:
        print("Next scene!")
    print(f"Brightness: {brightness:.2f}", end="\r")
    time.sleep(0.05)
```

### Mapping MIDI to named variables

The `MetaInput` helper is deprecated but still available for projects that rely on auto-labeling controls. It falls back to keyboard mode when no MIDI device is discovered.

```python
import time
import lunar_tools as lt

controls = lt.MetaInput(force_device="akai_lpd8")

while True:
    # Automatically assigns the variable name 'intensity' to slider B0.
    intensity = controls.get(akai_lpd8="B0", val_min=0.0, val_max=1.0)
    trigger = controls.get(akai_lpd8="A0", button_mode="toggle")
    if trigger:
        print(f"Intensity engaged at {intensity:.2f}")
    time.sleep(0.1)
```

If you are writing new code prefer using `MidiInput` directly—it is explicit, easier to test, and will continue receiving updates.

## Keyboard fallback (`inputs` extra)

When no MIDI hardware is plugged in you can still prototype interactions using `KeyboardInput`. Register slider-like behaviour with arrow keys.

```python
import time
import lunar_tools as lt

keyboard = lt.KeyboardInput()

while True:
    toggle = keyboard.get("x", button_mode="toggle")
    slider = keyboard.get("z", val_min=0.0, val_max=1.0, val_default=0.5)
    print(f"toggle={toggle} slider={slider:.2f}", end="\r")
    time.sleep(0.05)
```

Tap `z` once to focus the emulated slider, then use the up/down arrow keys to step through the configured range.

See [`examples/midi_meta_example.py`](../examples/midi_meta_example.py) for a full MIDI walkthrough.

## Presentation bootstrap

Prefer the presentation-layer factory when wiring controllers into larger demos.

```python
from lunar_tools import MessageBusConfig
from lunar_tools.presentation.input_stack import (
    ControlInputStackConfig,
    bootstrap_control_inputs,
)

stack = bootstrap_control_inputs(
    ControlInputStackConfig(
        use_meta=True,
        attach_message_bus=True,
        message_bus_config=MessageBusConfig(zmq_bind=False),
    )
)

values = stack.poll_and_broadcast(
    {
        "intensity": {"akai_lpd8": "B0", "val_min": 0.0, "val_max": 1.0},
        "trigger": {"akai_lpd8": "A0", "button_mode": "toggle"},
    }
)
```

`stack.communication` exposes the optional message bus, and `stack.close()` will
stop any registered receivers.
