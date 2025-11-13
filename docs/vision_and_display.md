# Vision and Display

Visual output ranges from windowed previews to GPU pipelines and offline movie rendering. Install the extras that map to the APIs you need:
- `display` for `Renderer`, `GridRenderer`, `torch_utils`, and font rendering.
- `imaging` for generative models (OpenAI DALL·E, FAL Flux, Replicate SDXL).
- `video` for MoviePy/ffmpeg wrappers.
- `camera` to pair with [`WebCam`](inputs.md).

## Rendering windows

### Renderer

`Renderer` opens an SDL/OpenGL window and blits numpy arrays, Pillow images, or torch tensors. Use RGBA (height, width, 4) for best results.

```python
import numpy as np
import lunar_tools as lt

renderer = lt.Renderer(width=1280, height=720, window_title="Prototype feed")

while True:
    frame = (np.random.rand(720, 1280, 4) * 255).astype("uint8")
    renderer.render(frame)
```

Hints:
- Disable VSYNC in your GPU driver if you need ultra-low-latency swaps.
- On macOS the backend defaults to OpenCV/pygame because native OpenGL support varies by chipset.
- For tiled dashboards check out `lt.GridRenderer`.

### GridRenderer

Arrange multiple feeds on screen without building your own compositor.

```python
import numpy as np
import lunar_tools as lt

grid = lt.GridRenderer(nmb_rows=2, nmb_cols=2, shape_hw=(480, 640))

while True:
    tiles = [
        (np.random.rand(480, 640, 3) * 255).astype("uint8")
        for _ in range(4)
    ]
    grid.update(tiles)
    grid.render()
```

## Display bootstrap

Phase D adds a presentation-layer factory so scripts can compose renderers,
message buses, and vision providers without reimplementing wiring.

```python
from lunar_tools import MessageBusConfig
from lunar_tools.presentation.display_stack import (
    DisplayStackConfig,
    bootstrap_display_stack,
)

stack = bootstrap_display_stack(
    DisplayStackConfig(
        backend="gl",
        attach_message_bus=True,
        include_image_generators=True,
        message_bus_config=MessageBusConfig(
            zmq_bind=False,
            zmq_host="127.0.0.1",
            zmq_port=5557,
            zmq_default_address="frames",
        ),
    )
)

renderer = stack.renderer
bus = stack.communication.message_bus  # requires attach_message_bus=True
generators = stack.generators          # present when include_image_generators=True
```

`stack.close()` stops registered receivers/endpoints on shutdown.

### Webcam CLI

Run the shared CLI when you want a configurable webcam preview without writing
Python glue:

```bash
python -m lunar_tools.presentation.webcam_display \
    --config examples/configs/webcam_display.yaml
```

`camera` and `display_stack` sections mirror the dataclasses. Override settings on
the command line (`--cam-id`, `--backend`, `--mirror`, `--print-fps`, etc.). See
[`configuration.md`](configuration.md) for schema details.

## Generative imagery (`imaging` extra)

### OpenAI DALL·E 3

```python
import lunar_tools as lt

generator = lt.Dalle3ImageGenerator()
image, revised_prompt = generator.generate(
    "A kinetic sculpture made of mirrors and soft neon light",
)
image.save("outputs/sculpture.png")
```

### SDXL Turbo via Replicate

```python
import lunar_tools as lt

sdxl = lt.SDXL_TURBO()
image, url = sdxl.generate("Time-lapse streaks of a city at dusk", mode="photo")
print("Preview:", url)
image.save("outputs/city.png")
```

For more providers (Flux, GLIF) inspect `lunar_tools/image_gen.py`.

### Selecting providers via the service registry

Phase C adds a registry so you can pick generators dynamically without importing
individual adapters:

```python
from lunar_tools.image_gen import create_image_generators

generators = create_image_generators(include_glif=False)

dalle = generators.get("openai")
result = dalle.generate("Minimalist pavilion bathed in sunrise light")
result.image.save("outputs/pavilion.png")

# Switch providers later in the same script
flux = generators.get("flux")
flux.generate("Posterised glitch art of a howling wolf").image.save("outputs/wolf.png")
```

Aliases such as `"openai"`/`"dalle"` and `"sdxl"`/`"replicate_lcm"` are
available; call `generators.registry.available()` to inspect the list.

## Movie helpers (`video` extra)

### Stitching frames into a movie

```python
import numpy as np
import lunar_tools as lt

saver = lt.MovieSaver("renders/showreel.mp4", fps=30)
for _ in range(180):
    frame = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
    saver.write_frame(frame)
saver.finalize()
```

### Reading frames back

```python
import lunar_tools as lt

reader = lt.MovieReader("renders/showreel.mp4")
for _ in range(reader.nmb_frames):
    frame = reader.get_next_frame()
    # Process frame...
```

Other utilities:
- `lt.add_sound(video_path, audio_path, output_path)`
- `lt.concatenate_movies([...], output_path)`
- `lt.MovieSaverThreaded` for background encoding while the main loop keeps running.

### Movie bootstrap

Use the presentation factory when you need to bundle writing, messaging, and
optional vision providers:

```python
from lunar_tools import MessageBusConfig
from lunar_tools.presentation.movie_stack import (
    MovieStackConfig,
    bootstrap_movie_stack,
)

stack = bootstrap_movie_stack(
    MovieStackConfig(
        output_path="renders/showreel.mp4",
        attach_message_bus=True,
        include_image_generators=True,
        message_bus_config=MessageBusConfig(zmq_bind=True, zmq_port=5557),
    )
)

writer = stack.writer
generators = stack.generators
```

Call `stack.close()` once writing or streaming completes to stop endpoints.

## Torch image utilities (`display` extra)

`lunar_tools.torch_utils` includes filters and interpolation helpers that operate on CUDA tensors without forcing copies back to CPU. A quick example:

```python
import torch
from lunar_tools import GaussianBlur

image = torch.rand(1, 3, 720, 1280, device="cuda")
blur = GaussianBlur(kernel_size=5, sigma=1.2)
smoothed = blur(image)
```

These are the same building blocks the renderer relies on internally, making it easy to slot them into your own pipelines.
