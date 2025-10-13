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
