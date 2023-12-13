# Introduction
Welcome to Lunar Tools, a comprehensive toolkit designed to fascilitate the programming of interactive exhibitions. Our suite of simple, modular tools is crafted to offer a seamless and hopefully bug-free experience for both exhibitors and visitors.

# Installation
```bash
pip install git+https://github.com/lunarring/lunar_tools
```

On Ubuntu, you may have to install additional dependencies

```bash
sudo apt-get install libasound2-dev
sudo apt-get install libportaudio2
```

Our system includes a convenient automatic mode for reading and writing API keys. This feature enables you to dynamically set your API key as needed, and the file will be stored on your local computer.

However, if you prefer, you can specify your API keys in your shell configuration file (e.g. ~/.bash_profile or ~/.zshrc or ~/.bash_rc). In this case, paste the below lines with the API keys you want to add.
```bash
export OPENAI_API_KEY="XXX"
export REPLICATE_API_TOKEN="XXX"
export ELEVEN_API_KEY="XXX"
```




# Audio
## AudioRecorder
```python
import lunar_tools as lt
import time
audio_recorder = lt.AudioRecorder()
audio_recorder.start_recording("myvoice.mp3")
time.sleep(3)
audio_recorder.stop_recording()    
```

## Speech2Text
```python
import lunar_tools as lt
import time
speech_detector = lt.Speech2Text()
speech_detector.start_recording()
time.sleep(3)
translation = speech_detector.stop_recording()
print(f"translation: {translation}")
```

## Play sounds
```python
import lunar_tools as lt
player = lt.SoundPlayer()
player.play_sound("myvoice.mp3")
```
The playback is threaded and does not block the main application. You can stop the playback via: 
```python
player.stop_sound()
```

## Text2Speech OpenAI
```python
import lunar_tools as lt
text2speech = lt.Text2SpeechOpenAI()
text2speech.change_voice("nova")
text2speech.generate("hey there can you hear me?", "hervoice.mp3")
```

The Text2Speech can also directly generate and play back the sound via: 
```python
text2speech.play("hey there can you hear me?")
```

## Text2Speech elevenlabs
```python
text2speech = Text2SpeechElevenlabs()
text2speech.change_voice("FU5JW1L0DwfWILWkNpW6")
text2speech.play("hey there can you hear me?")
```


# Large Language Models
## GPT4
```python
import lunar_tools as lt
gpt4 = lt.GPT4()
msg = gpt4.generate("tell me about yourself")
```

# Logging and terminal printing
```python
import lunar_tools as lt
logger = lt.LogPrint()  # No filename provided, will use default current_dir/logs/%y%m%d_%H%M
logger.print("white")
logger.print("red", "red")
logger.print("green", "green")
```    

# Image gen
## Generate Images with Dall-e-3
```python
import lunar_tools as lt
dalle3 = lt.Dalle3ImageGenerator()
image, revised_prompt = dalle3.generate("a beautiful red house with snow on the roof, a chimney with smoke")
```

# Camera
## Get image from webcam

```python
import lunar_tools as lt
cam = lt.WebCam()
img = cam.get_img()
```

# Fast rendering
Allows to fast render images from torch, numpy or PIL in a window. Can be directly from the GPU, without need to copy.
```python
import lunar_tools as lt
import torch
from PIL import Image
sz = (1080, 1920)
renderer = lt.Renderer(width=sz[1], height=sz[0])
while True:
    # image = np.random.rand(sz[0],sz[1],4) * 255 # numpy array
    # image = Image.fromarray(np.uint8(np.random.rand(sz[0],sz[1],4) * 255)) # PIL array
    image = torch.rand((sz[0],sz[1],4)) * 255 # Torch tensors
    renderer.render(image)
```

# Movie handling
## Saving a series of images as movie
```python
import lunar_tools as lt
ms = lt.MovieSaver("my_movie.mp4", fps=24)
for _ in range(10):
    img = (np.random.rand(512, 1024, 3) * 255).astype(np.uint8)
    ms.write_frame(img)
ms.finalize()
```

## Loading movie and retrieving frames
```python
import lunar_tools as lt
mr = lt.MovieReader("my_movie.mp4")
for _ in range(mr.nmb_frames):
    img = mr.get_next_frame()
```



# Devinfos
## Testing
pip install pytest

make sure you are in base folder
```python
pytest lunar_tools/tests/
```

## Get requirements
```python
pipreqs . --force
```



