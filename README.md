# Introduction
Welcome to Lunar Tools, a comprehensive toolkit designed to fascilitate the programming of interactive exhibitions. Our suite of simple, modular tools is crafted to offer a seamless and hopefully bug-free experience for both exhibitors and visitors.

# Installation
Make sure you have python>=3.10.
```bash
python -m pip install git+https://github.com/lunarring/lunar_tools
```
## Ubuntu
On Ubuntu, you may have to install additional dependencies for sound playback/recording.

```bash
sudo apt-get install libasound2-dev libportaudio2
```

For running the midi controllers, you might have to create a symlink:
```bash
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s alsa-lib/libasound_module_conf_pulse.so libasound_module_conf_pulse.so
```

## API Keys

Our system includes a convenient automatic mode for reading and writing API keys. This feature enables you to dynamically set your API key as needed, and the file will be stored on your local computer.
However, if you prefer, you can specify your API keys in your shell configuration file (e.g. ~/.bash_profile or ~/.zshrc or ~/.bash_rc). In this case, paste the below lines with the API keys you want to add.
```bash
export OPENAI_API_KEY="XXX"
export REPLICATE_API_TOKEN="XXX"
export ELEVEN_API_KEY="XXX"
```

# Inputs
## AudioRecorder
```python
import lunar_tools as lt
import time
audio_recorder = lt.AudioRecorder()
audio_recorder.start_recording("myvoice.mp3")
time.sleep(3)
audio_recorder.stop_recording()    
```

## Get image from webcam
```python
import lunar_tools as lt
cam = lt.WebCam()
img = cam.get_img()
```

## Midi Inputs
Allow real-time change of variables, ideal for changing variables on the fly during a infinete *while True* loop.. The logic is that buttons can change boolean variables and sliders can change numerical values. 
For buttons, we support the following modes:
* **toggle**: button activation toggles the state, like a light switch.
* **held_down**: returns if the button is currently, at this moment, being pressed down.
* **pressed_once**: checks if the button has been pressed since the last time we checked, returning a single time "True" if the button is held down (at the beginning). 
* **released_once**: checks if the button has been released since the last time we checked, returning a single time "True" if the button is held down (at the end). 

For sliders, the default is a range between 0.0 and 1.0. The default return value is the middle point between your supplied val_min and val_max, e.g. 0.5.
  
We currently support akai lpd8 and akai midimix devices. However, in principle all midi devices can be added, you just need to specify it in the midi_configs/your_device.yml. In depth how-to coming soon!
We think of the midi device as a grid, where we name the colums with letters ("A", "B", "C", ...) and the rows with numbers (0, 1 , 2, ...). This allows us to identify the buttons/sliders, e.g. "A0" is the most upper left button/slider, and "A1" is the one below it. 


```python
import lunar_tools as lt
akai_lpd8 = lt.MidiInput(device_name="akai_lpd8")

while True:
    time.sleep(0.1)
    a0 = akai_lpd8.get("A0", button_mode='toggle') # toggle switches the state with every press between on and off
    b0 = akai_lpd8.get("B0", button_mode='held_down') # is_held checks if the button is pressed down at the moment
    c0 = akai_lpd8.get("C0", button_mode='released_once') # down_once checks if the button was pressed since we checked last time
    e0 = akai_lpd8.get("E0", val_min=3, val_max=6) # e0 is a slider float between val_min and val_max
    print(f"a0: {a0}, b0: {b0}, c0: {c0}, e0: {e0}")
```

# Outputs
## Play sounds
```python
import lunar_tools as lt
player = lt.SoundPlayer()
player.play_sound("myvoice.mp3")
```

## Real-time display (Torch, PIL, numpy)
Allows to fast render images from torch, numpy or PIL in a window. Can be directly from the GPU, without need to copy. Works with np arrays, PIL.Image and torch tensors (just uncomment from below).
```python
import lunar_tools as lt
import torch
from PIL import Image
import numpy as np
sz = (1080, 1920)
renderer = lt.Renderer(width=sz[1], height=sz[0])
while True:
    # image = np.random.rand(sz[0],sz[1],4) * 255 # numpy array
    # image = Image.fromarray(np.uint8(np.random.rand(sz[0],sz[1],4) * 255)) # PIL array
    image = torch.rand((sz[0],sz[1],4)) * 255 # Torch tensors
    renderer.render(image)
```

note you can speed-up opengl render calls by upto a factor of 3 by disabling
VSYNC on your system
On Ubuntu do: Run nvidia-settings 2. Screen 0 > OpenGl > Sync to VBLank ->
off

## FPS Tracker
Track frame rates and segment timings with color-coded console output.

```python
import lunar_tools as lt

fps = lt.FPSTracker()
while True:
    fps.start_segment("process")
    # Your processing code here
    
    fps.start_segment("render")
    # Your rendering code here
    
    fps.print_fps()  # Output: FPS: 60.0 | process: 8.3ms | render: 5.2ms
```



# Language

## RealtimeVoice OpenAI

### Simple Example
```python
instructions = "Respond briefly and with a sarcastic attitude."
rtv = RealTimeVoice(instructions=instructions)
rtv.start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    rtv.stop()
    print("\nExiting.")
```

### Advanced Example
```python
instructions = "Respond briefly and with a sarcastic attitude."
temperature = 0.6
voice = "echo"
mute_mic_while_ai_speaking = True # that's the default already, just FYI

# Optional: callback for when the whisper transcription is done
async def on_user_transcript(transcript: str):
    print(f"(on_user_transcript) User said: {transcript}")

# Optional: callback for when the transcript of the voice response is there
async def on_ai_transcript(transcript: str):
    print(f"(on_ai_transcript) AI replied: {transcript}")

# Optional: callback for when the audio has been completely played
async def on_audio_complete():
    print("(on_audio_complete) AI audio has been completely played.")

rtv = RealTimeVoice(
    instructions=instructions,
    on_user_transcript=on_user_transcript,
    on_ai_transcript=on_ai_transcript,
    on_audio_complete=on_audio_complete,
    model="gpt-4o-mini-realtime-preview-2024-12-17",
    temperature=temperature,
    voice=voice,
    mute_mic_while_ai_speaking=mute_mic_while_ai_speaking,
    max_response_output_tokens="inf",
)

rtv.start()

# Let's inject an initial message so we have a conversation started. We can do this at any point!
rtv.inject_message("Hello AI, what's up?")

try:
    while True:
        cmd = input("Commands: (p) pause, (r) resume, (s) stop, (i) inject <msg>, (u) update_instructions <text>, (t) print_transcript\n> ").strip()
        if cmd.lower() == "p":
            rtv.pause()
        elif cmd.lower() == "r":
            rtv.resume()
        elif cmd.lower() == "s":
            rtv.stop()
            break
        elif cmd.lower().startswith("i "):
            message = cmd[len("i "):].strip()
            rtv.inject_message(message)
        elif cmd.lower().startswith("u "):
            new_instructions = cmd[len("u "):].strip()
            rtv.update_instructions(new_instructions)
        elif cmd.lower() == "t":
            print("\n".join([f"{entry.timestamp} {entry.role}: {entry.message}" for entry in rtv.transcripts]))
        else:
            print("Unknown command.")
except KeyboardInterrupt:
    rtv.stop()
    print("\nExiting.")
```


## OpenAI
```python
import lunar_tools as lt
openai = lt.OpenAIWrapper()
msg = openai.generate("tell me about yourself")
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

# Image generation APIs
## Generate Images with Dall-e-3
```python
import lunar_tools as lt
dalle3 = lt.Dalle3ImageGenerator()
image, revised_prompt = dalle3.generate("a beautiful red house with snow on the roof, a chimney with smoke")
```

## Generate Images with SDXL Turbo
```python
import lunar_tools as lt
sdxl_turbo = lt.SDXL_TURBO()
image, img_url = sdxl_turbo.generate("An astronaut riding a rainbow unicorn", "cartoon")
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

# Communication
## ZMQ
We have a unified client-server setup in this example, which demonstrates bidirectional communication using ZMQPairEndpoints.
```python
# Create server and client
server = lt.ZMQPairEndpoint(is_server=True, ip='127.0.0.1', port='5556')
client = lt.ZMQPairEndpoint(is_server=False, ip='127.0.0.1', port='5556')

# Client: Send JSON to Server
client.send_json({"message": "Hello from Client!"})
time.sleep(0.01)
# Server: Check for received messages
server_msgs = server.get_messages()
print("Messages received by server:", server_msgs)

# Server: Send JSON to Client
server.send_json({"response": "Hello from Server!"})
time.sleep(0.01)

# Client: Check for received messages
client_msgs = client.get_messages()
print("Messages received by client:", client_msgs)

# Bidirectional Image Sending
sz = (800, 800)
client_image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)
server_image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)

# Client sends image to Server
client.send_img(client_image)
time.sleep(0.01)
server_received_image = server.get_img()
if server_received_image is not None:
    print("Server received image from Client")

# Server sends image to Client
server.send_img(server_image)
time.sleep(0.01)
client_received_image = client.get_img()
if client_received_image is not None:
    print("Client received image from Server")
```

### Real-time display example with remote streaming
Remote streaming allows to generate images on one PC, typically with a beefy GPU, and to show them on another one, which may not have a GPU. The streaming is handled via ZMQ and automatically compresses the images using jpeg compression.

Sender code example: generates an image and sends it to receiver. This is your backend server with GPU and we are emulating the image creation process by generating random arrays.
```python
import lunar_tools as lt
import numpy as np
sz = (800, 800)
ip = '127.0.0.1'

server = lt.ZMQPairEndpoint(is_server=True, ip='127.0.0.1', port='5556')

while True:
    test_image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)
    img_reply = server.send_img(test_image)
```


Reveiver code example: receives the image and renders it, this would be the frontend client e.g. a macbook.
```python
import lunar_tools as lt
sz = (800, 800)
ip = '127.0.0.1'

client = lt.ZMQPairEndpoint(is_server=False, ip='127.0.0.1', port='5556')
renderer = lt.Renderer(width=sz[1], height=sz[0])
while True:
    image = client.get_img()
    if image is not None:
        renderer.render(image)
```


## OSC
```python
import lunar_tools as lt
import numpy as np
import time
sender = lt.OSCSender('127.0.0.1')
receiver = lt.OSCReceiver('127.0.0.1')
    
for i in range(10):
    time.sleep(0.1)
    # sends two sinewaves to the respective osc variables
    val1 = (np.sin(0.5*time.time())+1)*0.5
    sender.send_message("/env1", val1)

receiver.get_all_values("/env1")
```

# Logging and terminal printing
```python
import lunar_tools as lt
logger = lt.LogPrint()  # No filename provided, will use default current_dir/logs/%y%m%d_%H%M
logger.print("white")
logger.print("red", "red")
logger.print("green", "green")
```

# Health status reporting via telegram
Obtain a bot here: https://docs.tracardi.com/qa/how_can_i_get_telegram_bot/
Next you will need to update your bashrc or bash_profile with the telegram bot env variables.


```bash
export TELEGRAM_BOT_TOKEN='XXX'
export TELEGRAM_CHAT_ID='XXX'
```

See the below example for the basic supported features
```python
health_reporter = lt.HealthReporter("Name of exhibit")
    
# in while loop, always report that exhibit is alive. This happens in a thread already.
for i in range(100): #while lopp
        health_reporter.report_alive()
    
# we can also just throw exceptions in there!
try: 
    x = y
except Exception as e:
    health_reporter.report_exception(e)
    
# or send something friendly!
health_reporter.report_message("friendly manual message")
```

# Devinfos
## Testing
pip install pytest

make sure you are in base folder
```python
python -m pytest lunar_tools/tests/
```

## Get requirements
```python
pipreqs . --force
```

