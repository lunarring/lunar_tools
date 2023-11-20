# Installation
pip install git+https://github.com/lunarring/lunar_tools

# Audio
## AudioRecorder
```python
import lunar_tools as lt
import time
audio_recorder = lt.AudioRecorder()
audio_recorder = AudioRecorder()
audio_recorder.start_recording("myvoice.mp3")
time.sleep(3)
audio_recorder.stop_recording()
```

## SpeechDetector
```python
import lunar_tools as lt
import time
speech_detector = lt.SpeechDetector(init_audiorecorder=True)
speech_detector.start_recording()
time.sleep(3)
speech_detector.stop_recording()
```

# Image gen
## Generate Images with Dall-e-3
```python
dalle3 = Dalle3ImageGenerator()
image, revised_prompt = dalle3.generate("a beautiful red house with snow on the roof, a chimney with smoke")
```

