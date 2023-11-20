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
import time
speech_detector = SpeechDetector(init_audiorecorder=True)
speech_detector.start_recording()
time.sleep(3)
speech_detector.stop_recording()
```


