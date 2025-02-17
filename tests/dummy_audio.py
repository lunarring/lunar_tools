import os

class DummyAudioRecorder:
    """
    A dummy audio recorder that mimics the interface of AudioRecorder.
    Instead of recording from a microphone, it immediately writes dummy audio data to a file.
    """
    def __init__(self, *args, **kwargs):
        self.is_recording = False
        self.output_filename = None

    def start_recording(self, output_filename=None, max_time=None):
        self.is_recording = True
        if output_filename is None:
            output_filename = "dummy_output.mp3"
        self.output_filename = output_filename

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            # Immediately write dummy data to simulate a recording.
            with open(self.output_filename, "wb") as f:
                f.write(b"DUMMY_AUDIO")
