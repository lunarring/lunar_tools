import os
import tempfile

class DummyAudioRecorder:
    """
    A dummy audio recorder that mimics the interface of AudioRecorder.
    Instead of recording from a microphone, it immediately writes dummy audio data to a file.
    """
    def __init__(self, *args, **kwargs):
        self.is_recording = False
        self.output_filename = None
        self._temp_files = []

    def start_recording(self, output_filename=None, max_time=None):
        self.is_recording = True
        if output_filename is None:
            temp_file = tempfile.NamedTemporaryFile(prefix="lunar_dummy_recording_", suffix=".mp3", delete=False)
            temp_file.close()
            output_filename = temp_file.name
            self._temp_files.append(output_filename)
        self.output_filename = output_filename

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            # Immediately write dummy data to simulate a recording.
            with open(self.output_filename, "wb") as f:
                f.write(b"DUMMY_AUDIO")
            if self.output_filename in self._temp_files:
                try:
                    os.remove(self.output_filename)
                finally:
                    self._temp_files.remove(self.output_filename)
