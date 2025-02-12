import os
import unittest
from lunar_tools.dummy_audio import DummyAudioRecorder
import lunar_tools.audio as audio_module

class TestDummyAudioRecorder(unittest.TestCase):
    def setUp(self):
        # Monkey-patch AudioRecorder with DummyAudioRecorder
        self.original_AudioRecorder = audio_module.AudioRecorder
        audio_module.AudioRecorder = DummyAudioRecorder

    def tearDown(self):
        # Revert the monkey-patch
        audio_module.AudioRecorder = self.original_AudioRecorder
        # Cleanup the dummy file if exists
        dummy_file = "dummy_test.mp3"
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

    def test_dummy_audio_recording_creates_file(self):
        from lunar_tools.audio import AudioRecorder  # this will be our DummyAudioRecorder now
        recorder = AudioRecorder()
        dummy_file = "dummy_test.mp3"
        recorder.start_recording(output_filename=dummy_file)
        recorder.stop_recording()

        # Check that file exists
        self.assertTrue(os.path.exists(dummy_file), "Dummy audio file was not created.")

        # Check the file contains the expected dummy data
        with open(dummy_file, "rb") as f:
            content = f.read()
        self.assertEqual(content, b"DUMMY_AUDIO", "Dummy audio file content is not as expected.")

if __name__ == "__main__":
    unittest.main()
