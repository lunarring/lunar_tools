import unittest
import os
import time
from pydub import AudioSegment
from audio import AudioRecorder
import sys
sys.path.append(os.path.abspath('../lunar_tools'))

class TestAudioRecorder(unittest.TestCase):
    """
    Test cases for the AudioRecorder class.
    """

    def test_recording_exists_and_nonzero(self):
        """
        Test that a 3-second recording creates a file and contains a nonzero signal.
        """
        recorder = AudioRecorder()
        test_filename = 'test_recording.mp3'

        # Start recording
        recorder.start_recording(output_filename=test_filename)
        time.sleep(3)  # Record for 3 seconds
        recorder.stop_recording()

        # Check if file exists
        self.assertTrue(os.path.exists(test_filename), "Recording file does not exist.")

        # Check if the recording is not silent (nonzero signal)
        recording = AudioSegment.from_mp3(test_filename)
        self.assertNotEqual(len(recording.get_array_of_samples()), 0, "Recording is silent.")

        # Clean up: remove the test file
        os.remove(test_filename)

    # Additional tests can be added here

if __name__ == '__main__':
    unittest.main()
