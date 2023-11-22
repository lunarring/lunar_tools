import unittest
import os
import time
from pydub import AudioSegment
import sys
import string
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
from audio import AudioRecorder
from audio import SpeechDetector

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

class TestSpeechDetector(unittest.TestCase):
    def setUp(self):
        # Set up the SpeechDetector instance
        self.speech_detector = SpeechDetector()

    def test_translate_myvoice(self):
        # Test to ensure that translating 'myvoice.mp3' returns 'I am a butterfly'
        audio_file = 'lunar_tools/tests/myvoice.mp3'
        expected_transcript = 'I am a butterfly'
        result = self.speech_detector.translate(audio_file)

        # Strip punctuation from the result for comparison
        result = result.translate(str.maketrans('', '', string.punctuation))

        self.assertEqual(result, expected_transcript, f"Transcript does not match expected text. Got: {result}")


    def test_translate_non_existent_file(self):
        # Test to handle non-existent file error
        with self.assertRaises(FileNotFoundError):
            self.speech_detector.translate("non_existent_file.mp3")
if __name__ == '__main__':
    
    unittest.main()
