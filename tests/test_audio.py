import unittest
import os
import time
from pydub import AudioSegment
import sys
import string
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
from audio import AudioRecorder, Speech2Text, Text2SpeechOpenAI

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

class TestSpeech2Text(unittest.TestCase):
    def setUp(self):
        # Set up the Speech2Text instance
        self.speech_detector = Speech2Text()

    def test_translate_myvoice(self):
        # Test to ensure that translating 'myvoice.mp3' returns 'I am a butterfly'
        audio_file = 'tests/myvoice.mp3'
        expected_transcript = 'I am a butterfly'
        result = self.speech_detector.translate(audio_file)

        # Strip punctuation from the result for comparison
        result = result.translate(str.maketrans('', '', string.punctuation))

        self.assertEqual(result, expected_transcript, f"Transcript does not match expected text. Got: {result}")


    def test_translate_non_existent_file(self):
        # Test to handle non-existent file error
        with self.assertRaises(FileNotFoundError):
            self.speech_detector.translate("non_existent_file.mp3")
            
            
class TestText2SpeechOpenAI(unittest.TestCase):

    def setUp(self):
        # Initialize the Text2SpeechOpenAI instance with default parameters
        self.Text2SpeechOpenAI = Text2SpeechOpenAI()

    def test_change_voice_valid_voice(self):
        # Test changing the voice to a valid model
        new_voice = "echo"
        self.Text2SpeechOpenAI.change_voice(new_voice)
        self.assertEqual(self.Text2SpeechOpenAI.voice_model, new_voice)

    def test_change_voice_invalid_voice(self):
        # Test changing the voice to an invalid model
        new_voice = "invalid_voice"
        with self.assertRaises(ValueError):
            self.Text2SpeechOpenAI.change_voice(new_voice)

    def test_generate_speech_output_file(self):
        # Test generating speech and writing to a file
        test_text = "Hello world"
        test_output_filename = "test_output.mp3"
        self.Text2SpeechOpenAI.change_voice("echo")
        self.Text2SpeechOpenAI.generate(text=test_text, output_filename=test_output_filename)
        
        # Check if the file is created
        self.assertTrue(os.path.exists(test_output_filename))
        
        # Optional: Check if the file size seems reasonable (not empty)
        self.assertGreater(os.path.getsize(test_output_filename), 0)

        # Cleanup: Remove the created file
        os.remove(test_output_filename)

if __name__ == '__main__':
    
    unittest.main()
