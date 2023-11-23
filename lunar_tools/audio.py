#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
import tempfile
from pydub import AudioSegment
import threading
import os
import time
from openai import OpenAI
from lunar_tools.logprint import LogPrint
import simpleaudio


class AudioRecorder:
    """
    A class to handle audio recording.

    Attributes:
        audio_format (pyaudio.paInt16): Format of the audio recording.
        channels (int): Number of audio channels.
        rate (int): Sampling rate.
        chunk (int): Number of frames per buffer.
        frames (list): List to hold audio frames.
        is_recording (bool): Flag to check if recording is in progress.
        audio (pyaudio.PyAudio): PyAudio instance.
        stream (pyaudio.Stream): Audio stream.
        output_filename (str): Output file name.
        logger: A logging instance. If None, a default logger will be used.
    """

    def __init__(
        self,
        audio_format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        chunk=1024,
        logger=None
    ):
        """
        Initialize the audio recorder.

        Args:
            audio_format (constant): Format of the audio recording.
            channels (int): Number of audio channels.
            rate (int): Sampling rate.
            chunk (int): Number of frames per buffer.
        """
        self.audio_format = audio_format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.output_filename = None
        self.logger = logger if logger else LogPrint()

    def _record(self, max_time=None):
        """
        Internal method to handle the audio recording process.
        Converts the recorded frames to MP3 format.

        Args:
            max_time (int, optional): Maximum recording time in seconds.
        """
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        self.logger.print("Recording...")
        self.frames = []
        start_time = time.time()
        while self.is_recording:
            if max_time and (time.time() - start_time) >= max_time:
                break
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        self.logger.print("Finished recording.")
        self.stream.stop_stream()
        self.stream.close()
        
        # Convert to MP3
        raw_data = b''.join(self.frames)
        wav_audio = AudioSegment(
            data=raw_data,
            sample_width=self.audio.get_sample_size(self.audio_format),
            frame_rate=self.rate,
            channels=self.channels
        )
        wav_audio.export(self.output_filename, format="mp3")

    def start_recording(self, output_filename=None, max_time=None):
        """
        Start the audio recording.

        Args:
            output_filename (str): The filename for the output file. If None, a temporary file is created.
            max_time (int, optional): Maximum recording time in seconds.
        """
        if not self.is_recording:
            self.is_recording = True
            if output_filename is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                self.output_filename = temp_file.name
                temp_file.close()
            else:
                if not output_filename.endswith('.mp3'):
                    raise ValueError("Output filename must have a .mp3 extension")
                self.output_filename = output_filename
            self.thread = threading.Thread(target=self._record, args=(max_time,))
            self.thread.start()

    def stop_recording(self):
        """
        Stop the audio recording and join the recording thread.
        """
        if self.is_recording:
            self.is_recording = False
            self.thread.join()


class Speech2Text:
    def __init__(self, client=None, logger=None, audio_recorder=None):
        """
        Initialize the Speech2Text with an OpenAI client, a logger, and an audio recorder.

        Args:
            client: An instance of OpenAI client. If None, it will be created using the OPENAI_API_KEY.
            logger: A logging instance. If None, a default logger will be used.
            audio_recorder: An instance of an audio recorder. If None, recording functionalities will be disabled.

        Raises:
            ValueError: If no OpenAI API key is found in the environment variables.
        """
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client
        self.logger = logger if logger else LogPrint()

        if audio_recorder is None:
            self.audio_recorder = AudioRecorder(logger=logger)
        else:
            self.audio_recorder = audio_recorder

    def start_recording(self, output_filename=None, max_time=None):
        """
        Start the audio recording.
        Args:
            output_filename (str): The filename for the output file. If None, a temporary file is created.
            max_time (int, optional): Maximum recording time in seconds.
        Raises:
            ValueError: If the audio recorder is not available.
        """
        if self.audio_recorder is None:
            raise ValueError("Audio recorder is not available")
        self.audio_recorder.start_recording(output_filename, max_time)

    def stop_recording(self):
        """
        Stop the audio recording.

        Raises:
            ValueError: If the audio recorder is not available.
        """
        if self.audio_recorder is None:
            raise ValueError("Audio recorder is not available")
        self.audio_recorder.stop_recording()
        return self.translate(self.audio_recorder.output_filename)

    def translate(self, audio_filepath):
        """
        Translate the audio file to text using OpenAI's translation model.

        Args:
            audio_filepath: The file path of the audio file to be translated.

        Returns:
            str: The transcribed text.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")
        with open(audio_filepath, "rb") as audio_file:
            transcript = self.client.audio.translations.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text


class Text2Speech:
    def __init__(
        self, 
        client=None, 
        logger=None, 
        text_source=None, 
        voice_model="nova", 
        sound_player=None
    ):
        """
        Initialize the Text2Speech with an OpenAI client, a logger, a text source, a default voice model, and optionally a sound player.
        """
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client
        self.logger = logger if logger else LogPrint()
        # Initialize the sound player only if provided
        self.sound_player = sound_player
        self.output_filename = None  # Initialize output filename
        self.voice_model = voice_model

    def play(self, text=None):
        """
        Play a generated speech file. Instantiates SoundPlayer if it's not already.
        """
        self.generate(text)
        if self.sound_player is None:
            self.sound_player = SoundPlayer()
        self.sound_player.play_sound(self.output_filename)

    def stop(self):
        """
        Stop the currently playing speech file.
        """
        if self.sound_player:
            self.sound_player.stop_sound()


    def generate(self, text, output_filename=None):
        """
        Generate speech from text.
    
        Args:
            text (str): The text to be converted into speech. 
            output_filename (str): The filename for the output file. If None, a default filename is used.
    
        Raises:
            ValueError: If the text source is not available.
        """
        if text is None or len(text)==0:
            raise ValueError("text is invalid!")
        text = text if text is not None else self.text_source.get_text()
    
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice_model,
            input=text
        )
        
        self.output_filename = output_filename if output_filename else "output_speech.mp3"
        response.stream_to_file(self.output_filename)
        self.logger.print(f"Generated speech saved to {self.output_filename}")


    def change_voice(self, new_voice):
        """
        Change the voice model for speech generation.

        Args:
            new_voice (str): The new voice model to be used.
        """
        if new_voice in self.list_available_voices():
            self.voice_model = new_voice
            self.logger.print(f"Voice model changed to {new_voice}")
        else:
            raise ValueError(f"Voice '{new_voice}' is not a valid voice model.")

    @staticmethod
    def list_available_voices():
        """
        List the available voice models.

        Returns:
            list: A list of available voice models.
        """
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class SoundPlayer:
    def __init__(self):
        self._play_thread = None
        self._playback_object = None

    def _play_sound_threaded(self, sound):
        self._playback_object = simpleaudio.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        self._playback_object.wait_done()  # Wait until sound has finished playing

    def play_sound(self, file_path):
        # Stop any currently playing sound
        self.stop_sound()

        # Load the sound file
        sound = AudioSegment.from_file(file_path)

        # Start a new thread for playing the sound
        self._play_thread = threading.Thread(target=self._play_sound_threaded, args=(sound,))
        self._play_thread.start()

    def stop_sound(self):
        if self._play_thread and self._play_thread.is_alive():
            if self._playback_object:
                self._playback_object.stop()
            self._play_thread.join()

    

#%% EXAMPLE USE        
if __name__ == "__main__":
    # audio_recorder = AudioRecorder()
    # audio_recorder.start_recording("myvoice.mp3")
    # time.sleep(3)
    # audio_recorder.stop_recording()
    
    # audio_recorder.start_recording("myvoice2.mp3")
    # time.sleep(3)
    # audio_recorder.stop_recording()
    
    
    # speech_detector = Speech2Text()
    # speech_detector.start_recording()
    # time.sleep(3)
    # translation = speech_detector.stop_recording()
    # print(f"translation: {translation}")
    
    # speech_detector.start_recording()
    # time.sleep(3)
    # translation = speech_detector.stop_recording()
    # print(f"translation: {translation}")
    
    # Example Usage
    text2speech = Text2Speech()
    text2speech.change_voice("nova")
    text2speech.play("test hello!")
    # player = SoundPlayer()
    # player.play_sound("/tmp/bla.mp3")
    # player.stop_sound()
