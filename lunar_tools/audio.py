#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
import wave
from pydub import AudioSegment
import threading


class AudioRecorder:
    def __init__(self, output_filename, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
        self.output_filename = output_filename
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def _record(self):
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        print("Recording...")
        self.frames = []

        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        print("Finished recording.")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(self.output_filename + ".wav", 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Convert to MP3
        wav_audio = AudioSegment.from_wav(self.output_filename + ".wav")
        mp3_filename = self.output_filename + ".mp3"
        wav_audio.export(mp3_filename, format="mp3")

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.thread = threading.Thread(target=self._record)
            self.thread.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.thread.join()

