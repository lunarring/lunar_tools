import asyncio

class RealTimeVoice:
    def __init__(self, instructions, on_ai_audio_complete, verbose=False):
        self.instructions = instructions
        self.on_ai_audio_complete = on_ai_audio_complete
        self.verbose = verbose
        self._audio_complete_pending = False
        self._loop = None

    def onAIAudioComplete(self):
        if self._audio_complete_pending and self._loop:
            self._audio_complete_pending = False
            asyncio.run_coroutine_threadsafe(self.on_ai_audio_complete(), self._loop)
