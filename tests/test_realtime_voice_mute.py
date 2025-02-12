import asyncio
import pytest
from lunar_tools.realtime_voice import RealTimeVoice

async def dummy_callback(*args, **kwargs):
    pass

def test_mute_unmute():
    rtv = RealTimeVoice(
        instructions="Test instructions",
        on_user_transcript=dummy_callback,
        on_ai_transcript=dummy_callback,
        on_ai_audio_complete=dummy_callback,
        verbose=False
    )
    # Initially, microphone should not be muted
    assert rtv._mic_muted == False

    # Mute mic and test the flag
    rtv.mute_mic()
    assert rtv._mic_muted == True

    # Unmute mic and test the flag again
    rtv.unmute_mic()
    assert rtv._mic_muted == False
