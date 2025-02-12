import asyncio
import threading
import time
import pytest

from lunar_tools.realtime_voice import RealTimeVoice

def test_realtime_voice_audio_complete_callback():
    # Flag to verify callback invocation.
    called_flag = {"called": False}

    async def dummy_audio_complete_callback():
        called_flag["called"] = True

    # Instantiate RealTimeVoice with the dummy callback.
    rtv = RealTimeVoice(
        instructions="Test instructions",
        on_ai_audio_complete=dummy_audio_complete_callback,
        verbose=False
    )

    # Set up an event loop to support asynchronous callbacks.
    loop = asyncio.new_event_loop()
    rtv._loop = loop

    def run_loop():
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    # Simulate an audio complete event.
    with rtv._audio_complete_lock:
        rtv._audio_complete_pending = True

    rtv.onAIAudioComplete()

    # Allow some time for the asynchronous callback to execute.
    time.sleep(0.3)

    # Clean up: stop the event loop.
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    loop.close()

    assert called_flag["called"] == True, "The on_ai_audio_complete callback should have been triggered."
