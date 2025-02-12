import asyncio
import threading
import time
import pytest

from lunar_tools import realtime_voice

def test_on_ai_audio_complete_callback_triggered():
    flag = {"called": False}

    async def dummy_callback():
        flag["called"] = True

    # Create an instance with the dummy callback.
    rtv = realtime_voice.RealTimeVoice(
        instructions="Test instructions",
        on_ai_audio_complete=dummy_callback,
        verbose=False
    )

    # Simulate that an audio complete event is pending.
    with rtv._audio_complete_lock:
        rtv._audio_complete_pending = True

    # Set up a new event loop and run it in a separate thread.
    loop = asyncio.new_event_loop()
    rtv._loop = loop

    def run_loop():
        loop.run_forever()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    # Invoke the onAIAudioComplete method.
    rtv.onAIAudioComplete()

    # Allow some time for the asynchronous callback to execute.
    time.sleep(0.3)

    # Stop the event loop.
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=1)
    loop.close()

    assert flag["called"] == True, "The on_ai_audio_complete callback should have been triggered."
