import asyncio
import time
import threading

from lunar_tools import realtime_voice

def test_integration_on_ai_audio_complete():
    flag = {"called": False}

    async def dummy_callback():
        flag["called"] = True

    # Initialize RealTimeVoice with dummy instructions and the dummy callback.
    rtv = realtime_voice.RealTimeVoice(
        instructions="Integration Test",
        on_ai_audio_complete=dummy_callback,
        verbose=True,
    )

    # Simulate that the audio complete condition is pending.
    rtv._audio_complete_pending = True

    # Create a new event loop and assign it to the instance.
    loop = asyncio.new_event_loop()
    rtv._loop = loop

    # Run the event loop in a separate thread.
    def run_loop():
        loop.run_forever()
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    # Trigger the onAIAudioComplete to schedule the asynchronous callback.
    rtv.onAIAudioComplete()

    # Allow a short time for the callback to run.
    time.sleep(0.3)

    # Clean up the event loop.
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=1)
    loop.close()

    # Assert the callback was executed.
    assert flag["called"] == True, "Callback should have been executed"
