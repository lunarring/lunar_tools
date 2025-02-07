import asyncio
import time
import threading

from lunar_tools import realtime_voice

def test_on_ai_audio_complete_existence():
    # Verify that the module does NOT expose 'onAudioComplete'
    assert not hasattr(realtime_voice, "onAudioComplete"), "Module should not have onAudioComplete"
    # Verify that the RealTimeVoice class exposes the renamed 'onAIAudioComplete'
    assert hasattr(realtime_voice.RealTimeVoice, "onAIAudioComplete"), "RealTimeVoice should have onAIAudioComplete"

def test_on_ai_audio_complete_functionality():
    flag = {"called": False}

    async def dummy_callback():
        flag["called"] = True

    # Create an instance of RealTimeVoice with the dummy callback.
    rtv = realtime_voice.RealTimeVoice(
        instructions="Test instructions",
        on_ai_audio_complete=dummy_callback,
        verbose=True
    )
    # Simulate that audio has been produced.
    rtv._audio_complete_pending = True

    # Set up an event loop for the instance.
    loop = asyncio.new_event_loop()
    rtv._loop = loop
    # Start the loop in a separate thread.
    def run_loop():
        loop.run_forever()
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    # Call the onAIAudioComplete method to trigger the callback.
    rtv.onAIAudioComplete()
    # Allow a short time for the scheduled callback to run.
    time.sleep(0.2)
    # Clean up the loop.
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=1)
    loop.close()

    assert flag["called"] == True, "The on_ai_audio_complete callback should have been called."
