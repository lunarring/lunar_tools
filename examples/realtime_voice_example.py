import asyncio
import threading
import time

from lunar_tools.realtime_voice import RealTimeVoice

# Global flag to demonstrate that the dummy callback was invoked.
_audio_complete_flag = {"called": False}

async def dummy_audio_complete_callback():
    print("Dummy audio complete callback invoked.")
    _audio_complete_flag["called"] = True

def run_realtime_voice_example():
    instructions = "Provide a brief response with sarcasm."
    rtv = RealTimeVoice(
        instructions=instructions,
        on_ai_audio_complete=dummy_audio_complete_callback,
        verbose=True
    )

    # Start the realtime voice session.
    rtv.start()
    time.sleep(1)  # Allow background thread to initialize.

    # Demonstrate basic operations.
    print("Pausing RealTimeVoice...")
    rtv.pause()
    time.sleep(0.5)
    
    print("Resuming RealTimeVoice...")
    rtv.resume()
    time.sleep(0.5)

    print("Muting microphone...")
    rtv.mute_mic()
    time.sleep(0.5)

    print("Unmuting microphone...")
    rtv.unmute_mic()
    time.sleep(0.5)

    print("Injecting message...")
    rtv.inject_message("Hello, AI!")
    time.sleep(0.5)

    # Manually simulate an audio complete event.
    with rtv._audio_complete_lock:
        rtv._audio_complete_pending = True
    print("Simulating audio complete event.")
    rtv.onAIAudioComplete()
    time.sleep(1)

    # Stop the realtime voice session.
    print("Stopping RealTimeVoice session...")
    rtv.stop()

    print("RealtimeVoice example completed.")

if __name__ == "__main__":
    run_realtime_voice_example()
    print(f"_audio_complete_flag: {_audio_complete_flag}")
