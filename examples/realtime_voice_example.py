import time
from lunar_tools.presentation.audio_stack import (
    AudioStackConfig,
    AudioConversationController,
    bootstrap_audio_stack,
)
from lunar_tools.presentation.realtime_voice import RealTimeVoice

_audio_complete_flag = {"called": False}

async def dummy_audio_complete_callback():
    print("Dummy audio complete callback invoked.")
    _audio_complete_flag["called"] = True

def bootstrap_audio_controller() -> AudioConversationController:
    """
    Bootstrap the audio stack and return a controller that the demo can reuse for
    transcription + text-to-speech flows outside the realtime loop.
    """
    audio_config = AudioStackConfig(
        enable_playback=True,
        blocking_playback=True,
        include_elevenlabs=False,
    )
    services, synthesiser = bootstrap_audio_stack(audio_config)
    controller = AudioConversationController(services, synthesiser=synthesiser)
    return controller

def run_realtime_voice_example():
    controller = bootstrap_audio_controller()
    services = controller.services
    instructions = (
        "Interactive RealTimeVoice example. Commands:\n"
        "s: Start, p: Pause, r: Resume, m: Mute/Unmute, i: Inject message,\n"
        "u: Update instructions, c: Capture+Speak via AudioConversationController,\n"
        "t: Print realtime transcript buffer (when Deepgram is configured), q: Quit"
    )
    rtv = RealTimeVoice(
        instructions=instructions,
        on_ai_audio_complete=dummy_audio_complete_callback,
        verbose=True,
        audio_controller=controller,
    )
    
    print(instructions)
    muted = False
    realtime_service = services.realtime_transcription
    if realtime_service:
        realtime_service.start()
        print("Realtime transcription available: use command 't' to inspect buffered text.")
    
    while True:
        cmd = input(
            "Enter command "
            "(s=Start, p=Pause, r=Resume, m=Mute/Unmute, "
            "i=Inject, u=Update instructions, c=Capture+Speak, "
            "t=Realtime transcript, q=Quit): "
        ).strip().lower()
        if cmd == 's':
            print("Starting RealTimeVoice session...")
            rtv.start()
            time.sleep(1)
        elif cmd == 'p':
            print("Pausing RealTimeVoice session...")
            rtv.pause()
        elif cmd == 'r':
            print("Resuming RealTimeVoice session...")
            rtv.resume()
        elif cmd == 'm':
            if not muted:
                print("Muting microphone...")
                rtv.mute_mic()
                muted = True
            else:
                print("Unmuting microphone...")
                rtv.unmute_mic()
                muted = False
        elif cmd == 'i':
            test_message = input("Enter message to inject: ")
            print(f"Injecting message: {test_message}")
            rtv.inject_message(test_message)
        elif cmd == 'u':
            new_instructions = input("Enter new instructions: ")
            print(f"Updating instructions to: {new_instructions}")
            rtv.update_instructions(new_instructions)
        elif cmd == 'c':
            print("Capturing audio via AudioConversationController...")
            transcript = controller.capture_transcript(max_time=3)
            if transcript:
                print(f"Captured transcript: {transcript}")
                controller.speak(f"You just said: {transcript}")
            else:
                print("No transcript captured (short or silent recording).")
        elif cmd == 't':
            if realtime_service:
                print("Realtime transcript buffer:", realtime_service.transcript())
            else:
                print("Realtime transcription is not configured. Install the Deepgram extra.")
        elif cmd == 'q':
            print("Stopping RealTimeVoice session and exiting...")
            rtv.stop()
            break
        else:
            print("Unknown command. Please try again.")
    
    if realtime_service:
        realtime_service.stop()
    print("RealtimeVoice interactive example completed.")

if __name__ == "__main__":
    run_realtime_voice_example()
    print(f"_audio_complete_flag: {_audio_complete_flag}")
