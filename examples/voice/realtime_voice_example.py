import time
from lunar_tools.realtime_voice import RealTimeVoice

_audio_complete_flag = {"called": False}

async def dummy_audio_complete_callback():
    print("Dummy audio complete callback invoked.")
    _audio_complete_flag["called"] = True

def run_realtime_voice_example():
    instructions = ("Interactive RealTimeVoice example. Commands:\n"
                    "s: Start, p: Pause, r: Resume, m: Mute/Unmute, i: Inject message, "
                    "u: Update instructions, q: Quit")
    rtv = RealTimeVoice(
        instructions=instructions,
        on_ai_audio_complete=dummy_audio_complete_callback,
        verbose=True
    )
    
    print(instructions)
    muted = False
    
    while True:
        cmd = input("Enter command (s=Start, p=Pause, r=Resume, m=Mute/Unmute, i=Inject, u=Update instructions, q=Quit): ").strip().lower()
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
        elif cmd == 'q':
            print("Stopping RealTimeVoice session and exiting...")
            rtv.stop()
            break
        else:
            print("Unknown command. Please try again.")
    
    print("RealtimeVoice interactive example completed.")

if __name__ == "__main__":
    run_realtime_voice_example()
    print(f"_audio_complete_flag: {_audio_complete_flag}")
