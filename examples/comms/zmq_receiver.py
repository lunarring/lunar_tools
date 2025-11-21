import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import ZMQPairEndpoint


def main():
    parser = argparse.ArgumentParser(description="Receive and inspect ZMQ payloads.")
    parser.add_argument("--ip", default="127.0.0.1", help="IP/interface to bind the receiver to.")
    parser.add_argument("--port", default="5556", help="Port to bind.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Socket timeout in seconds.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between poll loops.")
    args = parser.parse_args()

    receiver = ZMQPairEndpoint(is_server=True, ip=args.ip, port=args.port, timeout=args.timeout)
    print(f"Receiver ready on tcp://{args.ip}:{args.port}. Press Ctrl+C to stop.")

    try:
        while True:
            for message in receiver.get_messages():
                print(f"[JSON] {message}")

            frame = receiver.get_img()
            if frame is not None:
                print(f"[IMAGE] shape={frame.shape}, dtype={frame.dtype}")

            audio = receiver.get_audio()
            if audio is not None:
                print(
                    "[AUDIO] samples={}, sample_rate={}, channels={}, dtype={}".format(
                        audio['data'].shape, audio['sample_rate'], audio['channels'], audio['dtype']
                    )
                )

            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("Stopping receiver...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
