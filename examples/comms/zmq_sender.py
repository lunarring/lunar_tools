import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import ZMQPairEndpoint


def main():
    parser = argparse.ArgumentParser(description="Send sample ZMQ payloads.")
    parser.add_argument("--ip", default="127.0.0.1", help="Receiver IP (where the ZMQ server is bound).")
    parser.add_argument("--port", default="5556", help="Receiver port.")
    parser.add_argument("--count", type=int, default=5, help="Number of send cycles.")
    parser.add_argument("--interval", type=float, default=0.5, help="Delay between send cycles (seconds).")
    parser.add_argument("--timeout", type=float, default=2.0, help="Socket timeout in seconds.")
    args = parser.parse_args()

    sender = ZMQPairEndpoint(is_server=False, ip=args.ip, port=args.port, timeout=args.timeout)
    print(f"Sending {args.count} payload batches to tcp://{args.ip}:{args.port}")

    try:
        for idx in range(args.count):
            sender.send_json({"message": f"Hello from sender #{idx}!"})

            image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
            sender.send_img(image)

            sample_rate = 44100
            duration = 0.2  # seconds
            n_samples = int(sample_rate * duration)
            audio = np.random.randn(n_samples, 2).astype(np.float32) * 0.05
            sender.send_audio(audio, sample_rate)

            print(f"Sent batch {idx + 1}/{args.count}")
            time.sleep(args.interval)

        print("Done sending.")
    finally:
        sender.stop()


if __name__ == "__main__":
    main()
