import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lunar_tools.comms import OSCReceiver


def main():
    parser = argparse.ArgumentParser(description="Visualize incoming OSC values.")
    parser.add_argument("--ip", default="127.0.0.1", help="IP address to bind the OSC receiver to.")
    parser.add_argument("--port", type=int, default=8003, help="Port for the OSC receiver.")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the visualization grid.")
    parser.add_argument("--rows", type=int, default=2, help="Number of rows in the visualization grid.")
    parser.add_argument(
        "--height",
        type=int,
        default=300,
        help="Height of each visualization panel (pixels). Width is derived from columns.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=500,
        help="Width of each visualization panel (pixels). Height is derived from rows.",
    )
    args = parser.parse_args()

    receiver = OSCReceiver(ip_receiver=args.ip, port_receiver=args.port)
    receiver.start_visualization(shape_hw_vis=(args.height, args.width), nmb_cols_vis=args.cols, nmb_rows_vis=args.rows)

    print(f"Listening for OSC messages on {args.ip}:{args.port}. Press Ctrl+C to exit.")
    try:
        while True:
            receiver.show_visualization()
            time.sleep(0.01)
    except KeyboardInterrupt:
        receiver.stop()
        print("Visualization stopped.")


if __name__ == "__main__":
    main()
