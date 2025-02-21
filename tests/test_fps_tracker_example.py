import sys
import os
import subprocess
import re

def test_fps_tracker_example_output():
    # Run the fxp_tracker_example.py script and capture its output.
    result = subprocess.run([sys.executable, os.path.join("examples", "fps_tracker_example.py")],
                            capture_output=True, text=True)
    output = result.stdout

    # Verify that all five segments are reported in the output.
    for i in range(1, 6):
        pattern = rf"Segment {i}: \d+\.\d+ms"
        if not re.search(pattern, output):
            raise AssertionError(f"Output does not contain expected segment info: Segment {i}")

if __name__ == "__main__":
    test_fps_tracker_example_output()
    print("Test passed.")