import io
import sys
import pytest
from examples.fps_tracker_example import main

def test_fps_tracker_example_runs():
    # Capture the output of the main function.
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "FPS:" in output, "Output does not contain FPS display."

if __name__ == "__main__":
    pytest.main()