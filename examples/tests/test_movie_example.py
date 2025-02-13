import os
import unittest
import subprocess

class TestMovieExample(unittest.TestCase):
    def test_movie_example_runs(self):
        output_file = "output_example.mp4"
        # Run the example script
        result = subprocess.run(["python", "examples/movie_example.py"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=f"Script exited with error: {result.stderr}")
        
        self.assertTrue(os.path.exists(output_file), "Output movie file does not exist.")
        self.assertTrue(os.path.getsize(output_file) > 0, "Output movie file is empty.")
        # Clean up the generated file
        os.remove(output_file)

if __name__ == "__main__":
    unittest.main()
