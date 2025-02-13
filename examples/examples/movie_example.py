import numpy as np
from lunar_tools.movie import MovieSaver
import os

def main():
    output_file = "output_example.mp4"
    fps = 2
    # Define frame dimensions
    height, width = 100, 100
    # Initialize MovieSaver with defined shape_hw
    ms = MovieSaver(output_file, fps=fps, shape_hw=[height, width])
    
    # Create 10 random frames and write them into the movie
    for i in range(10):
        frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        ms.write_frame(frame)
    
    ms.finalize()
    
    if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
        print(f"Movie successfully saved to {output_file}")
    else:
        print("Movie saving failed.")

if __name__ == "__main__":
    main()
