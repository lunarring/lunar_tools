import unittest
import os
import time
from pydub import AudioSegment
import sys
import string
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
from movie import MovieSaver, concatenate_movies, add_sound, add_subtitles_to_video, MovieReader
import numpy as np

def test_movie_creation():
    fps = 2
    list_fp_movies = []
    for k in range(4):
        fp_movie = f"/tmp/my_random_movie_{k}.mp4"
        list_fp_movies.append(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps)
        for _ in range(30):
            img = (np.random.rand(512, 1024, 3) * 255).astype(np.uint8)
            ms.write_frame(img)
        ms.finalize()

    for fp_movie in list_fp_movies:
        assert os.path.exists(fp_movie) and os.path.getsize(fp_movie) > 0



if __name__ == '__main__':
    
    unittest.main()
