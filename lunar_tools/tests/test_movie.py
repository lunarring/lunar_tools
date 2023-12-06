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

def test_movie_concat():
    fps = 2
    list_fp_movies = []
    for k in range(4):
        fp_movie = f"/tmp/my_random_movie_{k}.mp4"
        list_fp_movies.append(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps)
        for _ in range(10):
            img = (np.random.rand(512, 1024, 3) * 255).astype(np.uint8)
            ms.write_frame(img)
        ms.finalize()

    for fp_movie in list_fp_movies:
        assert os.path.exists(fp_movie) and os.path.getsize(fp_movie) > 0

    fp_final = "/tmp/final_concatenated_movie.mp4"
    concatenate_movies(fp_final, list_fp_movies)
    assert os.path.exists(fp_final) and os.path.getsize(fp_final) > 0


def test_reading():
    fps = 2
    nmb_frames = 100
    fp_movie = f"/tmp/my_ordered_movie.mp4"
    ms = MovieSaver(fp_movie, fps=fps)
    for i in range(nmb_frames):
        img = (np.ones((256, 256, 3)) * i).astype(np.uint8)
        ms.write_frame(img)
    ms.finalize()
    
    mr = MovieReader(fp_movie)
    assert mr.nmb_frames == nmb_frames
        

def test_add_sound():
    fps = 24
    nmb_frames = 500
    fp_movie = f"/tmp/my_ordered_movie.mp4"
    ms = MovieSaver(fp_movie, fps=fps)
    for i in range(nmb_frames):
        img = (np.ones((256, 256, 3)) * i).astype(np.uint8)
        ms.write_frame(img)
    ms.finalize()
    
    fp_final = "/tmp/movie_with_sound.mp4"
    
    add_sound(fp_final, "/tmp/my_ordered_movie.mp4", 'lunar_tools/tests/myvoice.mp3')
    
    assert os.path.exists(fp_final) and os.path.getsize(fp_final) > 0

    



if __name__ == '__main__':
    unittest.main()
