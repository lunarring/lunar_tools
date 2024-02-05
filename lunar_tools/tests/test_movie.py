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
from lunar_tools.movie import fill_frames_linear_interpolate

def test_fill_frames_linear_interpolate():
    # Create a black and a white frame with smaller dimensions
    black_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    white_frame = np.ones((10, 10, 3), dtype=np.uint8) * 255

    # Number of frames to interpolate set to exactly 256
    nmb_frames = 256

    # Perform linear interpolation
    interpolated_frames = fill_frames_linear_interpolate(black_frame, white_frame, nmb_frames)

    # Check if the middle frame is gray (all channels have the value 127 or 128 due to rounding)
    middle_frame = interpolated_frames[len(interpolated_frames) // 2]
    middle_pixel = middle_frame[5, 5]  # Assuming a 10x10 image, this gets the middle pixel
    if not np.all(np.isclose(middle_pixel, 127, atol=1)):
        print(f"Middle pixel value is not 127, but {middle_pixel}")
        assert False, "The middle frame should be gray with a middle pixel value close to 127."


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
