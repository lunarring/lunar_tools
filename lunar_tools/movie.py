import subprocess
import os
import numpy as np
from tqdm import tqdm
import cv2
from typing import List
import ffmpeg  # pip install ffmpeg-python. if error with broken pipe: conda update ffmpeg
from lunar_tools.utils import interpolate_linear
from PIL import Image
from typing import Union, List
import threading
import queue
from typing import Union
import time

class MovieSaver():
    def __init__(
            self,
            fp_out: str,
            fps: int = 24,
            shape_hw: List[int] = None,
            crf: int = 21,
            codec: str = 'libx264',
            preset: str = 'fast',
            pix_fmt: str = 'yuv420p',
            silent_ffmpeg: bool = True):
        r"""
        Initializes movie saver class - a human friendly ffmpeg wrapper.
        After you init the class, you can dump numpy arrays x into moviesaver.write_frame(x).
        Don't forget toi finalize movie file with moviesaver.finalize().
        Args:
            fp_out: str
                Output file name. If it already exists, it will be deleted.
            fps: int
                Frames per second.
            shape_hw: List[int, int]
                Output shape, optional argument. Can be initialized automatically when first frame is written.
            crf: int
                ffmpeg doc: the range of the CRF scale is 0–51, where 0 is lossless
                (for 8 bit only, for 10 bit use -qp 0), 23 is the default, and 51 is worst quality possible.
                A lower value generally leads to higher quality, and a subjectively sane range is 17–28.
                Consider 17 or 18 to be visually lossless or nearly so;
                it should look the same or nearly the same as the input but it isn't technically lossless.
                The range is exponential, so increasing the CRF value +6 results in
                roughly half the bitrate / file size, while -6 leads to roughly twice the bitrate.
            codec: int
                Number of diffusion steps. Larger values will take more compute time.
            preset: str
                Choose between ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.
                ffmpeg doc: A preset is a collection of options that will provide a certain encoding speed
                to compression ratio. A slower preset will provide better compression
                (compression is quality per filesize).
                This means that, for example, if you target a certain file size or constant bit rate,
                you will achieve better quality with a slower preset. Similarly, for constant quality encoding,
                you will simply save bitrate by choosing a slower preset.
            pix_fmt: str
                Pixel format. Run 'ffmpeg -pix_fmts' in your shell to see all options.
            silent_ffmpeg: bool
                Surpress the output from ffmpeg.
        """
        if len(os.path.split(fp_out)[0]) > 0:
            assert os.path.isdir(os.path.split(fp_out)[0]), "Directory does not exist!"

        self.fp_out = fp_out
        self.fps = fps
        self.crf = crf
        self.pix_fmt = pix_fmt
        self.codec = codec
        self.preset = preset
        self.silent_ffmpeg = silent_ffmpeg

        if os.path.isfile(fp_out):
            os.remove(fp_out)

        self.init_done = False
        self.nmb_frames = 0
        if shape_hw is None:
            self.shape_hw = [-1, 1]
        else:
            if len(shape_hw) == 2:
                shape_hw.append(3)
            self.shape_hw = shape_hw
            self.initialize()

        print(f"MovieSaver initialized. fps={fps} crf={crf} pix_fmt={pix_fmt} codec={codec} preset={preset}")

    def initialize(self):
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(self.shape_hw[1], self.shape_hw[0]), framerate=self.fps)
            .output(self.fp_out, crf=self.crf, pix_fmt=self.pix_fmt, c=self.codec, preset=self.preset)
            .overwrite_output()
            .compile()
        )
        if self.silent_ffmpeg:
            self.ffmpg_process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            self.ffmpg_process = subprocess.Popen(args, stdin=subprocess.PIPE)
        self.init_done = True
        self.shape_hw = tuple(self.shape_hw)
        print(f"Initialization done. Movie shape: {self.shape_hw}")

    def write_frame(self, out_frame: Union[np.ndarray, Image.Image]):
        r"""
        Function to dump a numpy array or a PIL Image as frame of a movie.
        Args:
            out_frame: Union[np.ndarray, Image.Image]
                If np.ndarray, it should be in np.uint8 format. Convert with np.astype(x, np.uint8).
                If Image.Image, it should be in 'RGB' mode.
                Dim 0: y
                Dim 1: x
                Dim 2: RGB (only for np.ndarray)
        """
        if isinstance(out_frame, Image.Image):
            out_frame = np.array(out_frame)
        assert len(out_frame.shape) == 3, "out_frame needs to be three dimensional, Y X C"
        assert out_frame.shape[2] == 3, f"need three color channels, but you provided {out_frame.shape[2]}."

        if not self.init_done:
            self.shape_hw = out_frame.shape
            self.initialize()

        assert self.shape_hw == out_frame.shape, f"You cannot change the image size after init. Initialized with {self.shape_hw}, out_frame {out_frame.shape}"

        # write frame
        self.ffmpg_process.stdin.write(
            out_frame
            .astype(np.uint8)
            .tobytes()
        )

        self.nmb_frames += 1

    def finalize(self):
        r"""
        Call this function to finalize the movie. If you forget to call it your movie will be garbage.
        """
        if self.nmb_frames == 0:
            print("You did not write any frames yet! nmb_frames = 0. Cannot save.")
            return
        self.ffmpg_process.stdin.close()
        self.ffmpg_process.wait()
        duration = int(self.nmb_frames / self.fps)
        print(f"Movie saved, {duration}s playtime, watch here: \n{self.fp_out}")


class MovieSaverThreaded:
    def __init__(self, fp_out: str, fps: int = 24, shape_hw: List[int] = None, crf: int = 21, codec: str = 'libx264', preset: str = 'fast', pix_fmt: str = 'yuv420p', silent_ffmpeg: bool = True):
        self.fp_out = fp_out
        self.fps = fps
        self.shape_hw = shape_hw
        self.crf = crf
        self.codec = codec
        self.preset = preset
        self.pix_fmt = pix_fmt
        self.silent_ffmpeg = silent_ffmpeg

        self.frame_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._write_frames)
        self.writer_thread.start()
        self.init_done = False
        self.nmb_frames = 0
        self.ffmpg_process = None
        self.finalized = False

    def initialize(self):
        args = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{self.shape_hw[1]}x{self.shape_hw[0]}', '-pix_fmt', 'rgb24', '-r', str(self.fps), '-i', '-', '-an', '-vcodec', 'mpeg4', self.fp_out]
        if self.silent_ffmpeg:
            self.ffmpg_process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            self.ffmpg_process = subprocess.Popen(args, stdin=subprocess.PIPE)
        self.init_done = True
        print(f"Initialization done. Movie shape: {self.shape_hw}")

    def _write_frames(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            self.ffmpg_process.stdin.write(frame.astype(np.uint8).tobytes())
            self.nmb_frames += 1

    def write_frame(self, out_frame: Union[np.ndarray, Image.Image]):
        if self.finalized:
            raise Exception("Cannot write to a finalized movie.")
        if isinstance(out_frame, Image.Image):
            out_frame = np.array(out_frame)
        assert len(out_frame.shape) == 3, "out_frame needs to be three dimensional, Y X C"
        assert out_frame.shape[2] == 3, f"need three color channels, but you provided {out_frame.shape[2]}."

        if not self.init_done:
            self.shape_hw = out_frame.shape
            self.initialize()

        assert self.shape_hw == out_frame.shape, f"You cannot change the image size after init. Initialized with {self.shape_hw}, out_frame {out_frame.shape}"

        self.frame_queue.put(out_frame)

    def finalize(self):
        self.frame_queue.put(None)
        while not self.frame_queue.empty():
            time.sleep(0.1)  # wait for all frames to be processed
            print(f"finalizing, writing {self.frame_queue.qsize()} remaining frames")
        self.writer_thread.join()
        self.ffmpg_process.stdin.close()
        self.ffmpg_process.wait()
        duration = int(self.nmb_frames / self.fps)
        print(f"Movie saved, {duration}s playtime, watch here: \n{self.fp_out}")
        self.finalized = True


def concatenate_movies(fp_final: str, list_fp_movies: List[str]):
    r"""
    Concatenate multiple movie segments into one long movie, using ffmpeg.

    Parameters
    ----------
    fp_final : str
        Full path of the final movie file. Should end with .mp4
    list_fp_movies : list[str]
        List of full paths of movie segments.
    """
    assert fp_final[-4] == ".", "fp_final seems to miss file extension: {fp_final}"
    for fp in list_fp_movies:
        assert os.path.isfile(fp), f"Input movie does not exist: {fp}"
        assert os.path.getsize(fp) > 100, f"Input movie seems empty: {fp}"

    if os.path.isfile(fp_final):
        os.remove(fp_final)

    # make a list for ffmpeg
    list_concat = []
    for fp_part in list_fp_movies:
        list_concat.append(f"""file '{fp_part}'""")

    # save this list
    fp_list = "tmp_move.txt"
    with open(fp_list, "w") as fa:
        for item in list_concat:
            fa.write("%s\n" % item)

    cmd = f'ffmpeg -f concat -safe 0 -i {fp_list} -c copy {fp_final}'
    subprocess.call(cmd, shell=True)
    os.remove(fp_list)
    if os.path.isfile(fp_final):
        print(f"concatenate_movies: success! Watch here: {fp_final}")
        
    
def add_sound(fp_final, fp_silentmovie, fp_sound):
    r"""
    Function to add sound to a silet video.
    """
    cmd = f'ffmpeg -i {fp_silentmovie} -i {fp_sound} -c copy -map 0:v:0 -map 1:a:0 {fp_final}'
    subprocess.call(cmd, shell=True)
    if os.path.isfile(fp_final):
        print(f"add_sound: success! Watch here: {fp_final}")
    
    
def add_subtitles_to_video(
        fp_input: str,
        fp_output: str,
        subtitles: list,
        fontsize: int = 50,
        font_name: str = "Arial",
        color: str = 'yellow'
    ):
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
    r"""
    Function to add subtitles to a video.
    
    Args:
        fp_input (str): File path of the input video.
        fp_output (str): File path of the output video with subtitles.
        subtitles (list): List of dictionaries containing subtitle information 
            (start, duration, text). Example:
            subtitles = [
                {"start": 1, "duration": 3, "text": "hello test"},
                {"start": 4, "duration": 2, "text": "this works"},
            ]
        fontsize (int): Font size of the subtitles.
        font_name (str): Font name of the subtitles.
        color (str): Color of the subtitles.
    """
    
    # Check if the input file exists
    if not os.path.isfile(fp_input):
        raise FileNotFoundError(f"Input file not found: {fp_input}")
    
    # Check the subtitles format and sort them by the start time
    time_points = []
    for subtitle in subtitles:
        if not isinstance(subtitle, dict):
            raise ValueError("Each subtitle must be a dictionary containing 'start', 'duration' and 'text'.")
        if not all(key in subtitle for key in ["start", "duration", "text"]):
            raise ValueError("Each subtitle dictionary must contain 'start', 'duration' and 'text'.")
        if subtitle['start'] < 0 or subtitle['duration'] <= 0:
            raise ValueError("'start' should be non-negative and 'duration' should be positive.")
        time_points.append((subtitle['start'], subtitle['start'] + subtitle['duration']))

    # Check for overlaps
    time_points.sort()
    for i in range(1, len(time_points)):
        if time_points[i][0] < time_points[i - 1][1]:
            raise ValueError("Subtitle time intervals should not overlap.")
    
    # Load the video clip
    video = VideoFileClip(fp_input)
    
    # Create a list to store subtitle clips
    subtitle_clips = []
    
    # Loop through the subtitle information and create TextClip for each
    for subtitle in subtitles:
        text_clip = TextClip(subtitle["text"], fontsize=fontsize, color=color, font=font_name)
        text_clip = text_clip.set_position(('center', 'bottom')).set_start(subtitle["start"]).set_duration(subtitle["duration"])
        subtitle_clips.append(text_clip)
    
    # Overlay the subtitles on the video
    video = CompositeVideoClip([video] + subtitle_clips)
    
    # Write the final clip to a new file
    video.write_videofile(fp_output)



class MovieReader():
    r"""
    Class to read in a movie.
    """

    def __init__(self, fp_movie):
        self.fp_movie = fp_movie
        self.video_player_object = cv2.VideoCapture(fp_movie)
        self.nmb_frames = int(self.video_player_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_movie = int(self.video_player_object.get(cv2.CAP_PROP_FPS))
        self.shape = [100, 100, 3]
        self.shape_is_set = False

    def get_next_frame(self):
        success, image = self.video_player_object.read()
        if success:
            if not self.shape_is_set:
                self.shape_is_set = True
                self.shape = image.shape
            return image
        else:
            return np.zeros(self.shape)


def interpolate_between_images(img1, img2, nmb_frames):
    list_imgs_interp = []  # Initialize the list to store interpolated images
    list_fracts_linblend = np.linspace(0, 1, nmb_frames)  # Generate linearly spaced fractions
    # Check if img1 or img2 are PIL Images and convert them to numpy arrays if so
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    for fract_linblend in list_fracts_linblend:
        img_blend = interpolate_linear(img1, img2, fract_linblend).astype(np.uint8)  # Corrected variable name from img0 to img1
        list_imgs_interp.append(img_blend)  # Append the blended image to the list
    return list_imgs_interp



def fill_up_frames_linear_interpolation(
        list_imgs: List[np.ndarray],
        fps_target: Union[float, int] = None,
        duration_target: Union[float, int] = None,
        nmb_frames_target: int = None):
    r"""
    Helper function to cheaply increase the number of frames given a list of images,
    using linear interpolation.
    The number of inserted frames between the images in the list will be 
    automatically adjusted so that the total of number
    of frames can be fixed precisely, using a random shuffling technique.

    Args:
        list_imgs: List[np.ndarray)
            List of images, between each image new frames will be inserted via linear interpolation.
        fps_target:
            OptionA: specify here the desired frames per second.
        duration_target:
            OptionA: specify here the desired duration of the transition in seconds.
        nmb_frames_target:
            OptionB: directly fix the total number of frames of the output.
    """

    # Sanity
    if nmb_frames_target is not None and fps_target is not None:
        raise ValueError("You cannot specify both fps_target and nmb_frames_target")
    if fps_target is None:
        assert nmb_frames_target is not None, "Either specify nmb_frames_target or nmb_frames_target"
    if nmb_frames_target is None:
        assert fps_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        assert duration_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        nmb_frames_target = fps_target * duration_target

    # Get number of frames that are missing
    nmb_frames_source = len(list_imgs) - 1
    nmb_frames_missing = nmb_frames_target - nmb_frames_source - 1

    if nmb_frames_missing < 1:
        return list_imgs

    if type(list_imgs[0]) == Image.Image:
        list_imgs = [np.asarray(l) for l in list_imgs]
    list_imgs_float = [img.astype(np.float32) for img in list_imgs]
    # Distribute missing frames, append nmb_frames_to_insert(i) frames for each frame
    mean_nmb_frames_insert = nmb_frames_missing / nmb_frames_source
    constfact = np.floor(mean_nmb_frames_insert)
    remainder_x = 1 - (mean_nmb_frames_insert - constfact)
    nmb_iter = 0
    while True:
        nmb_frames_to_insert = np.random.rand(nmb_frames_source)
        nmb_frames_to_insert[nmb_frames_to_insert <= remainder_x] = 0
        nmb_frames_to_insert[nmb_frames_to_insert > remainder_x] = 1
        nmb_frames_to_insert += constfact
        if np.sum(nmb_frames_to_insert) == nmb_frames_missing:
            break
        nmb_iter += 1
        if nmb_iter > 100000:
            print("add_frames_linear_interp: issue with inserting the right number of frames")
            break

    nmb_frames_to_insert = nmb_frames_to_insert.astype(np.int32)
    list_imgs_interp = []
    for i in range(len(list_imgs_float) - 1):
        img0 = list_imgs_float[i]
        img1 = list_imgs_float[i + 1]
        list_blended = interpolate_between_images(img0, img1, nmb_frames_to_insert[i]+2)
        if i == len(list_imgs_float) - 2:
            list_imgs_interp.extend(list_blended)
        else:
            list_imgs_interp.extend(list_blended[0:-1])

    return list_imgs_interp

if __name__ == "__main__":
    fps = 24
    list_fp_movies = []
    fp_movie = f"output.mp4"
    ms = MovieSaverThreaded(fp_movie, fps=fps)
    nmb_frames = 3000
    # imgs_all = (np.random.rand(nmb_frames, 512, 1024, 3) * 255).astype(np.uint8)
    for i in tqdm(range(nmb_frames)):
        # img = imgs_all[i]
        img = (np.random.rand(512, 1024, 3) * 255).astype(np.uint8)
        ms.write_frame(img)
    ms.finalize()
# 

if __name__ == "__main__z":
    fps = 2
    list_fp_movies = []
    for k in range(4):
        fp_movie = f"/tmp/my_random_movie_{k}.mp4"
        list_fp_movies.append(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps)
        for fn in tqdm(range(30)):
            img = (np.random.rand(512, 1024, 3) * 255).astype(np.uint8)
            ms.write_frame(img)
        ms.finalize()

    fp_final = "/tmp/my_concatenated_movie.mp4"
    concatenate_movies(fp_final, list_fp_movies)
