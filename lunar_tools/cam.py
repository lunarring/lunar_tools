import numpy as np
import cv2
import time
import sys
import time
import glob
import threading
import time
import glob
import os
from lunar_tools.utils import get_os_type


class WebCam():
    def __init__(self, cam_id=0, shape_hw=(576,1024), do_digital_exposure_accumulation=False):
        """
        """
        self.do_mirror = False
        self.shift_colors = True
        self.do_digital_exposure_accumulation = do_digital_exposure_accumulation
        self.exposure_buf_size = 3
        self.cam_id = cam_id
        self.shape_hw = shape_hw
        self.img_last = np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
        self.frame_buffer = []
        self.device_ptr = 0
        self.sleep_time_thread = 0.001 
        self.smart_init()
        
        self.threader_active = True
        self.acquire_image = True
        self.thread = threading.Thread(target=self.threader_runfunc_cam, daemon=True)
        self.thread.start()
            
                
    def init_linux(self):
        device_paths = glob.glob('/dev/video*')
        if len(device_paths) == 0:
            raise ValueError("No cameras found")
        if self.cam_id == -1:
            device_ptr = device_paths[0]
        else:
            device_ptr = f'/dev/video{self.cam_id}'
        self.cam = cv2.VideoCapture(device_ptr)
        
        while True:
            self.release()
            if self.cam_id == -1 and len(device_paths) > 1:
                device_paths.remove(device_ptr)
                device_ptr = device_paths[0]
                print(f"smart_init: using device_ptr {device_ptr}")
            
            self.cam = cv2.VideoCapture(device_ptr)
            self.set_cap_props()
            _, img = self.cam.read()
            if img is not None:
                break
            print("release loop...")
            time.sleep(0.2)
        self.device_ptr = device_ptr
        
    def init_mac(self):
        self.cam = cv2.VideoCapture(self.cam_id, cv2.CAP_AVFOUNDATION)
        
    def release(self):
        self.threader_active = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cam.release()
        cv2.destroyAllWindows()
        cv2.VideoCapture(self.cam_id).release()    

    def smart_init(self):
        if get_os_type() == "Linux":
            self.init_linux()
        elif get_os_type() == "MacOS":
            self.init_mac()
        else:
            raise NotImplementedError("Only Linux and Mac supported.")
        self.set_cap_props()
        
    def set_cap_props(self):
        codec = 0x47504A4D  # MJPG
        self.cam.set(cv2.CAP_PROP_FPS, 30.0)
        self.cam.set(cv2.CAP_PROP_FOURCC, codec)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.shape_hw[1])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.shape_hw[0])
        
    def change_resolution(self, new_shape_hw):
        self.shape_hw = new_shape_hw
        
        self.acquire_image = False
        time.sleep(0.2)
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.shape_hw[1])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.shape_hw[0])
        
        self.acquire_image = True

    def set_focus_inf(self):
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cam.set(cv2.CAP_PROP_FOCUS, 0)
        
    def autofocus_enable(self):
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
    def autofocus_disable(self):
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
    def set_focus(self, value):
        self.autofocus_disable()
        self.cam.set(cv2.CAP_PROP_FOCUS, value)

    def threader_runfunc_cam(self):
        while self.threader_active:
            if self.acquire_image:
                img = self.get_raw_image()
                if img is None:
                    print("threader_runfunc_cam: bad img is None. trying to repair...")
                    self.cam.release()
                    cv2.VideoCapture(self.device_ptr).release()
                    self.smart_init()
                    time.sleep(1)
                else:
                    # accumulate frames over time and average to reduce noise
                    if self.do_digital_exposure_accumulation:
                        self.frame_buffer.append(img)
                        
                        if len(self.frame_buffer) > self.exposure_buf_size:
                            self.frame_buffer = self.frame_buffer[1:]
                            frame_average = np.array(self.frame_buffer).astype(np.float32).mean(0)
                            img = frame_average.astype(np.uint8)
                    
                    self.img_last = self.process_raw_image(img)
                    
                time.sleep(self.sleep_time_thread)

    def get_raw_image(self):
        _, img = self.cam.read()
        if img is None or img.size < 100:
            print("get_raw_image: fail, image is bad.")
            return
        return img

    def process_raw_image(self, img):
        if self.shift_colors:
            img = np.flip(img, axis=2)
        if self.do_mirror:
            img = np.flip(img, 1)
        return img
    
    def get_img(self):
        return self.img_last

        
if __name__ == "__main__":
    from PIL import Image
    cam = WebCam(cam_id=0, do_digital_exposure_accumulation=True)
    # ir = WebCam(cam_id=2)
    
    while True:
        img = cam.get_img()
        cv2.imshow('webcam', img[:,:,::-1])
        cv2.waitKey(1)
    

    
    
