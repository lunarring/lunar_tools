import asyncio
import numpy as np
import time

from lunar_tools.cam import WebCam

# Dummy implementation of get_raw_image to return a fixed dummy image.
def dummy_get_raw_image(self):
    # Return a dummy image that is not all zeros.
    dummy_img = np.full((self.shape_hw[0], self.shape_hw[1], 3), 127, dtype=np.uint8)
    return dummy_img

async def test_async_capture_loop_updates():
    # Monkey-patch get_raw_image.
    original_get_raw_image = WebCam.get_raw_image
    WebCam.get_raw_image = dummy_get_raw_image

    loop = asyncio.get_running_loop()
    cam = WebCam(loop=loop)
    
    # Allow the async capture loop to run for a short while.
    await asyncio.sleep(0.1)
    
    # Check that img_last has been updated (it should not be all zeros).
    assert np.any(cam.img_last != 0), "img_last was not updated by the async capture loop."
    
    # Check that camera_fps is set (should be > 0).
    assert cam.camera_fps > 0, "camera_fps was not updated by the async capture loop."
    
    # Clean up.
    await cam.stop()
    WebCam.get_raw_image = original_get_raw_image

async def test_async_stop():
    # Monkey-patch get_raw_image.
    original_get_raw_image = WebCam.get_raw_image
    WebCam.get_raw_image = dummy_get_raw_image

    loop = asyncio.get_running_loop()
    cam = WebCam(loop=loop)
    await asyncio.sleep(0.1)
    
    # Stop the camera async loop.
    await cam.stop()
    
    # Wait briefly to ensure the capture loop has terminated.
    await asyncio.sleep(0.05)
    
    # Verify that the async loop has stopped.
    assert cam.threader_active == False, "Camera async loop did not stop."
    
    WebCam.get_raw_image = original_get_raw_image

def main():
    asyncio.run(test_async_capture_loop_updates())
    asyncio.run(test_async_stop())
    print("All async camera tests passed.")

if __name__ == "__main__":
    main()