# based on https://github.com/jbaron34/torchwindow

import sys
import ctypes
import warnings
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
from sys import platform
from PIL import Image
import cv2
import random
from lunar_tools.utils import get_os_type
import pygame
import threading
import time

if get_os_type() == "Linux":
    try:
        from cuda import cudart as cu
    except ImportError:
        try:
            from cuda.bindings import runtime as cu
        except ImportError:
            raise ImportError("Could not import CUDA runtime. Please ensure cuda-python is properly installed.")
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning)
        import sdl2
        import sdl2.ext
    
    from sdl2 import video
    from OpenGL import GL as gl   

import logging

logger = logging.getLogger(__name__)

# CUDA/SDL Exception classes
class SDLException(Exception):
    pass

class CudaException(Exception):
    pass

class OpenGLException(Exception):
    pass

# Shaders
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
from OpenGL.GL.shaders import compileProgram, compileShader


VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

vec4 colors[3] = vec4[3](
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(0.0, 0.0, 1.0, 1.0)
);

void main() {
    gl_Position = positions[gl_VertexID];
    fragColor = colors[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec4 fragColor;
smooth in vec2 texcoords;

out vec4 outputColour;

uniform sampler2D texSampler;

void main()
{
    outputColour = texture(texSampler, texcoords);
}
"""


def create_shader_program():
    vertex_shader = compileShader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER)
    fragment_shader = compileShader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
    return compileProgram(vertex_shader, fragment_shader)

class PeripheralEvent():
    def __init__(self):
        self.keycode = -1
        self.mouse_button_state = -1
        self.mouse_posX = -1
        self.mouse_posY = -1

# keycode conversion function  SDL2 -> OpenCV convention
def sdl_to_cv2_keycode(sdl_keycode):
    # Map for common keys (alphabets and numbers)
    if sdl2.SDL_SCANCODE_A <= sdl_keycode <= sdl2.SDL_SCANCODE_Z:
        return ord(chr(sdl_keycode - sdl2.SDL_SCANCODE_A + ord('a')))
    elif sdl2.SDL_SCANCODE_1 <= sdl_keycode <= sdl2.SDL_SCANCODE_0:
        return ord(chr(sdl_keycode - sdl2.SDL_SCANCODE_1 + ord('1')))
    
    # Digits
        if sdl2.SDL_SCANCODE_1 <= sdl_keycode <= sdl2.SDL_SCANCODE_0:
            return ord(chr(sdl_keycode - sdl2.SDL_SCANCODE_1 + ord('1')))    
    
    special_keys_map = {
        sdl2.SDL_SCANCODE_RETURN: 13,     # Enter key
        sdl2.SDL_SCANCODE_ESCAPE: 27,     # Escape key
        sdl2.SDL_SCANCODE_BACKSPACE: 8,   # Backspace key
        sdl2.SDL_SCANCODE_TAB: 9,         # Tab key
        sdl2.SDL_SCANCODE_SPACE: 32,      # Space key
        sdl2.SDL_SCANCODE_F1: 0x700000,   # F1 key
        sdl2.SDL_SCANCODE_F2: 0x710000,   # F2 key
        sdl2.SDL_SCANCODE_RETURN: 13,     # Enter key
        sdl2.SDL_SCANCODE_ESCAPE: 27,     # Escape key
        sdl2.SDL_SCANCODE_BACKSPACE: 8,   # Backspace key
        sdl2.SDL_SCANCODE_TAB: 9,         # Tab key
        sdl2.SDL_SCANCODE_SPACE: 32,      # Space key
        sdl2.SDL_SCANCODE_F1: 0x700000,   # F1 key
        sdl2.SDL_SCANCODE_F2: 0x710000,   # F2 key
        sdl2.SDL_SCANCODE_F3: 0x720000,   # F3 key
        sdl2.SDL_SCANCODE_F4: 0x730000,   # F4 key
        # ... (Continue for other function keys F5 to F12)
        sdl2.SDL_SCANCODE_RIGHT: 0x270000,  # Right arrow key
        sdl2.SDL_SCANCODE_LEFT: 0x250000,   # Left arrow key
        sdl2.SDL_SCANCODE_DOWN: 0x280000,   # Down arrow key
        sdl2.SDL_SCANCODE_UP: 0x260000,     # Up arrow key        
        # Add other special keys here
        # ...
    }
    
    if sdl_keycode in special_keys_map:
        return special_keys_map.get(sdl_keycode, -1)
    elif sdl_keycode == -1:
        pass
    else:
        print('sdl_to_cv2_keycode -> unknown key code')
        return -1

class Renderer:
    def __init__(self, width: int = 1920, height: int = 1080, 
                 gpu_id: int = 0,
                 window_title: str = "lunar_render_window",
                 do_fullscreen: bool = False,
                 do_window_refresh_no_freeze = True,  # TODO: setting it to True might interfere with event polling, such as grabbing inputs from keyboard/mouse; so far only implemented for "gl" backend
                 backend = None,
                 display_id: int = None):
        
        self.window_title = window_title
        self.gpu_id = gpu_id
        self.width = width
        self.height = height
        self.do_fullscreen = do_fullscreen
        self.do_window_refresh_no_freeze = do_window_refresh_no_freeze
        self.display_id = display_id
        
        if backend is None:
            if get_os_type() == "Linux":
                self.backend = 'gl'
            else:
                self.backend = 'opencv'
        else:
            assert backend in ['gl', 'opencv', 'pygame', 'cudagl']
		
            if get_os_type() == "MacOS":
              if backend != 'opencv' and backend != 'pygame':
                backend = 'pygame'

            self.backend = backend
        
        if self.backend == 'cudagl':
            self.cuda_is_setup = False
            self.cudasdl_setup()
            self.cudagl_setup()
            self.cuda_setup()
            self.running = True
        elif self.backend == 'gl':
            self.sdl_setup()
            self.running = True
        elif self.backend == 'pygame':
            self.pygame_setup(self.display_id)
            
    def sdl_setup(self):
        # Initialize SDL2
        sdl2.ext.init()
        
        if self.do_fullscreen:
            flags_window = sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_FULLSCREEN
        else:
            flags_window = sdl2.SDL_WINDOW_OPENGL
        
        # Create an SDL2 window
        self.sdl_window = sdl2.SDL_CreateWindow(b"Image Viewer", 
                                            sdl2.SDL_WINDOWPOS_CENTERED, 
                                            sdl2.SDL_WINDOWPOS_CENTERED, 
                                            self.width, self.height, 
                                            flags_window)
        if not self.sdl_window:
            raise Exception("Could not create SDL window:", sdl2.SDL_GetError())
        
        # Create OpenGL context
        self.context = sdl2.SDL_GL_CreateContext(self.sdl_window)
        
        # Enable 2D textures
        gl.glEnable(gl.GL_TEXTURE_2D)
        
        # Generate a texture ID and bind it
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Event polling thread
        if self.do_window_refresh_no_freeze:
            self.running = True
            self.event_thread = threading.Thread(target=self.gl_poll_events_thread, daemon=True)
            self.event_thread.start()

    def cudasdl_setup(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO):
            raise SDLException(sdl2.SDL_GetError())

        if self.do_fullscreen:
            flags_window = sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_FULLSCREEN
        else:
            flags_window = sdl2.SDL_WINDOW_OPENGL

        self.sdl_window = sdl2.SDL_CreateWindow(
            self.window_title.encode(),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.width,
            self.height,
            flags_window,
        )
        
        if not self.sdl_window:
            raise SDLException(sdl2.SDL_GetError())

        # Force OpenGL 3.3 'core' context.
        # Must set *before* creating GL context!
        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MINOR_VERSION, 3)
        video.SDL_GL_SetAttribute(
            video.SDL_GL_CONTEXT_PROFILE_MASK, video.SDL_GL_CONTEXT_PROFILE_CORE
        )
        self.gl_context = sdl2.SDL_GL_CreateContext(self.sdl_window)
        
    def cudagl_setup(self):
        self.shader_program = create_shader_program()
        self.vao = gl.glGenVertexArrays(1)

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            self.width,
            self.height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def cuda_setup(self):
        if self.cuda_is_setup:
            return

        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise SDLException(sdl2.SDL_GetError())

        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise OpenGLException(
                "OpenGL context may be running on integrated graphics"
            )

        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to register opengl texture")

        self.cuda_is_setup = True

    def cudagl_draw_internal(self):
        gl.glUseProgram(self.shader_program)
        try:
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
        sdl2.SDL_GL_SwapWindow(self.sdl_window)

    def gl_get_events(self):
        event = sdl2.SDL_Event()
        
        # retrieve presses keyboard key codes
        key_states = sdl2.SDL_GetKeyboardState(None)
        
        # handle mouse presses
        mouse_posX, mouse_posY = ctypes.c_int(0), ctypes.c_int(0)

        pressed_key_code = -1
        mouse_buttonstate = -1
        if self.running:
            key_press_tracker_array = np.zeros((sdl2.SDL_NUM_SCANCODES,), dtype=np.int8)
            while sdl2.SDL_PollEvent(ctypes.byref(event)):
                mouse_buttonstate = sdl2.mouse.SDL_GetMouseState(ctypes.byref(mouse_posX), ctypes.byref(mouse_posY))
                
                if (event.type == sdl2.SDL_WINDOWEVENT and event.window.event == sdl2.SDL_WINDOWEVENT_CLOSE):
                    self.running = False
                    self.gl_close()
    
                # Exit code
                if key_states[sdl2.SDL_SCANCODE_ESCAPE]:
                    self.running = False
                    self.gl_close()
                    sys.exit(0)
                
                for key_code in range(sdl2.SDL_NUM_SCANCODES):
                    # key pressed
                    if key_states[key_code] and key_press_tracker_array[key_code] == 0:
                        key_press_tracker_array[key_code] == 1
                        pressed_key_code = key_code
                        
                    # key released
                    elif not key_states[key_code] and key_press_tracker_array[key_code] == 1:
                        key_press_tracker_array[key_code] == 0
            
            if self.backend == 'cudagl':
                self.cudagl_draw_internal()
            
        # convert key code to SDL2 convention
        pressed_key_code = sdl_to_cv2_keycode(pressed_key_code)
        
        peripheralEvent = PeripheralEvent()
        peripheralEvent.pressed_key_code = pressed_key_code
        peripheralEvent.mouse_button_state = mouse_buttonstate
        peripheralEvent.mouse_posX = mouse_posX.value
        peripheralEvent.mouse_posY = mouse_posY.value
        
        return peripheralEvent
    
    def gl_render(self, image):

        if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        else:
            if type(image) == np.ndarray:
                pass
            elif type(image) == Image.Image:
                image = np.array(image)
            else:
                raise Exception('render function received input of unknown type')        
                
        # clamp and set correct type
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        
        # reshape if there is a mismatched between supply image size and window size
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height))
            
        image = np.flip(image, axis=0).copy()
            
        # Load image data into the texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.width, self.height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)
        
        events = sdl2.ext.get_events()

        # Handle SDL2 events
        # events = sdl2.ext.get_events()
        # for event in events:
        #     if event.type == sdl2.SDL_QUIT:
        #         self.cleanup()
        #         return

        # Clear the screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        # Render the texture as a quad
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(-1.0, -1.0)
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(1.0, -1.0)
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(1.0, 1.0)
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(-1.0, 1.0)
        gl.glEnd()
        
        # Swap buffers
        sdl2.SDL_GL_SwapWindow(self.sdl_window)
        
        pressed_key_code = self.gl_get_events()
        return pressed_key_code        

    def cudagl_render(self, image):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "cudagl_render requires PyTorch. Install it via 'pip install torch' "
                "or 'pip install lunar-tools[torch]'."
            )

        # first check if input data types are valid
        if isinstance(image, torch.Tensor):
            if image.device.type == 'cpu':
                # force placement on GPU
                image = image.to(f'cuda:{self.gpu_id}')
        else:
            # cast as torch tensor / place on GPU
            if type(image) == np.ndarray:
                image = torch.from_numpy(image).to(f'cuda:{self.gpu_id}')
            elif type(image) == Image.Image:
                image = torch.from_numpy(np.array(image)).to(f'cuda:{self.gpu_id}')
            else:
                raise Exception('render function received input of unknown type')
                
        # bring to OpenGL-standard range
        image = image.float() / 255
                
        # clamp to the valid range
        image = torch.clamp(image, 0, 1)
        
        # resize 
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = torch.nn.functional.interpolate(image.permute([2,0,1])[None], size=(self.height, self.width))
            image = image[0].permute([1,2,0])
        
        # transpose X/Y for openGL consistency
        # image = image.permute((1,0,2))
                
        # check for number of channels
        if len(image.shape) == 3:
            if image.shape[2] == 3: # add rgbA channel
                ones = torch.ones(image.size(0), image.size(1), 1, device=image.device)
                image = torch.cat((image, ones), -1)
            elif image.shape[2] == 4: # correct number of channels -> pass
                pass
            else:
                raise Exception('render function received the wrong number of channels')
        elif len(image.shape) == 2:  # grayscale input  -> to RGBA
            image = torch.cat((image, )*4, -1)
        else:
            raise Exception('render function received the wrong number of channels')
            
        # do rendering
        if not self.running:
            return
        if not self.cuda_is_setup:
            self.cuda_setup()
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to get mapped array")
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            image.data_ptr(),
            4 * 4 * self.width,
            4 * 4 * self.width,
            self.height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to copy from tensor to texture")

        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to unmap graphics resource")
        pressed_key_code = self.gl_get_events()
        return pressed_key_code
        
    def cv2_render(self, image):
        if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        else:
            if type(image) == np.ndarray:
                pass
            elif type(image) == Image.Image:
                image = np.array(image)
            else:
                raise Exception('render function received input of unknown type')        
                
        # clamp and set correct type
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        
        # reshape if there is a mismatched between supply image size and window size
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height))
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow(self.window_title, image) 
        cv2_keycode = cv2.waitKey(1)
        if cv2_keycode == 27:  # exit code
            cv2.destroyAllWindows()
            sys.exit(0)
            
        if self.do_fullscreen:
            cv2.namedWindow(self.window_title,cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
        peripheralEvent = PeripheralEvent()
        peripheralEvent.pressed_key_code = cv2_keycode
        
        # not implemented
        peripheralEvent.mouse_button_state = -1
        peripheralEvent.mouse_posX = -1
        peripheralEvent.mouse_posY = -1            
            
        return peripheralEvent

    def pygame_setup(self, display_id=None):
        pygame.init()
        flags = (pygame.NOFRAME | pygame.RESIZABLE) if self.do_fullscreen else 0
        self.screen = pygame.display.set_mode((self.width, self.height), flags, display=display_id)
        pygame.display.set_caption(self.window_title)
        self.running = True

    def pygame_render(self, image):
        if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        else:
            if type(image) == np.ndarray:
                pass
            elif type(image) == Image.Image:
                image = np.array(image)
            else:
                raise Exception('render function received input of unknown type')
        
        # clamp and set correct type
        image = np.clip(image, 0, 255).astype(np.uint8)

        # reshape if there is a mismatched between supply image size and window size
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.rot90(image)  # Pygame requires rotation to match orientation
        image = np.flip(image, axis=0)

        surf = pygame.surfarray.make_surface(image)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

        pressed_key_code = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    pygame.quit()
                    sys.exit(0)
                pressed_key_code = event.key

        peripheralEvent = PeripheralEvent()
        peripheralEvent.pressed_key_code = pressed_key_code
        
        # not implemented
        peripheralEvent.mouse_button_state = -1
        peripheralEvent.mouse_posX = -1
        peripheralEvent.mouse_posY = -1
            
        return peripheralEvent
            
    def render(self, image):
        if self.backend == 'cudagl':
            peripheralEvent = self.cudagl_render(image)
        elif self.backend == 'gl':
            peripheralEvent = self.gl_render(image)
        elif self.backend == 'opencv':
            peripheralEvent = self.cv2_render(image)
        elif self.backend == 'pygame':
            peripheralEvent = self.pygame_render(image)
        return peripheralEvent

    def gl_close(self):
        self.running = False
        sdl2.SDL_GL_DeleteContext(self.gl_context)
        sdl2.SDL_DestroyWindow(self.sdl_window)
        sdl2.SDL_Quit()
        
    def gl_poll_events_thread(self):
        # Poll SDL events in a separate thread
        while self.running:
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    self.cleanup()
                    exit(0)
            time.sleep(1.0)  # Prevent busy-looping           


class GridRenderer():
    def __init__(self, nmb_rows, nmb_cols, shape_hw, backend=None, window_title='lunar_render_window'):
        """
        nmb_rows: Number of tiles in vertical direction
        nmb_cols: Number of tiles in horizontal direction
        shape_hw: (H,W) = tuple (height,width) of individual tile
        """
        self.H = shape_hw[0]
        self.W = shape_hw[1]
        self.nmb_rows = nmb_rows
        self.nmb_cols = nmb_cols
        self.canvas = np.zeros((nmb_rows, nmb_cols, shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
        self.renderer = Renderer(width=shape_hw[1]*nmb_cols, height=shape_hw[0]*nmb_rows, backend=backend, window_title=window_title)
        
    def update(self, tiles):
        """
        Concatenate image tiles into one large canvas.
    
        :param tiles: NumPy array of shape nmb_rowsxnmb_colsxHxW*C
                       nmb_rows: Number of tiles in vertical direction
                       nmb_cols: Number of tiles in horizontal direction
                       H: Height of each tile
                       W: Width of each tile
                       C: Number of RGB channels
        :return: NumPy array representing the large canvas with shape (nmb_rows*H)x(nmb_cols*W)
        """
        if isinstance(tiles, list):
            tiles = self.list_to_tiles(tiles)
        nmb_rows, nmb_cols, H, W, C = tiles.shape
        fail_msg = 'GridRenderer->inject_tiles: tiles shape inconsistent with initialization'
        assert (nmb_rows == self.nmb_rows) and (nmb_cols == self.nmb_cols), print(fail_msg)
        assert (H == self.H) and (W == self.W), print(fail_msg)
        
        # Reshape and transpose to bring tiles next to each other
        self.canvas = tiles.transpose(0, 2, 1, 3, 4).reshape(nmb_rows*H, nmb_cols*W, C)
        
    def list_to_tiles(self, list_images):
        """
        Reshape image tiles from list to tensor.
    
        :param list_images: list of images of shape H*W*3
        :return: NumPy array of shape nmb_rowsxnmb_colsxHxW*C
        """        
        
        grid_input = np.zeros((self.nmb_rows, self.nmb_cols, self.H, self.W, 3), dtype=np.uint8)
        for m in range(self.nmb_rows):
            for n in range(self.nmb_cols):
                if m*self.nmb_cols + n < len(list_images):
                    grid_input[m,n,:,:,:] = list_images[m*self.nmb_cols + n]
        return grid_input        

    def render(self):
        """
        Render canvas abd find the index of the tile given a mouse click pixel coordinate on the 2D canvas.
        :return: A tuple (m, n) representing the tile index in the range (0..nmb_rows, 0..nmb_cols).
        """
        
        peripheralEvent = self.renderer.render(self.canvas)
        
        if peripheralEvent.mouse_button_state > 0:
            x = peripheralEvent.mouse_posX
            y = peripheralEvent.mouse_posY
            
            m = y // self.H
            n = x // self.W
            return m, n
        else:
            return -1, -1
        
if __name__ == '__main__z':
    import time
    nmb_rows = 2
    nmb_cols = 2
    shape_hw = (200, 300)
    gridrenderer = GridRenderer(nmb_rows, nmb_cols, shape_hw)


    while True:
        time.sleep(0.1)
        list_imgs = []
        for _ in range(nmb_rows * nmb_cols):
            img = Image.new('RGB', shape_hw[::-1], (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            list_imgs.append(img)
        gridrenderer.update(list_imgs)
        gridrenderer.render()



if __name__ == '__main__':
    
    sz = (512, 1024)
    renderer = Renderer(width=sz[1], height=sz[0], backend='gl', do_fullscreen=False)
    
    while True:
        # numpy array
        image = np.random.rand(sz[0]//4,sz[1]//4,3)*255
        image[:50,:,:] = 0
        image[:,:50,:] = 0
        
        # PIL array
        # image = Image.fromarray(np.uint8(np.random.rand(sz[0],sz[1],4)*255))
        
        # Torch tensors
        # image = torch.rand((sz[0]//2,sz[1]//2,4))*255
        # image = torch.rand((sz[0],sz[1]), device='cuda:0')*255
        # image = torch.rand((sz[0],sz[1],3), device='cuda:0')*255
        # image = torch.rand((sz[0],sz[1],4), device='cuda:0')*255
        renderer.render(image)


    