# based on https://github.com/jbaron34/torchwindow

import sys
import ctypes
import warnings
import numpy as np
import torch
from sys import platform
from PIL import Image
import cv2

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    import sdl2

from sdl2 import video
from OpenGL import GL as gl
from cuda import cudart as cu

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

class LunarRenderer:
    def __init__(self, width: int = 800, height: int = 600, 
                 gpu_id: int = 0,
                 window_title: str = "lunar_render_window"):
        
        self.window_title = window_title
        self.gpu_id = gpu_id
        self.width = width
        self.height = height
        
        if platform == "linux" or platform == "linux2":
            self.backend = 'gl'
            
            self.cuda_is_setup = False
            self.running = True
    
            self.sdl_setup()
            self.gl_setup()
            self.cuda_setup()
        elif platform == "darwin":
            self.backend = 'opencv'
        elif platform == "win32":
            self.backend = 'opencv'
            
    def sdl_setup(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO):
            raise SDLException(sdl2.SDL_GetError())

        self.sdl_window = sdl2.SDL_CreateWindow(
            self.window_title.encode(),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.width,
            self.height,
            sdl2.SDL_WINDOW_OPENGL,
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

    def gl_setup(self):
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

    def gl_draw_internal(self):
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

    def gl_step(self):
        event = sdl2.SDL_Event()
        
        # retrieve presses keyboard key codes
        key_states = sdl2.SDL_GetKeyboardState(None)
        
        # handle mouse presses
        mouse_posX, mouse_posY = ctypes.c_int(0), ctypes.c_int(0)
        mouse_buttonstate = sdl2.mouse.SDL_GetMouseState(ctypes.byref(mouse_posX), ctypes.byref(mouse_posX))
        # Print x and y as "native" ctypes values
        # print(x, y)
        # Print x and y as Python values
        # print(x.value, y.value)
        # print(buttonstate)

        if self.running:
            key_down = False
            while sdl2.SDL_PollEvent(ctypes.byref(event)):
                if (event.type == sdl2.SDL_WINDOWEVENT and event.window.event == sdl2.SDL_WINDOWEVENT_CLOSE):
                    self.running = False
                    self.gl_close()
    
                # Exit code
                if key_states[sdl2.SDL_SCANCODE_ESCAPE]:
                    self.running = False
                    self.gl_close()
                    sys.exit(0)
                    
                if key_states[sdl2.SDL_SCANCODE_A] and not key_down:
                    print('A key pressed')
                    key_down = True
                elif not key_states[sdl2.SDL_SCANCODE_A] and key_down:
                    print('A key released')
                    key_down = False
                    
            self.gl_draw_internal()

    def gl_render(self, image):
        
        # first check if input data types are valid
        if type(image) == torch.Tensor:
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
        
        # transpose X/Y for openGL consistency
        image = image.permute((1,0,2))
                
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
        self.gl_step()        
        
    def cv2_render(self, image):
        if type(image) == torch.Tensor:
            image = image.cpu().numpy()
        else:
            if type(image) == np.ndarray:
                pass
            elif type(image) == Image.Image:
                image = np.array(image)
            else:
                raise Exception('render function received input of unknown type')        
        
        cv2.imshow(self.window_title, image) 
        cv2_keycode = cv2.waitKey(1)
        if cv2_keycode == 27:  # exit code
            cv2.destroyAllWindows()
            sys.exit(0)
            
            
    def render(self, image):
        if self.backend == 'gl':
            self.gl_render(image)
        else:
            self.cv2_render(image)

    def gl_close(self):
        self.running = False
        sdl2.SDL_GL_DeleteContext(self.gl_context)
        sdl2.SDL_DestroyWindow(self.sdl_window)
        sdl2.SDL_Quit()

if __name__ == '__main__':
    
    sz = (1080, 1920)

    renderer = LunarRenderer(width=sz[1], height=sz[0])

    while True:
        # numpy array
        image = np.random.rand(sz[0],sz[1],4)*255
        
        # PIL array
        # image = Image.fromarray(np.uint8(np.random.rand(sz[0],sz[1],4)*255))
        
        # Torch tensors
        # image = torch.rand((sz[0],sz[1],4))*255
        # image = torch.rand((sz[0],sz[1]), device='cuda:0')*255
        image = torch.rand((sz[0],sz[1],3), device='cuda:0')*255
        # image = torch.rand((sz[0],sz[1],4), device='cuda:0')*255
        
        renderer.render(image)


    

