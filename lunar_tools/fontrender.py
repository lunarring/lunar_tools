from PIL import Image, ImageDraw, ImageFont
import numpy as np
import platform
import os
import pygame

def add_text_to_image(
        img_input, 
        text, 
        align='center', 
        y_pos=0.5, 
        fp_font=None,
        font_size=20,
        min_width=0,
        max_width=1,
        font_color=(0, 0, 0)
        ):
    """
    Add text to an image with specified alignment, position, font, size, and style.

    Args:
        img_input (tuple, np.ndarray, Image.Image): Input image, can be a tuple of (width, height),
                                                    a numpy array, or a PIL Image.
        text (str): Text to be added to the image.
        align (str, optional): Text alignment on the image. Options: 'left', 'center', 'right'. 
                               Defaults to 'center'.
        y_pos (float, optional): Vertical position of text, as a fraction of image height. 
                                 Defaults to 0.5.
        fp_font (str, optional): File pointer to font. Defaults to 'Arial'.
        font_size (int, optional): Initial font size. Adjusts to fit min_width and max_width. 
                                   Defaults to 20.
        min_width (float, optional): Minimum width of text as a fraction of image width. 
                                     Defaults to 0.
        max_width (float, optional): Maximum width of text as a fraction of image width. 
                                     Defaults to 1.
        font_color (tuple, optional): Font color in RGB. Defaults to black (0, 0, 0).


    Returns:
        Image.Image: PIL Image with text added.
    """
    
    # Create an image or load it based on the type of img_input
    if isinstance(img_input, tuple):
        image = Image.new('RGB', img_input, color=(255, 255, 255))
    elif isinstance(img_input, np.ndarray):
        image = Image.fromarray(img_input)
    elif isinstance(img_input, Image.Image):
        image = img_input
    else:
        raise ValueError("img_input must be a PIL.Image, numpy array, or a (width, height) tuple.")
    
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    scaled_min_width = min_width * width
    scaled_max_width = max_width * width
    
    # Determine font path based on OS and style
    os_name = platform.system()
    font_extension = '.ttf'
    if os_name == "Darwin":  # macOS
        fp_font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    elif os_name == "Linux":
        fp_font = "/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf"
    elif os_name == "Windows":
        fp_font = f"C:\\Windows\\Fonts\\Arial.ttf"
    else:
        raise ValueError("Unsupported operating system for font path.")
        
    assert os.path.exists(fp_font), f"{fp_font} does not exist! "
    
    font = ImageFont.truetype(fp_font, font_size)
    text_width = draw.textlength(text, font=font)
    
    # Adjust font size to fit within min and max width constraints
    while (text_width < scaled_min_width or text_width > scaled_max_width) and font_size < height:
        if text_width < scaled_min_width:
            font_size += 1
        else:
            font_size -= 1
    
        font = ImageFont.truetype(fp_font, font_size)
        text_width = draw.textlength(text, font=font)
    
    # Calculate text position based on alignment
    text_height = font_size
    if align == 'left':
        x = 0
    elif align == 'center':
        x = (width - text_width) // 2
    elif align == 'right':
        x = width - text_width
    else:
        raise ValueError("align must be 'left', 'center', or 'right'")
    
    y = int(y_pos * (height - text_height))
    
    draw.text((x, y), text, font=font, fill=font_color)
    
    return image



class PopupInput:
    def __init__(self, window_size, font_size=None, font_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """
        Initialize PopupInput with window size, font size, font color, and background color.
        """
        self.window_size_subject_acronym = window_size
        self.font_size = font_size if font_size else window_size[1] // 3
        self.font_color = font_color
        self.bg_color = bg_color

    def get_input(self):
        """
        Display a popup window to get user input.
        """
        pygame.init()
        screen = pygame.display.set_mode(self.window_size_subject_acronym)
        pygame.display.set_caption('Subject info')

        running = True
        input_text = ""
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        running = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode

            screen.fill(self.bg_color)
            font = pygame.font.Font(None, self.font_size)
            text = font.render(input_text, 1, self.font_color)
            textpos = text.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
            screen.blit(text, textpos)

            pygame.display.flip()
        pygame.quit()
        return input_text



if __name__ == "__main__":

    # Creating new samples using the updated function
    image1 = add_text_to_image((700, 500), 'center', font_size=50)
