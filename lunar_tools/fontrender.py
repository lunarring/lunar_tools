from PIL import Image, ImageDraw, ImageFont
import numpy as np

def add_text_to_image(
        img_input, 
        text, 
        align='center', 
        y_pos=0.5, 
        font_path='Arial.ttf', 
        font_size=20,
        width_fract=None
        ):
    # Create an image or load it based on the type of img_input
    if isinstance(img_input, tuple):
        # Create a new image with a white background
        image = Image.new('RGB', img_input, color=(255, 255, 255))
    elif isinstance(img_input, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(img_input)
    elif isinstance(img_input, Image.Image):
        # Use the provided PIL Image
        image = img_input
    else:
        raise ValueError("img_input must be a PIL.Image, numpy array, or a (width, height) tuple.")

    draw = ImageDraw.Draw(image)
    width, height = image.size

    if width_fract is not None:
        # Adjust font size based on width fraction
        font = ImageFont.truetype(font_path, font_size)
        text_width = draw.textlength(text, font=font)
        while text_width < width_fract * width and font_size < height:
            font_size += 1
            font = ImageFont.truetype(font_path, font_size)
            text_width = draw.textlength(text, font=font)
        font_size -= 1  # Decrease font size to fit in width

    font = ImageFont.truetype(font_path, font_size)
    text_width = draw.textlength(text, font=font)
    text_height = font_size  # Using font size as height

    if align == 'left':
        x = 0
    elif align == 'center':
        x = (width - text_width) // 2
    elif align == 'right':
        x = width - text_width
    else:
        raise ValueError("align must be 'left', 'center', or 'right'")
    
    y = int(y_pos * (height - text_height))

    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    
    return image

if __name__ == "__main__":
    # Creating new samples using the updated function
    image1 = add_text_to_image((400, 300), 'Center Aligned', 'center', 0.5, width_fract=0.8)
