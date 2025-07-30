#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import base64
import io
from PIL import Image
import sys
import os

# Add the parent directory to the path so we can import lunar_tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunar_tools.image_gen import FluxKontextImageGenerator

def process_image(editor_data, prompt, seed):
    """Process the image from the editor through Flux Kontext"""
    
    if editor_data is None:
        return None, "Please draw or upload an image first"
    
    if not prompt.strip():
        return None, "Please enter a prompt"
    
    try:
        # Extract the composite image from the editor data
        if isinstance(editor_data, dict) and 'composite' in editor_data:
            image = editor_data['composite']
        else:
            # If it's just an image directly
            image = editor_data
            
        if image is None:
            return None, "No image found in editor"
        
        # Convert PIL image to base64 data URI
        if isinstance(image, Image.Image):
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA'):
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(image)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_data = buffered.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{img_base64}"
        else:
            return None, "Invalid image format"
        
        # Initialize Flux Kontext generator
        flux_kontext = FluxKontextImageGenerator()
        
        # Generate edited image
        edited_image = flux_kontext.generate(
            prompt=prompt,
            image_url=data_uri,
            seed=int(seed) if seed else None,
            guidance_scale=3.5
        )
        
        return edited_image, "Image processed successfully!"
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)  # For debugging
        return None, error_msg

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Flux Kontext Image Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Flux Kontext Image Editor")
        gr.Markdown("Draw or upload an image on the left, enter a prompt, and see the AI-edited result on the right!")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Image Editor")
                image_editor = gr.ImageEditor(
                    label="Draw or Upload Image",
                    type="pil",
                    image_mode="RGB",
                    height=400,
                    width=400
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### Flux Kontext Result")
                output_image = gr.Image(
                    label="Edited Result",
                    type="pil",
                    height=400,
                    width=400
                )
        
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="Describe how you want to modify the image (e.g. 'Change the colors to sunset colors', 'Add a mountain in the background')",
                    lines=2
                )
            with gr.Column(scale=1):
                seed_input = gr.Number(
                    label="Seed (optional)",
                    value=420,
                    precision=0
                )
                
        with gr.Row():
            process_btn = gr.Button("Generate Edit", variant="primary", size="lg")
            status_text = gr.Textbox(label="Status", interactive=False)
        
        # Set up the processing
        process_btn.click(
            fn=process_image,
            inputs=[image_editor, prompt_input, seed_input],
            outputs=[output_image, status_text]
        )
        
        # Example prompts
        gr.Markdown("""
        ### Example Prompts:
        - "Change the colors to sunset colors but keep the original shapes"
        - "Make it look like a watercolor painting"
        - "Add snow and winter atmosphere"
        - "Transform into a cyberpunk style"
        - "Make it look like an oil painting"
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1") 