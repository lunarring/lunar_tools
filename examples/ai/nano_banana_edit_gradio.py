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

from lunar_tools.image_gen import NanoBananaEditImageGenerator


PRESET_DIMENSIONS = {
    "square_hd": (1024, 1024),
    "square": (768, 768),
    "portrait_4_3": (768, 1024),
    "portrait_16_9": (576, 1024),
    "landscape_4_3": (1024, 768),
    "landscape_16_9": (1024, 576),
}


def _pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if fmt == "PNG" else "RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def _extract_editor_images(editor_data, use_mask: bool):
    if editor_data is None:
        return None, None

    # Gradio ImageEditor returns a dict with 'composite' and optionally 'mask'
    if isinstance(editor_data, dict):
        composite = editor_data.get("composite")
        mask = editor_data.get("mask") if use_mask else None
    else:
        composite = editor_data
        mask = None

    if composite is None:
        return None, None

    # Ensure composite is RGB for compression to JPEG/PNG
    if isinstance(composite, Image.Image):
        # Prefer JPEG for main image
        if composite.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", composite.size, (255, 255, 255))
            if composite.mode == "RGBA":
                bg.paste(composite, mask=composite.split()[-1])
            else:
                bg.paste(composite)
            composite = bg
        elif composite.mode != "RGB":
            composite = composite.convert("RGB")
        image_data_uri = _pil_to_data_uri(composite, fmt="JPEG")
    else:
        return None, None

    mask_data_uri = None
    if use_mask and isinstance(mask, Image.Image):
        # Convert mask to binary PNG: white=edit, black=keep
        if mask.mode != "L":
            mask = mask.convert("L")
        # Normalize mask to 0/255
        mask = mask.point(lambda p: 255 if p > 0 else 0)
        mask_data_uri = _pil_to_data_uri(mask, fmt="PNG")

    return image_data_uri, mask_data_uri

def _get_editor_composite_pil(editor_data):
    """Return the editor's composite as a PIL RGB image, if available."""
    img = None
    if isinstance(editor_data, dict) and isinstance(editor_data.get("composite"), Image.Image):
        img = editor_data["composite"]
    elif isinstance(editor_data, Image.Image):
        img = editor_data
    if img is None:
        return None
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _build_gallery(editor_data, images_state):
    items = []
    editor_img = _get_editor_composite_pil(editor_data)
    if editor_img is not None:
        items.append(editor_img)
    for it in (images_state or []):
        items.append(it.get("annotated") or it.get("source"))
    return items


def _resolve_dimensions(image_size, custom_w, custom_h):
    """Return integer width/height for the selected preset or custom entry."""
    if image_size == "custom":
        width = custom_w if custom_w not in (None, "") else None
        height = custom_h if custom_h not in (None, "") else None
        try:
            width = int(width) if width is not None else None
            height = int(height) if height is not None else None
        except (TypeError, ValueError):
            raise ValueError("Custom width and height must be integers")
        if width is not None and width <= 0:
            raise ValueError("Custom width must be greater than 0")
        if height is not None and height <= 0:
            raise ValueError("Custom height must be greater than 0")
        return width, height

    dims = PRESET_DIMENSIONS.get(image_size)
    if dims:
        return dims
    return None, None


def process_images(editor_data, files, prompt, seed, image_size, custom_w, custom_h, images_state, current_index):
    if not prompt or not prompt.strip():
        return None, "Please enter a prompt"

    gen = NanoBananaEditImageGenerator()

    base_uri = None
    mask_uri = None
    editor_supplied = False
    seen_uris = set()
    reference_uris = []

    try:
        ed_uri, ed_mask = _extract_editor_images(editor_data, True)
        if ed_uri:
            base_uri = ed_uri
            mask_uri = ed_mask
            editor_supplied = True
            seen_uris.add(ed_uri)
    except Exception:
        # Editor might be empty or not ready; fall through to other sources
        mask_uri = None

    def _pil_item_to_uri(item):
        if not isinstance(item, Image.Image):
            return None
        return _pil_to_data_uri(item, fmt="JPEG")

    if not editor_supplied and current_index not in (None, "") and images_state:
        try:
            idx = int(current_index)
            if 0 <= idx < len(images_state):
                base_item = images_state[idx]
                comp = base_item.get("annotated") or base_item.get("source")
                candidate_uri = _pil_item_to_uri(comp)
                if candidate_uri:
                    base_uri = candidate_uri
                    seen_uris.add(candidate_uri)
        except (ValueError, TypeError):
            pass

    try:
        for item in (images_state or []):
            comp = item.get("annotated") or item.get("source")
            comp_uri = _pil_item_to_uri(comp)
            if comp_uri is None:
                continue
            if base_uri is None:
                base_uri = comp_uri
                seen_uris.add(comp_uri)
                continue
            if comp_uri == base_uri or comp_uri in seen_uris:
                continue
            reference_uris.append(comp_uri)
            seen_uris.add(comp_uri)
    except Exception as e:
        return None, f"File processing error: {e}"

    if base_uri:
        seen_uris.add(base_uri)
    if editor_supplied and mask_uri:
        ordered_image_uris = [base_uri] + reference_uris if base_uri else reference_uris
    else:
        ordered_image_uris = reference_uris + ([base_uri] if base_uri else [])

    if not ordered_image_uris:
        return None, "Please add at least one image (upload or draw)"

    try:
        resolved_width, resolved_height = _resolve_dimensions(image_size, custom_w, custom_h)
    except ValueError as err:
        return None, str(err)

    if image_size == "custom" and (resolved_width is None or resolved_height is None):
        return None, "Please provide both custom width and height"

    try:
        resolved_seed = int(seed) if seed not in (None, "") else None
    except (TypeError, ValueError):
        return None, "Seed must be an integer"

    requested_size = None if image_size == "custom" else image_size
    log_width = resolved_width if resolved_width is not None else "-"
    log_height = resolved_height if resolved_height is not None else "-"
    log_seed = resolved_seed if resolved_seed is not None else "-"

    try:
        mask_flag = 'yes' if (editor_supplied and mask_uri) else 'no'
        input_count = len(ordered_image_uris)
        print(
            f"[Demo] Sending request: input_images={input_count}, mask={mask_flag}, "
            f"size={requested_size or 'custom'}, w={log_width}, h={log_height}, seed={log_seed}"
        )
        gen_args = dict(
            prompt=prompt,
            mask_url=mask_uri if editor_supplied else None,
            seed=resolved_seed,
            num_images=1,
            image_size=requested_size,
        )
        if image_size == "custom":
            gen_args["width"] = resolved_width
            gen_args["height"] = resolved_height
        if len(ordered_image_uris) == 1:
            gen_args["image_url"] = ordered_image_uris[0]
        else:
            gen_args["image_urls"] = ordered_image_uris
        out = gen.generate(**gen_args)
        print("[Demo] Received one output image from API")
        if isinstance(out, Image.Image) and out.mode != "RGB":
            out = out.convert("RGB")
        size_label = None
        if resolved_width is not None and resolved_height is not None:
            size_label = f"{resolved_width}x{resolved_height}"
        elif requested_size:
            size_label = requested_size
        else:
            size_label = "default"
        return out, f"Sent 1 image; received 1 result (requested {size_label})"
    except Exception as e:
        return None, f"Generation error: {e}"


def on_files_change(files):
    state = []
    thumbs = []
    if files:
        for f in files:
            try:
                pil = None
                if isinstance(f, dict) and "name" in f:
                    pil = Image.open(f["name"]).convert("RGB")
                else:
                    pil = Image.open(f).convert("RGB")
                state.append({"source": pil, "annotated": None, "mask": None})
                thumbs.append(pil)
            except Exception:
                continue
    return state, thumbs


def on_gallery_select(evt: gr.SelectData, images_state):
    # evt.index is the selected image index in the gallery
    idx = getattr(evt, "index", None)
    if idx is None:
        return gr.update(), None
    if idx == 0:
        # First tile is the live editor image; keep editor as-is
        return gr.update(), None
    real_idx = idx - 1
    if images_state is None or real_idx >= len(images_state):
        return gr.update(), None
    item = images_state[real_idx]
    # Prefer annotated in editor for further editing
    editor_image = item.get("annotated") or item.get("source")
    return editor_image, real_idx


def save_annotation(editor_data, current_index, images_state):
    comp, mask = _extract_editor_images(editor_data, True)
    msg = ""
    if current_index is not None and comp is not None:
        # comp and mask here are data URIs; we want to store PIL for gallery
        def _from_data_uri(uri):
            header, b64 = uri.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64)))
        comp_pil = _from_data_uri(comp)
        mask_pil = _from_data_uri(mask) if mask else None
        images_state[current_index]["annotated"] = comp_pil
        images_state[current_index]["mask"] = mask_pil
        msg = "Annotation auto-saved"
    items = _build_gallery(editor_data, images_state)
    return images_state, items, msg


def on_size_change(choice):
    show_custom = choice == "custom"
    return gr.update(visible=show_custom), gr.update(visible=show_custom)


def create_interface():
    with gr.Blocks(title="Nano Banana Edit", theme=gr.themes.Soft()) as demo:
        gr.Markdown("Nano Banana Edit — draw or upload multiple images. Annotations are auto-used; mask is included when present.")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image_editor = gr.ImageEditor(
                    type="pil",
                    image_mode="RGB",
                    height=600,
                    container=False,
                    label="Draw or paste an image"
                )
                files = gr.Files(file_types=["image"], file_count="multiple", label="Or upload multiple images")

            with gr.Column(scale=1):
                input_gallery = gr.Gallery(label="Uploaded + annotated previews", height=300, columns=4)
                with gr.Row():
                    output_image = gr.Image(
                        label="Output",
                        height=300,
                        interactive=False,
                        show_download_button=True,
                        type="pil",
                        format="jpeg"
                    )
                adopt_btn = gr.Button("Use output as new starting point", variant="secondary")

        with gr.Row():
            prompt = gr.Textbox(placeholder="Describe the edit...", lines=3, show_label=False)

        with gr.Row():
            seed = gr.Number(value=420, precision=0, container=False, scale=1, label="Seed")
            image_size = gr.Dropdown(
                choices=[
                    "square_hd", "square", "portrait_4_3", "portrait_16_9",
                    "landscape_4_3", "landscape_16_9", "custom"
                ],
                value="square_hd",
                label="Image size",
                scale=1,
            )
            custom_w = gr.Number(value=1024, precision=0, visible=False, label="Width")
            custom_h = gr.Number(value=1024, precision=0, visible=False, label="Height")
            run_btn = gr.Button("Generate", variant="primary", scale=1)

        status = gr.Textbox(show_label=False, interactive=False, container=False)

        # State to track uploaded images and annotations
        images_state = gr.State([])
        current_index = gr.State(None)

        files.change(
            fn=on_files_change,
            inputs=[files],
            outputs=[images_state, input_gallery]
        )

        input_gallery.select(
            fn=on_gallery_select,
            inputs=[images_state],
            outputs=[image_editor, current_index]
        )

        # Auto-save to the selected upload when editing changes
        image_editor.change(
            fn=save_annotation,
            inputs=[image_editor, current_index, images_state],
            outputs=[images_state, input_gallery, status]
        )

        image_size.change(
            fn=on_size_change,
            inputs=[image_size],
            outputs=[custom_w, custom_h]
        )

        run_btn.click(
            fn=process_images,
            inputs=[image_editor, files, prompt, seed, image_size, custom_w, custom_h, images_state, current_index],
            outputs=[output_image, status],
        )

        def adopt_output_as_input_fn(img, images_state):
            if img is None:
                return images_state, gr.update(), gr.update(), None, "No output image to adopt"
            # Ensure PIL
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except Exception:
                    return images_state, gr.update(), gr.update(), None, "Unsupported output image format"
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Do NOT append to state to avoid duplication; make it the live editor image
            state = images_state or []
            thumbs = [img] + [(x.get("annotated") or x.get("source")) for x in state]
            return state, thumbs, img, None, "Output adopted as new starting point"

        adopt_btn.click(
            fn=adopt_output_as_input_fn,
            inputs=[output_image, images_state],
            outputs=[images_state, input_gallery, image_editor, current_index, status]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1")
