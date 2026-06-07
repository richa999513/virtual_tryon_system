from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Literal

import gradio as gr
import torch
from huggingface_hub import login
from PIL import Image, ImageOps

from src.fashn_vton.pipeline import TryOnPipeline


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "./weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def get_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        return None


HF_TOKEN = get_hf_token()
if HF_TOKEN:
    login(token=HF_TOKEN)

if not os.path.exists(os.path.join(WEIGHTS_DIR, "model.safetensors")):
    print("Downloading weights...")
    subprocess.run(
        ["python", "scripts/download_weights.py", "--weights-dir", WEIGHTS_DIR],
        check=True,
    )


pipeline: TryOnPipeline | None = None
pipeline_ready = False
pipeline_error: str | None = None


def load_pipeline() -> None:
    global pipeline, pipeline_ready, pipeline_error

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TryOnPipeline on {device}...")
        pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR, device=device)
        pipeline_ready = True
        print("Pipeline READY")
    except Exception as exc:
        pipeline_error = str(exc)
        print(f"Pipeline failed to load: {pipeline_error}")


threading.Thread(target=load_pipeline, daemon=True).start()


def _prepare_image(image: Image.Image | None) -> Image.Image:
    if image is None:
        raise gr.Error("Please upload both person and garment images.")
    return ImageOps.exif_transpose(image).convert("RGB")


def tryon(
    person_image: Image.Image | None,
    garment_image: Image.Image | None,
    category: Literal["tops", "bottoms", "one-pieces"],
    garment_photo_type: Literal["auto", "model", "flat-lay"],
    num_samples: int,
    num_timesteps: int,
    guidance_scale: float,
    seed: int,
    segmentation_free: bool,
) -> list[Image.Image]:
    if pipeline_error:
        raise gr.Error(f"Pipeline failed to load: {pipeline_error}")
    if not pipeline_ready or pipeline is None:
        raise gr.Error("Pipeline is still loading. Try again in a moment.")

    person = _prepare_image(person_image)
    garment = _prepare_image(garment_image)

    with torch.inference_mode():
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=int(num_samples),
            num_timesteps=int(num_timesteps),
            guidance_scale=guidance_scale,
            skip_cfg_last_n_steps=1,
            seed=int(seed),
            segmentation_free=segmentation_free,
        )

    return result.images


def pipeline_status() -> str:
    if pipeline_ready:
        return "Pipeline ready"
    if pipeline_error:
        return f"Pipeline error: {pipeline_error}"
    return "Pipeline loading"


with gr.Blocks(title="FASHN VTON") as demo:
    gr.Markdown("## FASHN VTON")
    status = gr.Textbox(label="Status", value=pipeline_status(), interactive=False)

    with gr.Row():
        person_input = gr.Image(label="Person image", type="pil", image_mode="RGB")
        garment_input = gr.Image(label="Garment image", type="pil", image_mode="RGB")

    with gr.Row():
        category_input = gr.Dropdown(
            ["tops", "bottoms", "one-pieces"],
            label="Category",
            value="tops",
            allow_custom_value=False,
        )
        garment_type_input = gr.Radio(
            ["auto", "model", "flat-lay"],
            label="Garment photo type",
            value="auto",
        )

    with gr.Row():
        samples_input = gr.Slider(1, 4, value=1, step=1, label="Samples")
        steps_input = gr.Slider(10, 50, value=24, step=1, label="Timesteps")
        guidance_input = gr.Slider(0.5, 3.0, value=1.4, step=0.1, label="Guidance")
        seed_input = gr.Number(value=42, precision=0, label="Seed")

    segmentation_input = gr.Checkbox(
        value=False,
        label="Segmentation-free mode",
        info="Leave off for kurtis, T-shirts, frocks, midis, and worn garment photos.",
    )

    with gr.Row():
        refresh_button = gr.Button("Refresh status")
        run_button = gr.Button("Run try-on", variant="primary")

    output_gallery = gr.Gallery(label="Output", columns=2, object_fit="contain", height=640)

    refresh_button.click(fn=pipeline_status, outputs=status)
    run_button.click(
        fn=tryon,
        inputs=[
            person_input,
            garment_input,
            category_input,
            garment_type_input,
            samples_input,
            steps_input,
            guidance_input,
            seed_input,
            segmentation_input,
        ],
        outputs=output_gallery,
    )


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
