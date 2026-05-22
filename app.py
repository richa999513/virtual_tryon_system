from io import BytesIO
from pathlib import Path
import uuid
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

from src.fashn_vton.pipeline import TryOnPipeline
import subprocess
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN")

if token:
    login(token=token)
WEIGHTS_DIR = "./weights"

if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

if not os.path.exists(os.path.join(WEIGHTS_DIR, "model.safetensors")):
    subprocess.run(["python", "scripts/download_weights.py", "--weights-dir", WEIGHTS_DIR])

# root_path="/" fixes asset loading issues inside Hugging Face proxy/iframes
app = FastAPI(
    title="FASHN VTON API",
    root_path="/"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading TryOnPipeline...")

pipeline = TryOnPipeline(
    weights_dir="./weights",
    device=None
)

print("Pipeline loaded successfully")


@app.get("/")
def home():
    return {
        "status": "running",
        "model": "FASHN-VTON"
    }


@app.post(
    "/tryon",
    summary="Generate Virtual Try-On",
    description=(
        "Upload a person photo and a garment photo to generate a virtual try-on image. "
        "\n\n**Valid Categories:**"
        "\n* `tops`: For shirts, t-shirts, blouses, hoodies, jackets, etc."
        "\n* `bottoms`: For pants, skirts, shorts, jeans, trousers, etc."
        "\n* `one-pieces`: For full dresses, jumpsuits, overalls, tunics, etc."
    )
)
async def tryon(
    person_image: UploadFile = File(..., description="Image of the person standing facing forward."),
    garment_image: UploadFile = File(..., description="Clear image of the clothing item (flat-lay or on a model)."),
    category: Literal["tops", "bottoms", "one-pieces"] = Form(
        ..., 
        description="The garment category type. Must select one of: tops, bottoms, one-pieces."
    ),
):
    try:
        # Extra safety check for raw API requests bypassing the Swagger schema
        if category not in ["tops", "bottoms", "one-pieces"]:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Invalid category '{category}'. Must be one of: 'tops', 'bottoms', 'one-pieces'"
                }
            )

        person_pil = Image.open(
            BytesIO(await person_image.read())
        ).convert("RGB")

        garment_pil = Image.open(
            BytesIO(await garment_image.read())
        ).convert("RGB")

        # FIXED: Removed 'num_samples' parameter which is unsupported by the local python package.
        # Adjusted num_timesteps to the model's standard balanced default (30).
        result = pipeline(
            person_image=person_pil,
            garment_image=garment_pil,
            category=category,
            garment_photo_type="flat-lay",
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
            segmentation_free=True,
        )

        output_filename = f"{uuid.uuid4()}.png"
        output_path = OUTPUT_DIR / output_filename
        result.images[0].save(output_path)

        return {
            "success": True,
            "output_image": str(output_path)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )