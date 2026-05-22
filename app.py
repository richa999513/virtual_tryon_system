from io import BytesIO
from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps  # Added ImageOps to handle EXIF rotation data cleanly

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


app = FastAPI(title="FASHN VTON API")

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
    description=(
        "Upload a person photo and a garment photo to generate a virtual try-on image. "
        "\n\n**Possible categories for the 'category' field:**"
        "\n* `tops` - For upper body wear (shirts, t-shirts, blouses, jackets, hoodies)"
        "\n* `bottoms` - For lower body wear (pants, shorts, skirts, jeans)"
        "\n* `one-pieces` - For full body wear (dresses, jumpsuits, overalls)"
    )
)
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form(
        ..., 
        description="The clothing category. Must be one of: tops, bottoms, one-pieces"
    ),
):
    try:
        if category not in ["tops", "bottoms", "one-pieces"]:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid category"
                }
            )

        # Read, fix hidden EXIF rotations, and force standard 3-channel RGB format
        person_raw = Image.open(BytesIO(await person_image.read()))
        person_pil = ImageOps.exif_transpose(person_raw).convert("RGB")

        garment_raw = Image.open(BytesIO(await garment_image.read()))
        garment_pil = ImageOps.exif_transpose(garment_raw).convert("RGB")

        # Keeping your exact original parameters, minus the ones causing the 500 errors
        result = pipeline(
            person_image=person_pil,
            garment_image=garment_pil,
            category=category,
            garment_photo_type="flat-lay",
            num_timesteps=10,
            guidance_scale=1.5,
            seed=42,
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