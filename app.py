from io import BytesIO
from pathlib import Path
import uuid
import os
import subprocess

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageOps

from huggingface_hub import login

from src.fashn_vton.pipeline import TryOnPipeline


# =========================================================
# Hugging Face Login
# =========================================================

token = os.getenv("HF_TOKEN")

if token:
    login(token=token)


# =========================================================
# Weights Setup
# =========================================================

WEIGHTS_DIR = "./weights"

os.makedirs(WEIGHTS_DIR, exist_ok=True)

if not os.path.exists(os.path.join(WEIGHTS_DIR, "model.safetensors")):
    print("Downloading weights...")
    subprocess.run(
        [
            "python",
            "scripts/download_weights.py",
            "--weights-dir",
            WEIGHTS_DIR,
        ],
        check=True,
    )


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="FASHN VTON API",
    version="1.0.0",
)

# =========================================================
# CORS FIX
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Output Directory
# =========================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve output images publicly
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# =========================================================
# Load Pipeline
# =========================================================

print("Loading TryOnPipeline...")

pipeline = TryOnPipeline(
    weights_dir="./weights",
    device=None,  # auto-detect
)

print("Pipeline loaded successfully")


# =========================================================
# Routes
# =========================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "FASHN-VTON"
    }


@app.post(
    "/tryon",
    description=(
        "Upload a person photo and a garment photo "
        "to generate a virtual try-on image.\n\n"
        "Categories:\n"
        "- tops\n"
        "- bottoms\n"
        "- one-pieces"
    )
)
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form(...),
):
    try:

        # =====================================================
        # Validate Category
        # =====================================================

        valid_categories = [
            "tops",
            "bottoms",
            "one-pieces"
        ]

        if category not in valid_categories:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": (
                        "Invalid category. "
                        "Must be one of: "
                        "tops, bottoms, one-pieces"
                    )
                }
            )

        # =====================================================
        # Load Images
        # =====================================================

        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()

        person_raw = Image.open(BytesIO(person_bytes))
        garment_raw = Image.open(BytesIO(garment_bytes))

        # Fix EXIF rotations + force RGB
        person_pil = (
            ImageOps.exif_transpose(person_raw)
            .convert("RGB")
        )

        garment_pil = (
            ImageOps.exif_transpose(garment_raw)
            .convert("RGB")
        )

        # =====================================================
        # Run Pipeline
        # =====================================================

        result = pipeline(
            person_image=person_pil,
            garment_image=garment_pil,
            category=category,
            garment_photo_type="flat-lay",

            # IMPORTANT:
            # Reduced from 10 -> 4
            # Huge speed improvement
            num_timesteps=4,

            guidance_scale=1.5,
            seed=42,
        )

        # =====================================================
        # Save Output
        # =====================================================

        output_filename = f"{uuid.uuid4()}.png"

        output_path = OUTPUT_DIR / output_filename

        result.images[0].save(output_path)

        # =====================================================
        # Return Public URL
        # =====================================================

        return {
            "success": True,
            "image_url": f"/outputs/{output_filename}"
        }

    except Exception as e:

        print("ERROR:", str(e))

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# =========================================================
# Direct Image Route (Optional)
# =========================================================

@app.get("/image/{filename}")
async def get_image(filename: str):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Image not found"
            }
        )

    return FileResponse(file_path)