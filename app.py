from io import BytesIO
from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

from src.fashn_vton.pipeline import TryOnPipeline

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


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form(...),
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

        person_pil = Image.open(
            BytesIO(await person_image.read())
        ).convert("RGB")

        garment_pil = Image.open(
            BytesIO(await garment_image.read())
        ).convert("RGB")

        result = pipeline(
            person_image=person_pil,
            garment_image=garment_pil,
            category=category,
            garment_photo_type="flat-lay",
            num_samples=1,
            num_timesteps=10,
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