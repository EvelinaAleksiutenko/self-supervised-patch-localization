from PIL import Image
import io
from fastapi import HTTPException


def _load_image(raw_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
