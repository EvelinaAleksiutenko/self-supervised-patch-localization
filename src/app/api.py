from __future__ import annotations

import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile

from src.config.config import Config
from src.utils.data import source_transform, patch_transform
from src.helpers.image import _load_image
from src.utils.model import SiamesePatchLocalizer
from src.schemas.prediction_response import PredictionResponse

cfg = Config()
model: SiamesePatchLocalizer | None = None
device: torch.device = torch.device("cpu")

IMG_TRANSFORM = source_transform(cfg)
PATCH_TRANSFORM = patch_transform(cfg)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, device
    device = torch.device(cfg.device)
    model = SiamesePatchLocalizer(cfg).to(device)

    ckpt_path = cfg.checkpoint_path
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    except FileNotFoundError:
        raise RuntimeError(
            f"Checkpoint not found at '{ckpt_path}'. Train the model first."
        )
    model.eval()
    yield


app = FastAPI(
    title="Patch Localization API",
    description="Predict the (y, x) top-left coordinates of a patch inside a source image.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    source: UploadFile = File(..., description="Source image (any format PIL supports)"),
    patch: UploadFile = File(..., description="Patch image to locate inside the source"),
):
    """Return predicted (y, x) top-left pixel coordinates of the patch in the source."""
    source_img = _load_image(await source.read())
    patch_img = _load_image(await patch.read())

    source_t = IMG_TRANSFORM(source_img).unsqueeze(0).to(device)
    patch_t = PATCH_TRANSFORM(patch_img).unsqueeze(0).to(device)

    with torch.no_grad():
        coords, _ = model(source_t, patch_t)

    y, x = coords[0].tolist()
    return PredictionResponse(y=round(y, 3), x=round(x, 3))
