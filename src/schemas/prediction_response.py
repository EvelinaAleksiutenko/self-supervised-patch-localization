from pydantic import BaseModel


class PredictionResponse(BaseModel):
    y: float
    x: float
