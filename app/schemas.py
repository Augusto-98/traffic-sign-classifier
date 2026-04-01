from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    top5: List[dict]

class ExplainResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    explanation: str