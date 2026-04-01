import os
import anthropic
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import predict
from app.schemas import PredictionResponse, ExplainResponse


from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Traffic Sign Classifier",
    description="Real-time traffic sign classification using EfficientNet-B0 trained on GTSRB",
    version="1.0.0"
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sign(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted.")
    image_bytes = await file.read()
    return predict(image_bytes)

@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_and_explain(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted.")
    image_bytes = await file.read()
    result = predict(image_bytes)

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"A traffic sign classifier detected: '{result['class_name']}' "
                f"with {result['confidence']*100:.1f}% confidence. "
                "In 2-3 sentences, explain what this sign means and what a driver should do."
            )
        }]
    )
    explanation = message.content[0].text

    return {
        "predicted_class": result["predicted_class"],
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "explanation": explanation
    }