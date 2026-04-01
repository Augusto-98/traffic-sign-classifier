import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from PIL import Image
import io

DEVICE = torch.device("cpu")
MODEL_PATH = "model_weights/efficientnet_gtsrb.pt"
NUM_CLASSES = 43

CLASS_NAMES = {
    0: "Speed limit 20km/h", 1: "Speed limit 30km/h", 2: "Speed limit 50km/h",
    3: "Speed limit 60km/h", 4: "Speed limit 70km/h", 5: "Speed limit 80km/h",
    6: "End of speed limit 80km/h", 7: "Speed limit 100km/h", 8: "Speed limit 120km/h",
    9: "No passing", 10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at next intersection", 12: "Priority road",
    13: "Yield", 14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5t prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing for vehicles over 3.5t"
}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.3337, 0.3064, 0.3171],
                         [0.2672, 0.2564, 0.2629])
])

def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

def predict(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top5_probs, top5_idx = torch.topk(probs, 5)
    top5 = [
        {"class": int(i), "class_name": CLASS_NAMES[int(i)], "confidence": round(float(p), 4)}
        for i, p in zip(top5_idx, top5_probs)
    ]

    pred_class = int(top5_idx[0])
    return {
        "predicted_class": pred_class,
        "class_name": CLASS_NAMES[pred_class],
        "confidence": round(float(top5_probs[0]), 4),
        "top5": top5
    }