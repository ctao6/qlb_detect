from fastapi import FastAPI, UploadFile, File
import io
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tensorflow.keras import models, layers

app = FastAPI(title="QLB Tree Health Classifier - DinoV2")

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "dinov2_head.weights.h5")
MODEL_NAME = "facebook/dinov2-small"
FEATURE_DIM = 768

device = "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
backbone = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

head = models.Sequential([
    layers.Input(shape=(FEATURE_DIM,)),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])

if os.path.exists(WEIGHTS_PATH):
    head.load_weights(WEIGHTS_PATH)
    print("DinoV2 Classification Head weights loaded successfully.")
else:
    print(f"Warning: Weights file not found at {WEIGHTS_PATH}!")

def extract_dinov2_features(pil_image: Image.Image) -> np.ndarray:
    img_resized = pil_image.convert("RGB").resize((224, 224))
    inputs = processor(images=img_resized, do_resize=False, do_center_crop=False, return_tensors="pt").to(device)
    with torch.no_grad():
        out = backbone(**inputs).last_hidden_state
    cls_tok = out[:, 0]
    patch_mean = out[:, 1:].mean(dim=1)
    features = torch.cat([cls_tok, patch_mean], dim=1).cpu().numpy()
    return features

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        features = extract_dinov2_features(image)
        prediction = float(head.predict(features, verbose=0)[0][0])

        if prediction >= 0.5:
            label = "INFESTED"
            confidence = prediction * 100
        else:
            label = "HEALTHY"
            confidence = (1 - prediction) * 100

        return {
            "status": "success",
            "label": label,
            "confidence": round(confidence, 2),
            "raw_score": prediction
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}