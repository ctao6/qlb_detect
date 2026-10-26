from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.model_loader import preprocess_image
import tensorflow as tf
import os

router = APIRouter()

model = None

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    try:
        image_bytes = await file.read()
        processed_img = preprocess_image(image_bytes)

        prediction = model.predict(processed_img)[0][0]

        if prediction >= 0.5:
            label, confidence = "Infested", float(prediction*100)
        else:
            label, confidence = "Healthy", float((1-prediction)*100)

        return {"status": "success", "label":label, "confidence":confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
