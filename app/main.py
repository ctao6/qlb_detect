#to run: uvicorn app.main:app --reload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints
from app.services.model_loader import build_and_load_model
import os

app = FastAPI(title="QLB Infestation Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
def startup_event():
    weights_path = os.path.join("models", "Weights.weights.h5")
    endpoints.model = build_and_load_model(weights_path)
    print("model successfully compiled and loaded")

app.include_router(endpoints.router)