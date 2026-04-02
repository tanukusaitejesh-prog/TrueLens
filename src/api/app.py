from fastapi import FastAPI
from pydantic import BaseModel
from src.extraction.activations import ActivationExtractor
from src.probing.probe import HallucinationProbe
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class PromptRequest(BaseModel):
    prompt: str

# 1. Initialize probe and extractor
probe = HallucinationProbe()
extractor = probe.extractor

# 2. Auto-load latest trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
if os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
    probe.load(MODEL_DIR)
    print("✓ Trained models loaded successfully.")
else:
    print("⚠ WARNING: No trained models found in /models. Run train.py first!")

@app.get("/")
def check():
    return "TrueLens API is running!"

@app.post("/detect")
def detect(req: PromptRequest):
    prompt = req.prompt
    
    # Check if models were trained on single layer (1024) or enhanced features (8196)
    if hasattr(probe.scaler, 'n_features_in_') and probe.scaler.n_features_in_ > 1024:
        acts = extractor.get_enhanced_features(prompt)
    else:
        acts = extractor.get_activation(prompt)
        
    acts = acts.reshape(1, -1)
    top = extractor.get_top_prediction(prompt)
    
    # Return JSON serialized scores + probabilities
    score = probe.predict(acts)
    
    return {
        "prompt": prompt,
        "activation_shape": list(acts.shape),
        "top_prediction": top,
        "scores": score
    }

@app.post("/attention")
def attention(req: PromptRequest):
    """
    Dedicated endpoint for attention patterns to avoid bloating the /detect endpoint.
    Retrieving all attention layers is very slow and data-heavy.
    """
    prompt = req.prompt
    pts = extractor.get_attention_patterns(prompt)
    
    return {
        "prompt": prompt,
        "tokens": pts["tokens"],
        "patterns": pts["patterns"]
    }

@app.get("/model-info")
def model_info():
    return {
        "model_name": extractor.model_name,
        "layer": extractor.layer,
        "device": extractor.device
    }
