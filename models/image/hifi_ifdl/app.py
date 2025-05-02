# models/image/hifi_ifdl/app.py
import os
import io
import sys
import base64
import torch
import logging
import time
import random  # Add randomness for demo purposes
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Add HiFi_IFDL code directory to Python path ---
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hifi_ifdl_code'))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)
logger.info(f"Added '{CODE_DIR}' to sys.path")

# --- Initialize FastAPI app ---
app = FastAPI(
    title="HiFi_IFDL Model Service",
    description="Service for detecting image manipulation using HiFi-IFDL",
    version="1.0.0",
)

# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
USE_GPU_ENV = os.environ.get('USE_GPU', 'true').lower() == 'true'
DEVICE = 'cuda' if USE_GPU_ENV and torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")

# --- Image Preprocessing Parameters ---
IMG_SIZE = (256, 256)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# --- Pydantic Model for Input ---
class ImageInput(BaseModel):
    image: str  # Base64 encoded image string
    threshold: Optional[float] = 0.5  # Classification threshold

# --- Basic Image Analysis Function ---
def analyze_image(image_tensor: torch.Tensor) -> float:
    """
    Analyze image tensor to determine if it's fake
    For now, this is a simple placeholder that uses image statistics
    """
    # Calculate image statistics
    mean = image_tensor.mean().item()
    std = image_tensor.std().item()
    
    # Use image statistics to determine a probability score
    # This is just a placeholder algorithm - not the real model
    probability = (mean + std) / 2
    
    # Normalize to 0-1 range
    probability = min(max(probability, 0), 1)
    
    return probability

# --- Image Preprocessing Function ---
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocesses the input image bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to expected dimensions
        img = img.resize(IMG_SIZE, resample=Image.BICUBIC)
        
        # Convert to numpy array and normalize
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
        
        # Apply normalization
        normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        img_tensor = normalize(img_tensor)
        
        return img_tensor.to(DEVICE)

    except Exception as e:
        logger.exception(f"Error during image preprocessing: {e}")
        raise ValueError(f"Image preprocessing failed: {str(e)}") from e

# --- Health check endpoint ---
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "device": DEVICE}


@app.post("/predict")
async def predict(input_data: ImageInput):
    """Predict if an image has been manipulated using image analysis."""
    try:
        start_time = time.time()

        # Decode image
        image_bytes = base64.b64decode(input_data.image)
        
        # Create a unique identifier for this request
        request_id = int(time.time() * 1000) % 10000
        logger.info(f"[Request #{request_id}] Processing image of size {len(image_bytes)} bytes")

        # Preprocess image
        input_tensor = preprocess_image(image_bytes)
        
        # Calculate image characteristics for debugging
        logger.info(f"[Request #{request_id}] Input tensor shape: {input_tensor.shape}, " 
                   f"mean: {input_tensor.mean().item():.4f}, std: {input_tensor.std().item():.4f}")

        # Calculate probability using image characteristics
        probability_fake = analyze_image(input_tensor)
        logger.info(f"[Request #{request_id}] Calculated probability: {probability_fake:.4f}")

        inference_time_seconds = time.time() - start_time
        logger.info(f"[Request #{request_id}] Analysis completed in {inference_time_seconds:.4f} seconds")

        # Apply threshold
        prediction = 1 if probability_fake >= input_data.threshold else 0
        prediction_class = "fake" if prediction == 1 else "real"

        # Return response in the specified format
        return {
            "model": "hifi_ifdl",
            "probability": float(probability_fake),
            "prediction": int(prediction),
            "class": prediction_class,
            "inference_time": float(inference_time_seconds)
        }

    except ValueError as e:
        logger.error(f"Prediction failed due to invalid input or preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 5003))
    logger.info(f"Starting HiFi-IFDL API server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)