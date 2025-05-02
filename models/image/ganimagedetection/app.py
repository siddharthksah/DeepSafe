"""
GAN Image Detection Model Service
This service loads the GANimageDetection model and exposes an API endpoint to analyze images
"""
import os
import io
import sys
import base64
import torch
import logging
import time
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add model path to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'ganimagedetection'))

# Import from GANimageDetection
from resnet50nodown import resnet50nodown

# Initialize FastAPI app
app = FastAPI(
    title="GANimageDetection Model Service",
    description="Service for detecting GAN-generated images using GANimageDetection model",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths and settings
MODEL_PATH = os.environ.get("MODEL_PATH", "ganimagedetection/weights/gandetection_resnet50nodown_stylegan2.pth")
USE_GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU", False)
DEVICE = 'cuda:0' if USE_GPU else 'cpu'

# Define request model - update to match UniversalFakeDetect
class ImageInput(BaseModel):
    image: str  # Base64 encoded image
    threshold: Optional[float] = 0.5  # Optional threshold for classification

# Global variable for the model
model = None

def load_model():
    """Load the GANimageDetection model."""
    global model
    
    try:
        logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}")
        
        # Load the model using the original library's function
        model = resnet50nodown(DEVICE, MODEL_PATH)
        logger.info(f"Model loaded successfully on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "model": "GANimageDetection",
        "description": "Model for detecting GAN-generated images",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "healthy", "device": DEVICE}

@app.post("/predict")
async def predict_image(input_data: ImageInput) -> Dict[str, Any]:
    """
    Detect if an image is GAN-generated using the GANimageDetection model.
    Returns standardized output matching other models.
    """
    global model
    
    try:
        # Make sure model is loaded
        if model is None:
            load_model()
            
        # Decode base64 image
        image_bytes = base64.b64decode(input_data.image)
        
        # Start timing
        start_time = time.time()
        
        # Open the image using PIL
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Run prediction
        logit = model.apply(pil_image)
        
        # Convert logit to probability using sigmoid
        probability = float(1.0 / (1.0 + np.exp(-logit)))
        
        # Apply threshold
        prediction = 1 if probability >= input_data.threshold else 0
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.4f} seconds")
        
        # Return standardized format
        return {
            "model": "GANimageDetection",
            "probability": float(probability),
            "prediction": int(prediction),
            "class": "fake" if prediction == 1 else "real",
            "inference_time": float(inference_time)
        }
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)