"""
CNN Deepfake Detection Model Service
This service loads the CNNDetection model and exposes an API endpoint to analyze images
"""
import os
import io
import sys
import base64
import torch
import logging
import time
import threading
import gc
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import torchvision.transforms as transforms
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
sys.path.append(os.path.join(current_dir, 'cnndetection'))

# Initialize FastAPI app
app = FastAPI(
    title="CNNDetection Model Service",
    description="Service for detecting deepfake images using CNNDetection model",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths and settings
MODEL_PATH = os.environ.get("MODEL_PATH", "cnndetection/weights/blur_jpg_prob0.5.pth")
USE_GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU", False)
DEVICE = 'cuda' if USE_GPU else 'cpu'
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "5"))  # Seconds to keep model loaded

# Define request model - update to match UniversalFakeDetect
class ImageInput(BaseModel):
    image: str  # Base64 encoded image
    threshold: Optional[float] = 0.5  # Optional threshold for classification

# Global variable for the model
model = None
model_lock = threading.Lock()
last_used_time = 0

def load_model():
    """Load the CNNDetection model."""
    global model, last_used_time
    
    # If model is already loaded, update timestamp and return
    if model is not None:
        last_used_time = time.time()
        return model
        
    with model_lock:  # Thread safety for concurrent requests
        # Check again after acquiring the lock
        if model is not None:
            last_used_time = time.time()
            return model
    
        try:
            logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}")
            
            # Import after path setup
            from networks.resnet import resnet50
            
            # Load the model
            model = resnet50(num_classes=1)
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict['model'])
            
            if USE_GPU:
                model.cuda()
                logger.info(f"Model moved to GPU")
            
            model.eval()
            
            # Update last used time
            last_used_time = time.time()
            
            logger.info(f"Model loaded successfully")
            
            # Clear CUDA cache to free up memory
            if USE_GPU:
                torch.cuda.empty_cache()
            gc.collect()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

def unload_model_if_idle():
    """Unload model if it's been idle for too long."""
    global model, last_used_time
    
    if model is None:
        return
        
    if time.time() - last_used_time > MODEL_TIMEOUT:
        with model_lock:
            if model is not None and time.time() - last_used_time > MODEL_TIMEOUT:
                logger.info(f"Unloading model after {MODEL_TIMEOUT} seconds of inactivity")
                # Delete model and clear memory
                del model
                model = None
                # Clear CUDA cache
                if USE_GPU:
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("Model unloaded and memory cleared")

def preprocess_image(image_bytes):
    """Preprocess the image for the model."""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply the same preprocessing as in the original CNNDetection
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return preprocess(image).unsqueeze(0)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "model": "CNNDetection",
        "description": "CNN-based deepfake image detection",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "lazy_loading": not PRELOAD_MODEL
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    model_file_exists = os.path.exists(MODEL_PATH)
    
    return {
        "status": "healthy" if model_file_exists else "missing_weights",
        "device": DEVICE,
        "model_loaded": model is not None,
        "lazy_loading": not PRELOAD_MODEL
    }

@app.post("/predict")
async def predict_image(input_data: ImageInput) -> Dict[str, Any]:
    """
    Detect if an image is a deepfake using the CNNDetection model.
    Returns standardized output matching other models.
    """
    global model
    
    try:
        # Make sure model is loaded
        if model is None:
            model = load_model()
        else:
            # Update timestamp if already loaded
            global last_used_time
            last_used_time = time.time()
            
        # Decode base64 image
        image_bytes = base64.b64decode(input_data.image)
        
        # Start timing
        start_time = time.time()
        
        # Optimize memory during inference
        if USE_GPU:
            torch.cuda.empty_cache()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Move tensor to appropriate device
        if USE_GPU:
            image_tensor = image_tensor.cuda()
        
        # Run prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
        
        # Apply threshold
        prediction = 1 if probability >= input_data.threshold else 0
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.4f} seconds")
        
        # Schedule unloading after timeout - run in background
        threading.Timer(5.0, unload_model_if_idle).start()
        
        # Return standardized format
        return {
            "model": "CNNDetection",
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
    """Load model on startup only if PRELOAD_MODEL is true."""
    if PRELOAD_MODEL:
        logger.info("Preloading model at startup (PRELOAD_MODEL=true)")
        try:
            load_model()
        except Exception as e:
            logger.error(f"Preloading failed: {str(e)}")
    else:
        logger.info("Model will be loaded on first request (PRELOAD_MODEL=false)")

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)