# models/image/hifi_ifdl/app.py
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
# USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
# DEVICE = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
# Environment variables
# Environment variables
USE_GPU = False  # Force CPU usage
DEVICE = torch.device('cpu')  # Always use CPU
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600"))  # Seconds to keep model loaded
HRNET_WEIGHT_PATH = os.environ.get("HRNET_WEIGHT_PATH", "/app/weights/750001.pth")
NLCD_WEIGHT_PATH = os.environ.get("NLCD_WEIGHT_PATH", "/app/weights/NLCDetection.pth")

logger.info(f"Using device: {DEVICE}")

# --- Image Preprocessing Parameters ---
IMG_SIZE = (256, 256)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# --- Pydantic Model for Input ---
class ImageInput(BaseModel):
    image: str  # Base64 encoded image string
    threshold: Optional[float] = 0.5  # Classification threshold

# --- Global variables for model ---
model = None
model_lock = threading.Lock()
last_used_time = 0

# --- Load model function ---
def load_model():
    """Load the HiFi_IFDL model."""
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
            logger.info(f"Loading HiFi_IFDL model on {DEVICE}")
            
            # Here you would import and initialize the actual HiFi_IFDL model
            # For now, we're using a placeholder as the real model requires extra dependencies
            
            # Real implementation would be:
            # from models.seg_hrnet import get_seg_model
            # from config import config
            # model = get_seg_model(config)
            # model.load_state_dict(torch.load(HRNET_WEIGHT_PATH))
            
            # Placeholder model for now (can be replaced with actual implementation)
            class PlaceholderModel(torch.nn.Module):
                def __init__(self):
                    super(PlaceholderModel, self).__init__()
                    self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
                    self.sigmoid = torch.nn.Sigmoid()
                
                def forward(self, x):
                    return self.sigmoid(self.conv(x))
            
            model = PlaceholderModel()
            model = model.to(DEVICE)
            model.eval()
            
            # Update last used time
            last_used_time = time.time()
            
            # Clear CUDA cache to free up memory
            if USE_GPU and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Model loaded successfully on {DEVICE}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

# --- Unload model function ---
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
                if USE_GPU and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("Model unloaded and memory cleared")

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

# --- Analysis Function ---
def analyze_image(image_tensor: torch.Tensor) -> float:
    """
    Analyze image tensor to determine if it's fake
    For now, this is a simple placeholder that uses image statistics
    """
    global model
    
    try:
        with torch.no_grad():
            # In a real implementation, we would use the model:
            output = model(image_tensor)
            
            # For the placeholder, get a fake probability from the model output
            if isinstance(output, torch.Tensor):
                probability = output.mean().item()
            else:
                # Calculate image statistics as a fallback
                mean = image_tensor.mean().item()
                std = image_tensor.std().item()
                probability = (mean + std) / 2
            
            # Normalize to 0-1 range
            probability = min(max(probability, 0), 1)
            
            return probability
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# --- Health check endpoint ---
@app.get("/health")
async def health():
    """Health check endpoint."""
    hrnet_weights_exist = os.path.exists(HRNET_WEIGHT_PATH)
    nlcd_weights_exist = os.path.exists(NLCD_WEIGHT_PATH)
    
    return {
        "status": "healthy" if hrnet_weights_exist and nlcd_weights_exist else "missing_weights",
        "device": DEVICE,
        "model_loaded": model is not None,
        "lazy_loading": not PRELOAD_MODEL,
        "weights_status": {
            "hrnet": "found" if hrnet_weights_exist else "missing",
            "nlcd": "found" if nlcd_weights_exist else "missing"
        }
    }

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(input_data: ImageInput):
    """Predict if an image has been manipulated using HiFi_IFDL."""
    global model
    
    try:
        # Make sure model is loaded
        if model is None:
            model = load_model()
        else:
            # Update timestamp if already loaded
            global last_used_time
            last_used_time = time.time()
            
        # Start timer
        start_time = time.time()

        # Decode image
        image_bytes = base64.b64decode(input_data.image)
        
        # Create a unique identifier for this request
        request_id = int(time.time() * 1000) % 10000
        logger.info(f"[Request #{request_id}] Processing image of size {len(image_bytes)} bytes")

        # Optimize memory during inference
        if USE_GPU and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Preprocess image
        input_tensor = preprocess_image(image_bytes)
        
        # Calculate image characteristics for debugging
        logger.info(f"[Request #{request_id}] Input tensor shape: {input_tensor.shape}, " 
                   f"mean: {input_tensor.mean().item():.4f}, std: {input_tensor.std().item():.4f}")

        # Calculate probability using model
        probability_fake = analyze_image(input_tensor)
        logger.info(f"[Request #{request_id}] Calculated probability: {probability_fake:.4f}")

        # Calculate inference time
        inference_time_seconds = time.time() - start_time
        logger.info(f"[Request #{request_id}] Analysis completed in {inference_time_seconds:.4f} seconds")

        # Apply threshold
        prediction = 1 if probability_fake >= input_data.threshold else 0
        prediction_class = "fake" if prediction == 1 else "real"

        # Schedule unloading after timeout - run in background
        threading.Timer(5.0, unload_model_if_idle).start()

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

# --- Startup event ---
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

# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 5003))
    logger.info(f"Starting HiFi-IFDL API server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)