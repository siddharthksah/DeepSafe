import os
import io
import time
import base64
import logging
import traceback
import threading
import gc
from typing import Dict, Any, Optional
import sys
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup more detailed logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="UniversalFakeDetect API", 
              description="API for Universal Fake Image Detector",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
# USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
USE_GPU = False
DEVICE = torch.device('cpu')  # Always use CPU
MODEL_PORT = int(os.environ.get('MODEL_PORT', 5004))
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600"))  # Seconds to keep model loaded

USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true" and torch.cuda.is_available()

# Check if CUDA is available
# DEVICE = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
logger.info(f"Using device: {DEVICE}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Define input model
from pydantic import BaseModel
from typing import Optional

class ImageInput(BaseModel):
    image_data: str  # Renamed field
    threshold: Optional[float] = 0.5

# Global variables for model
model = None
model_lock = threading.Lock()
last_used_time = 0
model_loading = False

# Download weights if not present
def download_weights():
    weights_path = 'universalfakedetect/pretrained_weights/fc_weights.pth'
    if not os.path.exists('universalfakedetect/pretrained_weights'):
        os.makedirs('universalfakedetect/pretrained_weights', exist_ok=True)
        logger.info("Created pretrained_weights directory")
    
    if not os.path.exists(weights_path):
        logger.info("Weights file not found, downloading...")
        import urllib.request
        url = "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth"
        urllib.request.urlretrieve(url, weights_path)
        logger.info(f"Downloaded weights to {weights_path}")

# Load the model
def load_model():
    """Load the UniversalFakeDetect model."""
    global model, last_used_time, model_loading
    
    # If model is already loaded, update timestamp and return
    if model is not None:
        last_used_time = time.time()
        return model
        
    with model_lock:  # Thread safety for concurrent requests
        # Check again after acquiring the lock
        if model is not None:
            last_used_time = time.time()
            return model
            
        # Set flag to indicate model is loading
        model_loading = True
        
        try:
            logger.info(f"Loading UniversalFakeDetect model on {DEVICE}...")
            
            # List directory contents for debugging
            logger.info(f"Current directory: {os.getcwd()}")
            logger.info(f"Directory contents: {os.listdir('.')}")
            
            if os.path.exists('universalfakedetect'):
                logger.info(f"universalfakedetect directory contents: {os.listdir('universalfakedetect')}")
            else:
                logger.error("universalfakedetect directory not found!")
                model_loading = False
                return None
                
            # Ensure weights exist
            download_weights()
            weights_path = 'universalfakedetect/pretrained_weights/fc_weights.pth'
                
            # Import model modules
            logger.info("Adding universalfakedetect to sys.path")
            sys.path.append(os.path.abspath('universalfakedetect'))
            
            logger.info("Importing get_model from models")
            try:
                # Try direct import first
                from models import get_model
                logger.info("Successfully imported get_model")
            except ImportError as e:
                logger.warning(f"Direct import failed: {str(e)}, trying alternate import")
                from universalfakedetect.models import get_model
                logger.info("Successfully imported get_model with alternate path")
            except Exception as e:
                logger.error(f"Error importing get_model: {str(e)}")
                logger.error(traceback.format_exc())
                model_loading = False
                return None
            
            # Initialize the model
            logger.info("Initializing model with CLIP:ViT-L/14")
            model = get_model("CLIP:ViT-L/14")
            
            # Load the weights
            logger.info(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            model.fc.load_state_dict(state_dict)
            
            # Move model to device and set to evaluation mode
            logger.info(f"Moving model to device: {DEVICE}")
            model.to(DEVICE)
            model.eval()
            
            # Update last used time
            last_used_time = time.time()
            
            # Clear CUDA cache to free up memory
            if USE_GPU:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Model loaded successfully!")
            model_loading = False
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            model_loading = False
            return None
        
@app.post("/unload", include_in_schema=True)
async def unload_model_endpoint():
    """Endpoint to manually unload the model."""
    global model, model_loading
    
    if model_loading:
        return {"status": "loading_busy", "message": "Model is currently being loaded, cannot unload now."}

    if model is None:
        return {"status": "not_loaded", "message": "Model is not currently loaded."}
    
    with model_lock:
        if model is not None: # Check again inside lock
            logger.info("Manually unloading UniversalFakeDetect model via /unload endpoint.")
            del model
            model = None
            # Clear CUDA cache if it was used (though DEVICE is 'cpu' here, good practice)
            if DEVICE.type == 'cuda':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()
            logger.info("UniversalFakeDetect model unloaded and memory cleared.")
            return {"status": "unloaded", "message": "Model unloaded successfully."}
        else: # Should not happen if initial check was model is not None
            return {"status": "already_unloaded", "message": "Model was already unloaded."}

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
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                gc.collect()
                logger.info("Model unloaded and memory cleared")

# Preprocess image for inference
def preprocess_image(image_bytes):
    try:
        # Read the image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess image
        from torchvision import transforms
        
        # These values are from the validate.py file in the repository
        mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP values
        std = [0.26862954, 0.26130258, 0.27577711]   # CLIP values
        
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return img_tensor.to(DEVICE)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

# Predict function
def predict(image_tensor, threshold=0.5):
    try:
        with torch.no_grad():
            # Forward pass through the model
            output = model(image_tensor).sigmoid().flatten().item()
            # The model outputs a score between 0 and 1, where higher values indicate fake images
            prediction = 1 if output >= threshold else 0
            
            return {
                "probability": float(output),
                "prediction": int(prediction),
                "class": "fake" if prediction == 1 else "real"
            }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint
@app.get("/")
def read_root():
    return {
        "model": "UniversalFakeDetect",
        "description": "Universal Fake Image Detector that Generalizes Across Generative Models",
        "authors": "Utkarsh Ojha, Yuheng Li, Yong Jae Lee",
        "paper": "https://arxiv.org/abs/2302.10174",
        "source": "https://github.com/WisconsinAIVision/UniversalFakeDetect",
        "model_loaded": model is not None,
        "lazy_loading": not PRELOAD_MODEL
    }

# Health check endpoint
@app.get("/health")
def health_check():
    global model, model_loading
    
    weights_path = 'universalfakedetect/pretrained_weights/fc_weights.pth'
    model_file_exists = os.path.exists(weights_path)
    
    if model is not None:
        return {"status": "healthy", "device": str(DEVICE), "model_loaded": True}
    elif model_loading:
        return {"status": "loading", "message": "Model is being loaded", "device": str(DEVICE)}
    elif not model_file_exists:
        return {"status": "missing_weights", "message": "Model weights not found", "device": str(DEVICE)}
    else:
        return {"status": "not_loaded", "message": "Model not loaded yet", "device": str(DEVICE)}

# Prediction endpoint
@app.post("/predict")
async def predict_image(input_data: ImageInput) -> Dict[str, Any]:
    # Check if model is loaded
    global model, last_used_time
    
    try:
        # Load model if not already loaded
        if model is None:
            logger.info("Model not loaded. Loading model now...")
            model = load_model()
            if model is None:
                raise HTTPException(status_code=500, detail="Failed to load model")
        else:
            # Update timestamp if already loaded
            last_used_time = time.time()
    
        # Decode base64 image
        image_bytes = base64.b64decode(input_data.image_data)
        
        # Start timing
        start_time = time.time()
        
        # Optimize memory during inference
        if USE_GPU:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Preprocess the image
        image_tensor = preprocess_image(image_bytes)
        
        # Get predictions
        results = predict(image_tensor, input_data.threshold)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Schedule unloading after timeout - run in background
        threading.Timer(5.0, unload_model_if_idle).start()
        
        # Return results
        return {
            "model": "UniversalFakeDetect",
            "probability": results["probability"],
            "prediction": results["prediction"],
            "class": results["class"],
            "inference_time": inference_time
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Load model on startup
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

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=MODEL_PORT, reload=False)