import os
import io
import sys
import base64
import torch
import logging
import time
import threading
import gc
import shutil
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model_code')
training_dir = os.path.join(model_dir, 'training')

# List all files in the weights directory for debugging
logger.info(f"Searching for model weights...")
weights_dir = os.path.join(model_dir, 'weights')
if os.path.exists(weights_dir):
    logger.info(f"Contents of weights directory: {os.listdir(weights_dir)}")
else:
    logger.warning(f"Weights directory does not exist: {weights_dir}")

# Add paths to sys.path in correct order
sys.path.insert(0, model_dir)
sys.path.insert(0, training_dir)
sys.path.insert(0, os.path.join(training_dir, 'detectors'))

# Create __init__.py files to make packages
def create_package_init_files():
    dirs_needing_init = [
        os.path.join(model_dir),
        os.path.join(training_dir),
        os.path.join(training_dir, 'detectors'),
        os.path.join(training_dir, 'metrics'),
        os.path.join(training_dir, 'loss'),
    ]
    
    for dir_path in dirs_needing_init:
        init_file = os.path.join(dir_path, '__init__.py')
        if not os.path.exists(init_file):
            try:
                with open(init_file, 'w') as f:
                    f.write('# Auto-generated to make this directory a Python package\n')
                logger.info(f"Created __init__.py in {dir_path}")
            except Exception as e:
                logger.warning(f"Could not create __init__.py in {dir_path}: {e}")

# Create necessary init files
create_package_init_files()

# Initialize FastAPI app
app = FastAPI(
    title="Wavelet-CLIP Deepfake Detection Model Service",
    description="Service for detecting deepfake images using Wavelet-CLIP model.",
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

# Model Configuration & Globals
MODEL_NAME = "wavelet_clip_detection"
MODEL_PATH = os.environ.get("MODEL_PATH", "model_code/weights/clip_wavelet_best.pth")
DEVICE = torch.device('cpu')  # Force CPU
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600"))  # Default 10 minutes

USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true" and torch.cuda.is_available()

# Global variables for model
model = None
model_lock = threading.Lock()
last_used_time = 0

from pydantic import BaseModel # Field not strictly needed for simple cases in V1
from typing import Optional

class ImageInput(BaseModel):
    image_data: str  # Renamed field
    threshold: Optional[float] = 0.5

def find_model_file():
    """Search for the model weights file"""
    # Check standard location
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model file found at standard path: {MODEL_PATH}")
        return MODEL_PATH
    
    # Check direct path in weights directory
    weights_dir = os.path.join(model_dir, 'weights')
    if os.path.exists(weights_dir):
        logger.info(f"Checking weights directory: {weights_dir}")
        
        # Look for files matching the pattern in main weights directory
        for file in os.listdir(weights_dir):
            if file.endswith('.pth') and 'clip_wavelet' in file:
                found_path = os.path.join(weights_dir, file)
                logger.info(f"Found model file: {found_path}")
                return found_path
    
    # Check if it's in a subfolder
    wavelet_subfolder = os.path.join(weights_dir, 'wavelet_clip_detection')
    if os.path.exists(wavelet_subfolder):
        logger.info(f"Checking subfolder: {wavelet_subfolder}")
        
        # Look for files in the subfolder
        for file in os.listdir(wavelet_subfolder):
            if file.endswith('.pth') and 'clip_wavelet' in file:
                found_path = os.path.join(wavelet_subfolder, file)
                logger.info(f"Found model file in subfolder: {found_path}")
                
                # Copy to expected location
                try:
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    shutil.copy2(found_path, MODEL_PATH)
                    logger.info(f"Copied model file to expected path: {MODEL_PATH}")
                    return MODEL_PATH
                except Exception as e:
                    logger.error(f"Error copying model file: {e}")
                    return found_path
    
    # Try a recursive search as a last resort
    logger.info("Performing recursive search for model file...")
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pth') and 'clip_wavelet' in file:
                found_path = os.path.join(root, file)
                logger.info(f"Found model file in recursive search: {found_path}")
                return found_path
    
    logger.warning(f"No model file found matching pattern 'clip_wavelet*.pth'")
    return None

def import_model_modules():
    """Import CLIPDetectorWavelet module while handling relative imports"""
    # Create a custom module-finder approach
    try:
        # Fix import issues with original code by importing specific modules first
        from training.metrics.registry import DETECTOR
        
        # Import the detector 
        from training.detectors.clip_detector_wavelet import CLIPDetectorWavelet
        
        logger.info("Successfully imported CLIPDetectorWavelet")
        return CLIPDetectorWavelet
    except ImportError as e:
        logger.error(f"Error importing model modules: {e}")
        # Try alternate import approach
        try:
            # Alternative: copy code snippets and create a simplified detector
            # For now, retry with explicit imports
            import training
            from training import detectors
            from training.detectors import clip_detector_wavelet
            return clip_detector_wavelet.CLIPDetectorWavelet
        except ImportError as e2:
            logger.error(f"Second attempt to import model failed: {e2}")
            raise

def create_detector_config():
    """Create a configuration dictionary for the detector"""
    return {
        "model_name": "clip_wavelet",
        "loss_func": "cross_entropy",
        "resolution": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

def load_model_internal():
    """Loads the Wavelet-CLIP detector model"""
    global model, last_used_time

    with model_lock:
        if model is not None:  # Check again after acquiring lock
            last_used_time = time.time()
            return

        logger.info(f"Loading Wavelet-CLIP model from {MODEL_PATH} onto {DEVICE}...")
        try:
            # Check for model file
            actual_model_path = find_model_file() or MODEL_PATH
            
            # Import model module
            CLIPDetectorWavelet = import_model_modules()
            
            # Create model configuration
            config = create_detector_config()
            
            # Initialize model
            wavelet_model = CLIPDetectorWavelet(config)
            
            # Load weights if available
            if os.path.exists(actual_model_path):
                logger.info(f"Loading weights from {actual_model_path}")
                state_dict = torch.load(actual_model_path, map_location=DEVICE)
                # Handle potential module prefix in state_dict
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                
                # Remove "module." prefix if it exists
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v
                
                wavelet_model.load_state_dict(new_state_dict, strict=False)
                logger.info("Model weights loaded successfully.")
            else:
                logger.warning(f"Model weights not found at {actual_model_path}. Using uninitialized model.")
            
            # Move to device and set to eval mode
            wavelet_model.to(DEVICE)
            wavelet_model.eval()
            
            # Assign to global variable
            model = wavelet_model
            last_used_time = time.time()
            logger.info("Wavelet-CLIP model loaded successfully.")
            
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            model = None
            raise
        finally:
            gc.collect()

def ensure_model_loaded():
    """Ensures the model is loaded, loading it if necessary."""
    global last_used_time
    if model is None:
        load_model_internal()
    else:
        last_used_time = time.time()  # Update last used time if already loaded

def unload_model_if_idle():
    """Unloads the model if it has been idle for longer than MODEL_TIMEOUT."""
    global model, last_used_time
    if model is None or PRELOAD_MODEL:  # Don't unload if preloaded or already unloaded
        return

    with model_lock:
        if model is not None and (time.time() - last_used_time > MODEL_TIMEOUT):
            logger.info(f"Unloading model due to inactivity (timeout: {MODEL_TIMEOUT}s).")
            del model
            model = None
            gc.collect()
            logger.info("Model unloaded and memory cleared.")

def preprocess_image(image_bytes):
    """Preprocess the image for model inference"""
    try:
        # Open image from bytes and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to model's expected resolution
        config = create_detector_config()
        image = image.resize((config["resolution"], config["resolution"]))
        
        # Convert to tensor and normalize
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"])
        ])
        
        # Use NumPy as a fallback if transforms fail
        try:
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as np_err:
            logger.warning(f"Transform failed, trying with NumPy: {np_err}")
            # Manual conversion to tensor
            np_img = np.array(image)
            np_img = np_img.transpose((2, 0, 1)) / 255.0  # HWC -> CHW and normalize
            # Normalize using the same parameters
            for i, (mean, std) in enumerate(zip(config["mean"], config["std"])):
                np_img[i] = (np_img[i] - mean) / std
            image_tensor = torch.from_numpy(np_img).float().unsqueeze(0)  # Add batch dimension
            
        return image_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "model_name": MODEL_NAME,
        "description": "Wavelet-CLIP model for deepfake detection",
        "model_path": MODEL_PATH,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "lazy_loading_enabled": not PRELOAD_MODEL,
        "model_timeout_seconds": MODEL_TIMEOUT if not PRELOAD_MODEL else "N/A (preloaded)"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    model_file_exists = os.path.exists(MODEL_PATH)
    status_message = "healthy"
    
    if not model_file_exists:
        # Try to find model weight file
        model_file = find_model_file()
        if model_file:
            status_message = "healthy"
            model_file_exists = True
        else:
            status_message = "error_missing_weights"
    
    return {
        "status": status_message,
        "model_name": MODEL_NAME,
        "device": str(DEVICE),
        "model_weights_found": model_file_exists,
        "model_loaded": model is not None
    }
    
@app.post("/unload")
async def unload_model():
    """Endpoint to manually unload the model"""
    global model
    
    if model is None:
        return {"status": "not_loaded", "message": "Model is not currently loaded"}
        
    logger.info("Manually unloading model")
    # Delete model and clear memory
    del model
    model = None
    # Fixed reference to USE_GPU
    if torch.cuda.is_available():  # Modified line
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model unloaded and memory cleared")
            
    return {"status": "success", "message": "Model unloaded successfully"}

@app.post("/predict")
async def predict(image_input: ImageInput):
    """
    Predict if an image is a deepfake using the Wavelet-CLIP model.
    
    Args:
        image_input: The input image data
        
    Returns:
        Dict containing prediction results
    """
    try:
        ensure_model_loaded()  # This will load the model if it's not already loaded
        
        if model is None:
            logger.error("Model components are not available for prediction.")
            raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

        start_time = time.time()

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_input.image_data)
            image_tensor = preprocess_image(image_bytes)
        except Exception as e:
            logger.error(f"Invalid image data: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        # Move tensor to the device
        image_tensor = image_tensor.to(DEVICE)

        # Prepare data dictionary for model input
        data_dict = {
            "image": image_tensor,
            "label": torch.tensor([0]).to(DEVICE)  # Dummy label for inference
        }

        # Perform inference
        with torch.no_grad():
            predictions = model(data_dict, inference=True)

        # Get probability of being fake (class 1)
        probability = predictions["prob"].item()
        
        # Apply threshold for classification
        prediction = 1 if probability >= image_input.threshold else 0
        class_label = "fake" if prediction == 1 else "real"
        
        inference_time = time.time() - start_time
        logger.info(f"Prediction completed in {inference_time:.4f}s. Prob Fake: {probability:.4f}")
        
        # Schedule model unload if not preloaded
        if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
            threading.Timer(MODEL_TIMEOUT + 5.0, unload_model_if_idle).start()

        return {
            "model": MODEL_NAME,
            "probability": float(probability),
            "prediction": int(prediction),
            "class": class_label,
            "inference_time": float(inference_time)
        }

    except HTTPException:
        raise  # Re-raise HTTPException directly
    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

@app.on_event("startup")
async def startup_event():
    """Startup event handler - preloads model if configured"""
    if PRELOAD_MODEL:
        logger.info("Preloading model at startup (PRELOAD_MODEL=true)")
        try:
            load_model_internal()
        except Exception as e:
            logger.error(f"Fatal error during model preloading: {e}. Service might not function.")
    else:
        logger.info("Model will be loaded on first request (PRELOAD_MODEL=false).")
    
    # Start a background timer to check for model unloading if not preloading
    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
        def periodic_unload_check():
            unload_model_if_idle()
            if model is None and not PRELOAD_MODEL:  # if model got unloaded and we are not preloading, stop timer
                return
            # Reschedule the check
            threading.Timer(MODEL_TIMEOUT / 2, periodic_unload_check).start()
        
        # Initial call after a short delay
        threading.Timer(MODEL_TIMEOUT / 2, periodic_unload_check).start()
        logger.info(f"Model idle check timer started (interval: {MODEL_TIMEOUT / 2}s).")

if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 5003))
    logger.info(f"Starting {MODEL_NAME} server on port {port} with CPU: {DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=port)