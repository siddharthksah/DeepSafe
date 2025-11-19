import os
import io
import sys
import base64
import torch
import logging
import time
import threading
import gc
from PIL import Image, ImageFile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from typing import Dict, Any, Optional

from contextlib import asynccontextmanager

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Model-specific imports from the cloned repository
# PYTHONPATH is set in Dockerfile to find 'model_code.src' or 'src' directly
try:
    from src.config import Config
    from src.model.dfdet import DeepfakeDetectionModel
    from lightning.fabric import Fabric
except ImportError as e:
    logger.error(f"Error importing model-specific modules: {e}. Check PYTHONPATH.")
    # Attempt to add path manually if Docker's PYTHONPATH isn't picked up as expected in some envs
    current_dir_for_app = os.path.dirname(os.path.abspath(__file__))
    model_code_path = os.path.join(current_dir_for_app, 'model_code')
    if model_code_path not in sys.path:
        sys.path.insert(0, model_code_path)
        logger.info(f"Added {model_code_path} to sys.path")
    
    # Retry imports
    from src.config import Config
    from src.model.dfdet import DeepfakeDetectionModel
    from lightning.fabric import Fabric

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    if PRELOAD_MODEL:
        logger.info("Preloading model at startup (PRELOAD_MODEL=true)")
        try:
            load_model_internal()
        except Exception as e:
            logger.error(f"Fatal error during model preloading: {e}. Service might not function.")
    else:
        logger.info("Model will be loaded on first request (PRELOAD_MODEL=false).")

    # Start a background timer to check for model unloading if not preloading
    timer_thread = None
    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
        def periodic_unload_check():
            unload_model_if_idle()
            # Reschedule the check only if the model is still loaded or the app is running
            # A more robust way would be to manage the timer cancellation in the 'yield' part
            if model is not None and not PRELOAD_MODEL: # Check if model still exists
                 timer_thread = threading.Timer(MODEL_TIMEOUT / 2, periodic_unload_check)
                 timer_thread.start()
        
        # Initial call after a short delay
        timer_thread = threading.Timer(MODEL_TIMEOUT / 2, periodic_unload_check)
        timer_thread.start()
        logger.info(f"Model idle check timer started (interval: {MODEL_TIMEOUT / 2}s).")
    
    yield  # Application runs here
    
    # Code to run on shutdown
    logger.info("Shutting down Yermandy CLIP detection service.")
    if timer_thread and timer_thread.is_alive():
        timer_thread.cancel()
        logger.info("Cancelled model idle check timer.")
    # Ensure model is unloaded on shutdown if it exists
    global model, preprocessing_fn, fabric
    if model is not None:
        logger.info("Unloading model on shutdown.")
        del model, preprocessing_fn, fabric
        model, preprocessing_fn, fabric = None, None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Yermandy CLIP Deepfake Detection Model Service",
    description="Service for detecting deepfake images using the Yermandy CLIP-based model.",
    version="1.0.0",
    lifespan=lifespan # Add the lifespan manager here
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Configuration & Globals ---
MODEL_NAME = "yermandy_clip_detection"
MODEL_PATH = os.environ.get("MODEL_PATH", "model_code/weights/model.ckpt")
DEVICE = torch.device('cpu') # Forcing CPU as per project requirements
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600")) # Default 10 minutes

USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true" and torch.cuda.is_available()

# Global variables for the model and related components
model: Optional[DeepfakeDetectionModel] = None
preprocessing_fn: Optional[callable] = None
fabric: Optional[Fabric] = None
model_lock = threading.Lock()
last_used_time = 0

from pydantic import BaseModel, Field # ConfigDict not needed here if no extra config

class ImageInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image string") # Renamed field
    threshold: Optional[float] = Field(0.35, ge=0.0, le=1.0, description="Classification threshold")

def load_model_internal():
    """Loads the deepfake detection model and its components."""
    global model, preprocessing_fn, fabric, last_used_time

    with model_lock:
        if model is not None: # Check again after acquiring lock
            last_used_time = time.time()
            return

        logger.info(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model weights not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu") # Load to CPU first
            
            # Ensure hyper_parameters are available
            if "hyper_parameters" not in ckpt:
                logger.error("Checkpoint does not contain 'hyper_parameters'. Cannot initialize model.")
                raise ValueError("Invalid checkpoint: missing 'hyper_parameters'")

            model_config = Config(**ckpt["hyper_parameters"])
            
            _model = DeepfakeDetectionModel(model_config)
            _model.load_state_dict(ckpt["state_dict"])
            _model.eval() # Set to evaluation mode
            _model.to(DEVICE) # Move to CPU

            _preprocessing_fn = _model.get_preprocessing()
            
            # For CPU, Fabric precision should be "32-true" or similar, not from checkpoint if it's like "bf16-mixed"
            # The original inference.py loads precision from ckpt["hyper_parameters"]["precision"]
            # For CPU, it's safer to override this.
            _fabric = Fabric(accelerator="cpu", devices=1, precision="32-true")
            # fabric.launch() is not needed here as we are not in a distributed script
            _model = _fabric.setup_module(_model) # Prepare model with Fabric

            # Assign to global variables
            model = _model
            preprocessing_fn = _preprocessing_fn
            fabric = _fabric
            
            last_used_time = time.time()
            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            # Reset globals if loading failed
            model = None
            preprocessing_fn = None
            fabric = None
            raise
        finally:
            gc.collect()


def ensure_model_loaded():
    """Ensures the model is loaded, loading it if necessary."""
    global last_used_time
    if model is None:
        load_model_internal()
    else:
        last_used_time = time.time() # Update last used time if already loaded

def unload_model_if_idle():
    """Unloads the model if it has been idle for longer than MODEL_TIMEOUT."""
    global model, preprocessing_fn, fabric, last_used_time
    if model is None or PRELOAD_MODEL: # Don't unload if preloaded or already unloaded
        return

    with model_lock:
        if model is not None and (time.time() - last_used_time > MODEL_TIMEOUT):
            logger.info(f"Unloading model due to inactivity (timeout: {MODEL_TIMEOUT}s).")
            del model
            del preprocessing_fn
            del fabric
            model = None
            preprocessing_fn = None
            fabric = None
            gc.collect()
            logger.info("Model unloaded and memory cleared.")

# --- FastAPI Endpoints ---
@app.on_event("startup")
async def startup_event():
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
            if model is None and not PRELOAD_MODEL : # if model got unloaded and we are not preloading, stop timer
                return
            # Reschedule the check
            threading.Timer(MODEL_TIMEOUT / 2, periodic_unload_check).start() 
        
        # Initial call after a short delay
        threading.Timer(MODEL_TIMEOUT / 2, periodic_unload_check).start()
        logger.info(f"Model idle check timer started (interval: {MODEL_TIMEOUT / 2}s).")


@app.get("/")
async def root():
    return {
        "model_name": MODEL_NAME,
        "description": "Deepfake detection model based on Yermandy's CLIP work.",
        "model_path": MODEL_PATH,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "lazy_loading_enabled": not PRELOAD_MODEL,
        "model_timeout_seconds": MODEL_TIMEOUT if not PRELOAD_MODEL else "N/A (preloaded)"
    }

@app.get("/health")
async def health():
    model_file_exists = os.path.exists(MODEL_PATH)
    status_message = "healthy"
    if not model_file_exists:
        status_message = "error_missing_weights"
    # Could add a quick inference test here if model is loaded for a more thorough check
    
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
    # Check if cuda is available instead of using undefined USE_GPU
    if torch.cuda.is_available():  # Modified line
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model unloaded and memory cleared")
            
    return {"status": "success", "message": "Model unloaded successfully"}

@app.post("/predict", response_model=Dict[str, Any])
async def predict(image_input: ImageInput):
    try:
        ensure_model_loaded() # This will load the model if it's not already loaded
        
        if model is None or preprocessing_fn is None or fabric is None:
             logger.error("Model components are not available for prediction.")
             raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

        start_time = time.time()

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_input.image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Invalid image data: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        # Preprocess the image
        # The preprocessing_fn from the model expects a PIL image and returns a tensor.
        # The model's forward pass expects a batch, so unsqueeze(0).
        image_tensor = preprocessing_fn(pil_image).unsqueeze(0)
        
        # Move tensor to the device Fabric prepared the model for (CPU in this case)
        # The dtype should be handled by Fabric setup_module or .to(DEVICE)
        image_tensor = image_tensor.to(DEVICE) # fabric.to_device(image_tensor) could also be used if fabric object is accessible

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor) # model is already setup by fabric

        # Process output
        # output.logits_labels is what the original inference.py uses
        # It's a tensor of shape [batch_size, num_classes], e.g., [1, 2] for [prob_real, prob_fake]
        probabilities_tensor = output.logits_labels.softmax(dim=1)
        
        # Probability of being FAKE is the second element (index 1)
        probability_fake = probabilities_tensor[0, 1].item() 
        
        prediction = 1 if probability_fake >= image_input.threshold else 0
        class_label = "fake" if prediction == 1 else "real"
        
        inference_time_seconds = time.time() - start_time
        logger.info(f"Prediction for {MODEL_NAME} completed in {inference_time_seconds:.4f}s. Prob Fake: {probability_fake:.4f}")
        
        # Schedule model unload if not preloaded
        if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
             threading.Timer(MODEL_TIMEOUT + 5.0, unload_model_if_idle).start() # Check slightly after timeout

        return {
            "model": MODEL_NAME,
            "probability": probability_fake,
            "prediction": prediction,
            "class": class_label,
            "inference_time": inference_time_seconds
        }

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=503, detail=f"Model weights missing: {e}")
    except HTTPException:
        raise # Re-raise HTTPException directly
    except Exception as e:
        logger.exception(f"Error during prediction: {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 5002))
    logger.info(f"Starting {MODEL_NAME} server on port {port} with CPU: {DEVICE}")
    # When running app.py directly, PYTHONPATH might need to be set if model_code is not in the same dir
    # This is handled by Dockerfile's ENV PYTHONPATH for containerized execution.
    
    # For local dev, if model_code is sibling to this app.py's dir (e.g. in models/image/yermandy_clip_detection/)
    # and model_code is cloned as models/image/yermandy_clip_detection/model_code/
    # local_mc_path = os.path.join(os.path.dirname(__file__), 'model_code')
    # if os.path.isdir(local_mc_path) and local_mc_path not in sys.path:
    #    sys.path.insert(0, local_mc_path)

    uvicorn.run(app, host="0.0.0.0", port=port)