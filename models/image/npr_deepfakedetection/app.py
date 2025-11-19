"""
NPR-DeepfakeDetection Model Service
This service loads the NPR-DeepfakeDetection model and exposes an API endpoint to analyze images.
It uses the Neural Pattern Residual (NPR) mechanism as described in the original paper.
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
from PIL import Image, ImageFile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import torchvision.transforms as transforms
from typing import Dict, Any, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

MODEL_REPO_SUBDIR = 'npr_deepfakedetection'
current_dir = os.path.dirname(os.path.abspath(__file__))
model_code_path = os.path.join(current_dir, MODEL_REPO_SUBDIR)

# --- CRITICAL IMPORT: Make this fail loudly if model code isn't found ---
if model_code_path not in sys.path:
    sys.path.insert(0, model_code_path)
    logger.info(f"Added {model_code_path} to sys.path")

try:
    from networks.resnet import resnet50 # This is now a global variable
    logger.info("Successfully imported resnet50 from npr_deepfakedetection.networks")
except ImportError as e:
    logger.critical(
        f"CRITICAL: Failed to import resnet50 from {model_code_path}/networks. "
        f"Error: {e}. Service cannot start without the model definition. "
        f"Ensure '{MODEL_REPO_SUBDIR}' is correctly cloned and accessible.",
        exc_info=True
    )
    # Exit if the core model class cannot be imported, as the service is non-functional.
    # Or, you could let FastAPI start but have health checks fail catastrophically.
    # For a model service, exiting might be cleaner.
    sys.exit(f"Fatal Error: Could not import resnet50: {e}")
# --- END CRITICAL IMPORT ---


app = FastAPI(
    title="NPR-DeepfakeDetection Model Service",
    description="Service for detecting deepfake images using NPR-DeepfakeDetection model.",
    version="1.0.2" # Incremented version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "npr_deepfakedetection"
WEIGHTS_FILENAME = "NPR.pth"
MODEL_FULL_PATH = os.path.join(model_code_path, "weights", WEIGHTS_FILENAME)

USE_GPU_ENV = os.environ.get("USE_GPU", "false").lower() == "true"
DEVICE = torch.device('cuda' if USE_GPU_ENV and torch.cuda.is_available() else 'cpu')
if USE_GPU_ENV and not torch.cuda.is_available():
    logger.warning("USE_GPU is true but CUDA is not available. Falling back to CPU.")
logger.info(f"Using device: {DEVICE}")

PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600"))

from pydantic import BaseModel, Field, ConfigDict # Add ConfigDict if needed

class ImageInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image string") # Renamed field
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    model_config = ConfigDict(protected_namespaces=()) # Pydantic V2 style for config

model: Optional[torch.nn.Module] = None
model_lock = threading.Lock()
last_used_time: float = 0.0

def load_model_internal():
    global model, last_used_time
    with model_lock:
        if model is not None:
            last_used_time = time.time()
            return

        logger.info(f"Loading {MODEL_NAME} model from {MODEL_FULL_PATH} onto {DEVICE}...")
        if not os.path.exists(MODEL_FULL_PATH):
            logger.error(f"Model weights not found at {MODEL_FULL_PATH}")
            raise FileNotFoundError(f"Model weights not found: {MODEL_FULL_PATH}")

        try:
            # 'resnet50' is now guaranteed to be in the global scope if the script reached this point
            _model = resnet50(num_classes=1)
            state_dict = torch.load(MODEL_FULL_PATH, map_location=DEVICE)
            
            if all(key.startswith('module.') for key in state_dict.keys()):
                logger.info("Removing 'module.' prefix from state_dict keys.")
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
            _model.load_state_dict(state_dict)
            _model.to(DEVICE)
            _model.eval()
            
            model = _model
            last_used_time = time.time()
            logger.info(f"{MODEL_NAME} model loaded successfully to {DEVICE}.")
            
        except FileNotFoundError as e_fnf: # Should be caught by earlier check, but good to have
            logger.error(f"Model file not found during load: {e_fnf}", exc_info=True)
            model = None
            raise RuntimeError(f"Model file not found for {MODEL_NAME}: {e_fnf}") from e_fnf
        except Exception as e_load: # Catch other torch.load or model init errors
            logger.error(f"An unexpected error occurred loading {MODEL_NAME} model: {e_load}", exc_info=True)
            model = None
            raise RuntimeError(f"Model loading error for {MODEL_NAME}: {e_load}") from e_load
        finally:
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

def ensure_model_loaded():
    global last_used_time
    if model is None:
        load_model_internal()
    else:
        last_used_time = time.time()

def unload_model_if_idle():
    global model, last_used_time
    if model is None or PRELOAD_MODEL:
        return
    with model_lock:
        if model is not None and (time.time() - last_used_time > MODEL_TIMEOUT):
            logger.info(f"Unloading {MODEL_NAME} model due to inactivity (timeout: {MODEL_TIMEOUT}s).")
            del model
            model = None
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"{MODEL_NAME} model unloaded and memory cleared.")

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        raise ValueError(f"Image preprocessing failed: {e}") from e

@app.on_event("startup")
async def startup_event_handler():
    if PRELOAD_MODEL:
        logger.info(f"Preloading {MODEL_NAME} model at startup (PRELOAD_MODEL=true).")
        try:
            load_model_internal()
        except Exception as e: # Catch RuntimeError from load_model_internal
            logger.error(f"Fatal error during {MODEL_NAME} model preloading: {e}. Service might not function correctly.", exc_info=True)
    else:
        logger.info(f"{MODEL_NAME} model will be loaded on first request (PRELOAD_MODEL=false).")

    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
        active_timer_npr = None
        def periodic_unload_check_runner():
            nonlocal active_timer_npr
            unload_model_if_idle()
            if model is not None or PRELOAD_MODEL: # Keep timer running if model still loaded or preloading
                active_timer_npr = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check_runner)
                active_timer_npr.daemon = True # Allow main program to exit even if timer is active
                active_timer_npr.start()
            else:
                logger.info(f"[{MODEL_NAME}] Model unloaded, stopping idle check timer.")

        active_timer_npr = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check_runner)
        active_timer_npr.daemon = True
        active_timer_npr.start()
        logger.info(f"{MODEL_NAME} model idle check timer initiated (interval: {MODEL_TIMEOUT / 2.0}s).")

@app.get("/", tags=["Info"])
async def root_endpoint():
    return {
        "model_name": MODEL_NAME,
        "description": "NPR-based deepfake image detection, using Neural Pattern Residuals.",
        "weights_path": MODEL_FULL_PATH,
        "device_used": str(DEVICE),
        "cuda_available_on_host": torch.cuda.is_available(),
        "model_currently_loaded": model is not None,
        "lazy_loading_enabled": not PRELOAD_MODEL,
        "model_idle_timeout_seconds": MODEL_TIMEOUT if not PRELOAD_MODEL else "N/A (preloaded)"
    }

@app.get("/health", tags=["System"])
async def health_check_endpoint():
    model_file_exists = os.path.exists(MODEL_FULL_PATH)
    status_msg = "healthy"
    if not model_file_exists:
        status_msg = "error_missing_weights"
    # Check if the resnet50 symbol is available globally (means critical import succeeded)
    elif 'resnet50' not in globals() or not callable(globals()['resnet50']):
        status_msg = "error_missing_model_definition"
        
    return {
        "status": status_msg,
        "model_name": MODEL_NAME,
        "device_configured": str(DEVICE),
        "model_weights_found": model_file_exists,
        "model_definition_imported": 'resnet50' in globals() and callable(globals()['resnet50']),
        "model_loaded": model is not None
    }
    
@app.post("/unload", tags=["System"], include_in_schema=True)
async def unload_model_endpoint():
    global model
    if model is None:
        return {"status": "not_loaded", "message": f"{MODEL_NAME} model is not currently loaded."}
    with model_lock:
        if model is not None:
            logger.info(f"Manually unloading {MODEL_NAME} model via /unload endpoint.")
            del model
            model = None
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"{MODEL_NAME} model unloaded and memory cleared.")
            return {"status": "unloaded", "message": f"{MODEL_NAME} model unloaded successfully."}
        else:
            return {"status": "already_unloaded", "message": f"{MODEL_NAME} model was already unloaded by another request."}

@app.post("/predict", response_model=Dict[str, Any])
async def predict_image_endpoint(input_data: ImageInput):
    try:
        ensure_model_loaded()
        if model is None:
             logger.error(f"{MODEL_NAME} model is not available for prediction (ensure_model_loaded failed).")
             raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

        start_time_pred = time.time()
        image_bytes = base64.b64decode(input_data.image_data)
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            output_logit = model(image_tensor)
            logger.debug(f"Raw logit from model: {output_logit.item()}")
            probability = torch.sigmoid(output_logit).item()
        
        prediction = 1 if probability >= input_data.threshold else 0
        class_label = "fake" if prediction == 1 else "real"
        inference_time = time.time() - start_time_pred
        
        logger.info(f"Prediction for {MODEL_NAME} completed in {inference_time:.4f}s. Prob Fake: {probability:.4f}, Threshold: {input_data.threshold}, Class: {class_label}")
        
        return {
            "model": MODEL_NAME,
            "probability": float(probability),
            "prediction": int(prediction),
            "class": class_label,
            "inference_time": float(inference_time)
        }
    except RuntimeError as e_rt_pred: # Catch errors from load_model_internal or PyTorch
        logger.error(f"Model runtime or loading error: {e_rt_pred}", exc_info=True)
        # Check if the error message matches the one from the log
        if "local variable 'resnet50' referenced before assignment" in str(e_rt_pred) or \
           "Model loading error" in str(e_rt_pred) or \
           "Model file not found" in str(e_rt_pred):
            # This indicates a problem during model loading, likely due to import or file issues
            raise HTTPException(status_code=503, detail=f"Model failed to load: {e_rt_pred}")
        raise HTTPException(status_code=500, detail=f"Model runtime error: {e_rt_pred}")
    except ValueError as e_val_pred:
        logger.error(f"Image processing error: {e_val_pred}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Image processing error: {e_val_pred}")
    except HTTPException:
        raise
    except Exception as e_pred:
        logger.error(f"Unexpected error processing prediction for {MODEL_NAME}: {e_pred}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected prediction error: {e_pred}")

if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 5001))
    # The critical import of resnet50 is now at the top.
    # If it fails, the script exits before uvicorn.run is called.
    logger.info(f"Starting {MODEL_NAME} server on port {port} with device: {DEVICE}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)