import os
import io
import sys
import base64
import torch
import torch.nn.functional as F
import logging
import time
import threading
import gc
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Add cloned repository's src to Python path ---
CLONED_MODEL_SRC_PATH = '/app/trufor_model_code/test_docker/src'
if CLONED_MODEL_SRC_PATH not in sys.path:
    sys.path.insert(0, CLONED_MODEL_SRC_PATH)

try:
    from config import _C as config_node, update_config
    from models.cmx.builder_np_conf import myEncoderDecoder
except ImportError as e:
    logger.error(f"Failed to import TruFor modules from '{CLONED_MODEL_SRC_PATH}'. Error: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise RuntimeError(f"TruFor core modules failed to import: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="TruFor Model Service (GitHub Version - CPU)",
    description="Service for detecting deepfake images using the TruFor model from GitHub, running on CPU.",
    version="1.0.1" #Increment version to signify CPU change
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
MODEL_NAME = "trufor" 
CONFIG_YAML_PATH = os.path.join(CLONED_MODEL_SRC_PATH, 'trufor.yaml')
MODEL_WEIGHTS_PATH = '/app/weights/trufor.pth.tar'

# Determine device based on environment variable, ensuring CPU
USE_GPU_ENV = os.environ.get("USE_GPU", "false").lower() == "true"
DEVICE = torch.device('cpu') # Hardcode to CPU as per user requirement

USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true" and torch.cuda.is_available()

if USE_GPU_ENV and torch.cuda.is_available():
    logger.warning("TruFor Service: USE_GPU=true and CUDA is available, but service is configured for CPU ONLY. Using CPU.")
elif USE_GPU_ENV and not torch.cuda.is_available():
    logger.info("TruFor Service: USE_GPU=true but CUDA not available. Using CPU as intended.")
else:
    logger.info("TruFor Service: Using CPU as USE_GPU=false or not set.")


PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600")) 

model_instance: Optional[myEncoderDecoder] = None
model_config_loaded = None
model_lock = threading.Lock()
last_used_time = 0

from pydantic import BaseModel, Field # Field is fine in V1 too
from typing import Optional

class ImageInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image string") # Renamed field
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")


def load_model_internal():
    global model_instance, model_config_loaded, last_used_time
    
    with model_lock:
        if model_instance is not None:
            last_used_time = time.time()
            return

        logger.info(f"Loading TruFor model from {MODEL_WEIGHTS_PATH} onto {DEVICE}...")
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            logger.error(f"Model weights not found at {MODEL_WEIGHTS_PATH}")
            raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")
        if not os.path.exists(CONFIG_YAML_PATH):
            logger.error(f"Model config YAML not found at {CONFIG_YAML_PATH}")
            raise FileNotFoundError(f"Model config YAML not found: {CONFIG_YAML_PATH}")

        try:
            cfg = config_node.clone() 
            cfg.defrost()
            cfg.merge_from_file(CONFIG_YAML_PATH)
            cfg.TEST.MODEL_FILE = MODEL_WEIGHTS_PATH 
            cfg.GPUS = (-1,) # Explicitly tell config to use CPU
            cfg.freeze()
            model_config_loaded = cfg

            _model = myEncoderDecoder(cfg=model_config_loaded)
            
            checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE) # Ensure map_location is CPU
            if 'state_dict' in checkpoint:
                _model.load_state_dict(checkpoint['state_dict'])
            else:
                _model.load_state_dict(checkpoint)
            
            _model = _model.to(DEVICE)
            _model.eval()

            model_instance = _model
            last_used_time = time.time()
            logger.info(f"TruFor model loaded successfully to {DEVICE}.")

        except Exception as e:
            logger.exception(f"Failed to load TruFor model: {e}")
            model_instance = None 
            raise
        finally:
            gc.collect() # General garbage collection


def ensure_model_loaded():
    if model_instance is None:
        load_model_internal()
    else:
        global last_used_time
        last_used_time = time.time()


def unload_model_if_idle():
    global model_instance, last_used_time
    if model_instance is None or PRELOAD_MODEL:
        return

    with model_lock:
        if model_instance is not None and (time.time() - last_used_time > MODEL_TIMEOUT):
            logger.info(f"Unloading TruFor model due to inactivity (timeout: {MODEL_TIMEOUT}s).")
            del model_instance
            model_instance = None
            gc.collect()
            logger.info("TruFor model unloaded and memory cleared.")


def preprocess_image_trufor(image_bytes: bytes) -> torch.Tensor:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_image) 
        img_tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float) / 256.0 
        return img_tensor.unsqueeze(0).to(DEVICE) 
    except Exception as e:
        logger.error(f"Error preprocessing image for TruFor: {e}")
        raise ValueError(f"Image preprocessing failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    if PRELOAD_MODEL:
        logger.info("Preloading TruFor model at startup (PRELOAD_MODEL=true)")
        try:
            load_model_internal()
        except Exception as e:
            logger.error(f"Fatal error during TruFor model preloading: {e}. Service might not function.")
    else:
        logger.info("TruFor model will be loaded on first request (PRELOAD_MODEL=false).")
    
    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
        def periodic_unload_check():
            unload_model_if_idle()
            if model_instance is None and not PRELOAD_MODEL:
                return 
            threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check).start()
        
        threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check).start()
        logger.info(f"TruFor model idle check timer started (interval: {MODEL_TIMEOUT / 2.0}s).")


@app.get("/", tags=["Info"])
async def root():
    return {
        "model_name": MODEL_NAME,
        "description": "TruFor Deepfake Detection Model (from GitHub, CPU Version)",
        "weights_path": MODEL_WEIGHTS_PATH,
        "device": str(DEVICE),
        "model_loaded": model_instance is not None,
        "lazy_loading_enabled": not PRELOAD_MODEL,
    }


@app.get("/health", tags=["System"])
async def health():
    weights_exist = os.path.exists(MODEL_WEIGHTS_PATH)
    config_exist = os.path.exists(CONFIG_YAML_PATH)
    status_msg = "healthy"
    if not weights_exist:
        status_msg = "error_missing_weights"
    elif not config_exist:
        status_msg = "error_missing_config"
    
    return {
        "status": status_msg,
        "model_name": MODEL_NAME,
        "device": str(DEVICE),
        "model_weights_found": weights_exist,
        "model_config_found": config_exist,
        "model_loaded": model_instance is not None
    }


@app.post("/predict", response_model=Dict[str, Any])
async def predict(input_data: ImageInput):
    try:
        ensure_model_loaded()
        if model_instance is None:
             logger.error("TruFor model is not available for prediction.")
             raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

        start_time = time.time()

        image_bytes = base64.b64decode(input_data.image_data)
        rgb_tensor = preprocess_image_trufor(image_bytes)

        with torch.no_grad():
            outputs = model_instance(rgb_tensor) 
            pred_map_logits, conf_map_logits, det_score_logits, _ = outputs

            if det_score_logits is None:
                logger.error("Detection score (det_score_logits) is None from the model.")
                if pred_map_logits is not None:
                    prob_fake = torch.sigmoid(pred_map_logits).mean().item() # Use softmax if multi-class logits
                    logger.warning(f"Using mean of localization map as fake probability: {prob_fake}")
                else:
                    raise HTTPException(status_code=500, detail="Model did not return a usable detection score or localization map.")
            else:
                prob_fake = torch.sigmoid(det_score_logits).item()
        
        prediction = 1 if prob_fake >= input_data.threshold else 0
        class_label = "fake" if prediction == 1 else "real"
        
        inference_time_seconds = time.time() - start_time
        logger.info(f"Prediction for {MODEL_NAME} completed in {inference_time_seconds:.4f}s. Prob Fake: {prob_fake:.4f}")
        
        if not PRELOAD_MODEL:
            threading.Timer(MODEL_TIMEOUT + 5.0, unload_model_if_idle).start()

        return {
            "model": MODEL_NAME,
            "probability": float(prob_fake),
            "prediction": int(prediction),
            "class": class_label,
            "inference_time": float(inference_time_seconds)
        }

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=503, detail=f"Model resources missing: {e}")
    except HTTPException:
        raise 
    except Exception as e:
        logger.exception(f"Error during TruFor prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")


@app.post("/unload", include_in_schema=True)
async def unload_model_endpoint():
    global model_instance
    if model_instance is None:
        return {"status": "not_loaded", "message": "TruFor model is not currently loaded."}
    with model_lock:
        if model_instance is not None:
            logger.info("Manually unloading TruFor model via /unload endpoint.")
            del model_instance
            model_instance = None
            gc.collect()
            logger.info("TruFor model unloaded and memory cleared.")
    return {"status": "unloaded", "message": "TruFor model unloaded successfully."}


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 5005))
    logger.info(f"Starting TruFor (GitHub version - CPU) API server on port {port} with device: {DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)