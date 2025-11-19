import os
import io
import sys
import base64
import torch
import logging
import time
import threading
import gc
import numpy as np
import yaml
from PIL import Image, ImageFile
from torchvision import transforms as T

# --- Monkey Patching CUDA functions ---
if not torch.cuda.is_available(): 
    logger_pre_patch = logging.getLogger("pre_patch_logger_spsl") 
    logger_pre_patch.propagate = False # Prevent duplicate handlers if root logger also gets configured
    if not logger_pre_patch.hasHandlers(): # Add handler only if none exist
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s")
        handler.setFormatter(formatter)
        logger_pre_patch.addHandler(handler)
        logger_pre_patch.setLevel(logging.INFO)

    logger_pre_patch.info("Patching torch.cuda functions for CPU-only environment (spsl_deepfake_detection).")
    
    _original_cuda_is_available = torch.cuda.is_available
    _original_cuda_get_device_name = None
    if hasattr(torch.cuda, 'get_device_name'):
        _original_cuda_get_device_name = torch.cuda.get_device_name

    def _patched_cuda_is_available():
        return False

    def _patched_cuda_get_device_name(device=None):
        return "CPU (Patched by spsl_app)" 
    
    torch.cuda.is_available = _patched_cuda_is_available
    torch.cuda.get_device_name = _patched_cuda_get_device_name
    
    if hasattr(torch.cuda, '_get_device_properties'):
        _original_cuda_get_device_properties = torch.cuda._get_device_properties
        class DummyDeviceProperties:
            def __init__(self):
                self.name = "CPU (Patched Properties by spsl_app)"
                # Add other attributes if needed by DeepfakeBench
        def _patched_cuda_get_device_properties(device=None):
            return DummyDeviceProperties()
        torch.cuda._get_device_properties = _patched_cuda_get_device_properties

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True # Ensure this config takes precedence if uvicorn also configures
)
logger = logging.getLogger(__name__)

deepfake_bench_root_path = '/app/DeepfakeBench'
deepfake_bench_training_path = os.path.join(deepfake_bench_root_path, 'training')
if deepfake_bench_root_path not in sys.path: sys.path.insert(0, deepfake_bench_root_path)
if deepfake_bench_training_path not in sys.path: sys.path.insert(0, deepfake_bench_training_path)

try:
    from detectors import DETECTOR 
    from networks import BACKBONE 
    from loss import LOSSFUNC     
    logger.info("Successfully imported DETECTOR, BACKBONE, LOSSFUNC from DeepfakeBench.")
except ImportError as e:
    logger.error(f"Failed to import from DeepfakeBench: {e}", exc_info=True)
    raise

app = FastAPI(
    title="SPSL Deepfake Detection Model Service",
    description="Service for detecting deepfake images using the SPSL model (via DeepfakeBench registry).",
    version="1.0.10" # Incremented version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "spsl_deepfake_detection" 
SPSL_DETECTOR_KEY_IN_REGISTRY = 'spsl' 

SPSL_CONFIG_PATH = os.path.join(deepfake_bench_training_path, 'config/config/detector/spsl.yaml')
SPSL_WEIGHTS_PATH = os.path.join(deepfake_bench_training_path, 'weights/spsl_faceforensics++.pth')
XCEPTION_PRETRAINED_PATH = os.path.join(deepfake_bench_training_path, 'pretrained/xception-b5690688.pth')
DLIB_PREDICTOR_FILE_PATH_FOR_CHECK = '/app/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'

DEVICE = torch.device('cpu')
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600"))

model_instance: Optional[torch.nn.Module] = None 
model_config_loaded = None
model_lock = threading.Lock()
last_used_time = 0.0
spsl_rgb_preprocessor_fn: Optional[callable] = None # Renamed to clarify it's for RGB only

from pydantic import BaseModel, Field
from typing import Optional

class ImageInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image string") # Renamed field
    threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Classification threshold")

def load_spsl_config_from_yaml(): 
    global model_config_loaded
    if model_config_loaded is None:
        logger.info(f"Loading SPSL specific config from: {SPSL_CONFIG_PATH}")
        if not os.path.exists(SPSL_CONFIG_PATH):
            config_dir_to_check = os.path.dirname(SPSL_CONFIG_PATH)
            if os.path.exists(config_dir_to_check): logger.error(f"Contents of {config_dir_to_check}: {os.listdir(config_dir_to_check)}")
            else: logger.error(f"Config directory {config_dir_to_check} does not exist.")
            raise FileNotFoundError(f"SPSL detector config file not found: {SPSL_CONFIG_PATH}")
            
        with open(SPSL_CONFIG_PATH, 'r') as f: detector_specific_config = yaml.safe_load(f)
        
        general_train_config_path = os.path.join(deepfake_bench_training_path, 'config/train_config.yaml')
        if os.path.exists(general_train_config_path):
            with open(general_train_config_path, 'r') as f_train: general_config_defaults = yaml.safe_load(f_train)
            merged_config = {**general_config_defaults, **detector_specific_config}
        else:
            logger.warning(f"General train_config.yaml not found. Using only detector-specific config.")
            merged_config = detector_specific_config
        
        merged_config['pretrained'] = XCEPTION_PRETRAINED_PATH 
        merged_config['resolution'] = merged_config.get('resolution', 256)
        merged_config['mean'] = merged_config.get('mean', [0.5, 0.5, 0.5])
        merged_config['std'] = merged_config.get('std', [0.5, 0.5, 0.5])   
        
        if 'backbone_config' not in merged_config: merged_config['backbone_config'] = {}
        merged_config['backbone_config']['inc'] = 4 # SPSL's Xception expects 4 input channels internally
        merged_config['backbone_config']['num_classes'] = merged_config['backbone_config'].get('num_classes', 2)
        merged_config['backbone_config']['dropout'] = merged_config['backbone_config'].get('dropout', False)
        merged_config['backbone_config']['mode'] = merged_config['backbone_config'].get('mode', 'original') 
        merged_config['cuda'] = False 

        model_config_loaded = merged_config
    return model_config_loaded

def build_spsl_rgb_image_preprocessor(): # Renamed: this preprocessor ONLY prepares the 3-channel RGB
    config = load_spsl_config_from_yaml()
    resolution = config['resolution']
    mean = config.get('mean', [0.5, 0.5, 0.5])
    std = config.get('std', [0.5, 0.5, 0.5])

    # This transform prepares the 3-channel RGB image as the SpslDetector expects for data_dict['image']
    transform_rgb_pil = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
        T.Normalize(mean=mean, std=std)
    ])
    return transform_rgb_pil

def load_model_internal():
    global model_instance, last_used_time, spsl_rgb_preprocessor_fn
    with model_lock:
        if model_instance is not None:
            last_used_time = time.time()
            return

        logger.info(f"[{MODEL_NAME_DISPLAY}] Initializing SPSL model (via DeepfakeBench registry)...")
        try:
            config = load_spsl_config_from_yaml()
            config['cuda'] = False 
            
            if not os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK):
                 logger.error(f"Dlib shape predictor NOT FOUND at: {DLIB_PREDICTOR_FILE_PATH_FOR_CHECK} (relative to CWD /app)")
            else:
                logger.info(f"Dlib shape predictor found at: {DLIB_PREDICTOR_FILE_PATH_FOR_CHECK}")

            if SPSL_DETECTOR_KEY_IN_REGISTRY not in DETECTOR.data:
                 raise RuntimeError(f"SPSL detector ('{SPSL_DETECTOR_KEY_IN_REGISTRY}') not registered.")
            
            spsl_model_class = DETECTOR[SPSL_DETECTOR_KEY_IN_REGISTRY] 
            _model = spsl_model_class(config) 

            if not os.path.exists(SPSL_WEIGHTS_PATH):
                raise FileNotFoundError(f"SPSL model weights not found: {SPSL_WEIGHTS_PATH}")

            checkpoint = torch.load(SPSL_WEIGHTS_PATH, map_location=DEVICE) 
            _model.load_state_dict(checkpoint, strict=True) 
            
            _model.to(DEVICE) 
            _model.eval()
            
            model_instance = _model
            spsl_rgb_preprocessor_fn = build_spsl_rgb_image_preprocessor() # Correct preprocessor
            last_used_time = time.time()
            logger.info(f"[{MODEL_NAME_DISPLAY}] SPSL Model and RGB preprocessor loaded successfully on CPU.")
            
        except Exception as e:
            logger.exception(f"[{MODEL_NAME_DISPLAY}] Failed to load SPSL model: {e}")
            model_instance, spsl_rgb_preprocessor_fn = None, None
            raise
        finally:
            gc.collect()

def ensure_model_loaded():
    if model_instance is None or spsl_rgb_preprocessor_fn is None: # Check both
        load_model_internal()
    else:
        global last_used_time
        last_used_time = time.time()

def unload_model_if_idle():
    global model_instance, last_used_time, spsl_rgb_preprocessor_fn
    if model_instance is None or PRELOAD_MODEL: return
    with model_lock:
        if model_instance is not None and (time.time() - last_used_time > MODEL_TIMEOUT):
            logger.info(f"[{MODEL_NAME_DISPLAY}] Unloading model due to inactivity.")
            del model_instance; del spsl_rgb_preprocessor_fn
            model_instance, spsl_rgb_preprocessor_fn = None, None
            gc.collect(); logger.info(f"[{MODEL_NAME_DISPLAY}] Model unloaded.")

@app.on_event("startup")
async def startup_event_handler_spsl(): 
    if PRELOAD_MODEL:
        logger.info(f"[{MODEL_NAME_DISPLAY}] Preloading model at startup.")
        try: load_model_internal()
        except Exception as e: logger.error(f"[{MODEL_NAME_DISPLAY}] Preloading error: {e}", exc_info=True)
    else:
        logger.info(f"[{MODEL_NAME_DISPLAY}] Lazy loading enabled for SPSL model.")
    
    active_timer_spsl = None 
    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
        def periodic_unload_check_runner_spsl():
            nonlocal active_timer_spsl 
            unload_model_if_idle()
            if model_instance is not None or PRELOAD_MODEL: 
                active_timer_spsl = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check_runner_spsl)
                active_timer_spsl.daemon = True 
                active_timer_spsl.start()
            else:
                logger.info(f"[{MODEL_NAME_DISPLAY}] Model unloaded, stopping idle check timer.")
        
        active_timer_spsl = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check_runner_spsl)
        active_timer_spsl.daemon = True
        active_timer_spsl.start()
        logger.info(f"[{MODEL_NAME_DISPLAY}] Idle check timer initiated.")

@app.get("/")
async def get_root_spsl(): 
    return {
        "model_name": MODEL_NAME_DISPLAY,
        "description": "SPSL Deepfake Detection Model (via DeepfakeBench Registry)",
        "device_used": str(DEVICE),
        "model_loaded_status": model_instance is not None and spsl_rgb_preprocessor_fn is not None,
    }

@app.get("/health")
async def get_health_spsl(): 
    model_files_ok = (
        os.path.exists(SPSL_WEIGHTS_PATH) and
        os.path.exists(SPSL_CONFIG_PATH) and
        os.path.exists(XCEPTION_PRETRAINED_PATH) and
        os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK) 
    )
    
    model_and_preprocessor_loaded = model_instance is not None and spsl_rgb_preprocessor_fn is not None
    current_status = "healthy"
    message = "SPSL service operational."

    if not torch.cuda.is_available(): 
        message += " (Running in CPU-only mode as configured by patch)."
    
    if not model_files_ok:
        current_status = "error_missing_files"
        missing_items = []
        if not os.path.exists(SPSL_WEIGHTS_PATH): missing_items.append("SPSL weights")
        if not os.path.exists(SPSL_CONFIG_PATH): missing_items.append("SPSL config")
        if not os.path.exists(XCEPTION_PRETRAINED_PATH): missing_items.append("Xception backbone weights")
        if not os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK): missing_items.append(f"Dlib predictor at {DLIB_PREDICTOR_FILE_PATH_FOR_CHECK}")
        message = f"Critical files missing: {', '.join(missing_items)}."
    elif not model_and_preprocessor_loaded and not PRELOAD_MODEL:
        current_status = "degraded_not_loaded"
        message = "SPSL model ready for lazy loading."
    elif not model_and_preprocessor_loaded and PRELOAD_MODEL:
        current_status = "error_preload_failed"
        message = "SPSL model was set to preload but failed. Check startup logs."
        
    return {
        "status": current_status,
        "model_name": MODEL_NAME_DISPLAY,
        "message": message,
        "torch_cuda_is_available_effective": torch.cuda.is_available(), 
        "details": {
            "spsl_weights_found": os.path.exists(SPSL_WEIGHTS_PATH),
            "spsl_config_found": os.path.exists(SPSL_CONFIG_PATH),
            "xception_backbone_weights_found": os.path.exists(XCEPTION_PRETRAINED_PATH),
            "dlib_shape_predictor_found (at /app/preprocessing/...)": os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK),
            "model_and_preprocessor_currently_loaded": model_and_preprocessor_loaded,
        }
    }

@app.post("/predict")
async def post_predict_spsl(input_data: ImageInput) -> Dict[str, Any]:
    try:
        ensure_model_loaded()
        if model_instance is None or spsl_rgb_preprocessor_fn is None:
            raise HTTPException(status_code=503, detail="SPSL Model/Preprocessor not available.")

        start_time = time.time()
        image_bytes = base64.b64decode(input_data.image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess to get the 3-channel normalized RGB tensor
        image_tensor_rgb_norm = spsl_rgb_preprocessor_fn(pil_image).unsqueeze(0).to(DEVICE)
        
        # The SpslDetector.forward() method itself handles phase generation and concatenation
        data_dict_for_model_forward = {'image': image_tensor_rgb_norm}

        with torch.no_grad():
            prediction_output_dict = model_instance.forward(data_dict_for_model_forward, inference=True)
        
        probability_fake = prediction_output_dict['prob'].item()
        prediction_class = 1 if probability_fake >= input_data.threshold else 0
        
        inference_time_seconds = time.time() - start_time
        logger.info(f"[{MODEL_NAME_DISPLAY}] Prediction: P(Fake)={probability_fake:.4f}, Time={inference_time_seconds:.4f}s")

        return {
            "model": MODEL_NAME_DISPLAY,
            "probability": float(probability_fake),
            "prediction": int(prediction_class),
            "class": "fake" if prediction_class == 1 else "real",
            "inference_time": float(inference_time_seconds)
        }
    except FileNotFoundError as e:
        logger.error(f"[{MODEL_NAME_DISPLAY}] File not found during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Model resources missing: {e}")
    except HTTPException:
        raise 
    except Exception as e:
        logger.exception(f"[{MODEL_NAME_DISPLAY}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during SPSL prediction: {e}")

@app.post("/unload", include_in_schema=True)
async def post_unload_spsl(): 
    global model_instance, spsl_rgb_preprocessor_fn
    if model_instance is None: return {"status": "not_loaded", "message": "SPSL model not loaded."}
    with model_lock:
        if model_instance is not None:
            del model_instance; del spsl_rgb_preprocessor_fn
            model_instance, spsl_rgb_preprocessor_fn = None, None
            gc.collect(); logger.info(f"[{MODEL_NAME_DISPLAY}] SPSL Model manually unloaded.")
            return {"status": "unloaded", "message": "SPSL model unloaded."}
    return {"status": "error", "message": "SPSL unload failed."}

if __name__ == "__main__":
    if not logger.hasHandlers(): 
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True 
        )
        logger = logging.getLogger(__name__) 

    port = int(os.environ.get("MODEL_PORT", 5006))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server directly on port {port} (CPU explicitly, CUDA patched)")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)