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
import torch.nn.functional as F # Import F for softmax

# --- Monkey Patching CUDA functions ---
if not torch.cuda.is_available(): 
    logger_pre_patch_ucf = logging.getLogger("pre_patch_logger_ucf") 
    logger_pre_patch_ucf.propagate = False 
    if not logger_pre_patch_ucf.hasHandlers():
        handler_ucf = logging.StreamHandler(sys.stdout)
        formatter_ucf = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s")
        handler_ucf.setFormatter(formatter_ucf)
        logger_pre_patch_ucf.addHandler(handler_ucf)
        logger_pre_patch_ucf.setLevel(logging.INFO)
    logger_pre_patch_ucf.info("Patching torch.cuda functions for CPU-only environment (ucf_deepfake_detection).")
    
    def _patched_cuda_is_available_ucf(): return False
    def _patched_cuda_get_device_name_ucf(device=None): return "CPU (Patched by ucf_app)"
    
    torch.cuda.is_available = _patched_cuda_is_available_ucf
    if hasattr(torch.cuda, 'get_device_name'): torch.cuda.get_device_name = _patched_cuda_get_device_name_ucf
    
    if hasattr(torch.cuda, '_get_device_properties'):
        class DummyDevicePropertiesUcf:
            def __init__(self): self.name = "CPU (Patched Properties by ucf_app)"
        def _patched_cuda_get_device_properties_ucf(device=None): return DummyDevicePropertiesUcf()
        torch.cuda._get_device_properties = _patched_cuda_get_device_properties_ucf

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
    force=True 
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
    logger.info("Successfully imported DETECTOR, BACKBONE, LOSSFUNC from DeepfakeBench for UCF.")
except ImportError as e:
    logger.error(f"Failed to import from DeepfakeBench for UCF: {e}", exc_info=True)
    raise

app = FastAPI(
    title="UCF Deepfake Detection Model Service",
    description="Service for detecting deepfake images using the UCF model (via DeepfakeBench registry).",
    version="1.0.2" # Version for UCF
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "ucf_deepfake_detection" 
UCF_DETECTOR_KEY_IN_REGISTRY = 'ucf' 

UCF_CONFIG_PATH = os.path.join(deepfake_bench_training_path, 'config/config/detector/ucf.yaml')
UCF_WEIGHTS_PATH = os.path.join(deepfake_bench_training_path, 'weights/ucf_faceforensics++.pth')
XCEPTION_PRETRAINED_PATH_UCF = os.path.join(deepfake_bench_training_path, 'pretrained/xception-b5690688.pth')
DLIB_PREDICTOR_FILE_PATH_FOR_CHECK_UCF = '/app/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'

DEVICE = torch.device('cpu')
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "600"))

model_instance_ucf: Optional[torch.nn.Module] = None 
model_config_loaded_ucf = None
model_lock_ucf = threading.Lock()
last_used_time_ucf = 0.0
ucf_image_preprocessor_fn: Optional[callable] = None 

from pydantic import BaseModel, Field
from typing import Optional

class ImageInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image string") # Renamed field
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")

def load_ucf_config_from_yaml(): 
    global model_config_loaded_ucf
    if model_config_loaded_ucf is None:
        logger.info(f"Loading UCF specific config from: {UCF_CONFIG_PATH}")
        if not os.path.exists(UCF_CONFIG_PATH):
            raise FileNotFoundError(f"UCF detector config file not found: {UCF_CONFIG_PATH}")
            
        with open(UCF_CONFIG_PATH, 'r') as f: detector_specific_config = yaml.safe_load(f)
        
        general_train_config_path = os.path.join(deepfake_bench_training_path, 'config/train_config.yaml')
        if os.path.exists(general_train_config_path):
            with open(general_train_config_path, 'r') as f_train: general_config_defaults = yaml.safe_load(f_train)
            merged_config = {**general_config_defaults, **detector_specific_config}
        else:
            logger.warning(f"General train_config.yaml not found. Using only UCF-specific config.")
            merged_config = detector_specific_config
        
        merged_config['pretrained'] = XCEPTION_PRETRAINED_PATH_UCF
        merged_config['resolution'] = merged_config.get('resolution', 256) 
        merged_config['mean'] = merged_config.get('mean', [0.5, 0.5, 0.5]) 
        merged_config['std'] = merged_config.get('std', [0.5, 0.5, 0.5])   
        
        if 'backbone_config' not in merged_config: merged_config['backbone_config'] = {}
        merged_config['backbone_config']['inc'] = merged_config['backbone_config'].get('inc', 3) 
        merged_config['backbone_config']['num_classes'] = merged_config['backbone_config'].get('num_classes', 2)
        merged_config['backbone_config']['dropout'] = merged_config['backbone_config'].get('dropout', False)
        merged_config['backbone_config']['mode'] = merged_config['backbone_config'].get('mode', 'adjust_channel') 
        merged_config['encoder_feat_dim'] = merged_config.get('encoder_feat_dim', 512) 
        merged_config['cuda'] = False 

        model_config_loaded_ucf = merged_config
        logger.info(f"UCF configuration fully loaded and patched for inference.")
    return model_config_loaded_ucf

def build_ucf_inference_preprocessor():
    config = load_ucf_config_from_yaml()
    resolution = config['resolution']
    mean = config['mean']
    std = config['std']

    transform_pil = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
        T.Normalize(mean=mean, std=std)
    ])
    return transform_pil

def load_model_internal_ucf(): 
    global model_instance_ucf, last_used_time_ucf, ucf_image_preprocessor_fn
    with model_lock_ucf:
        if model_instance_ucf is not None:
            last_used_time_ucf = time.time()
            return

        logger.info(f"[{MODEL_NAME_DISPLAY}] Initializing UCF model (via DeepfakeBench registry)...")
        try:
            config = load_ucf_config_from_yaml()
            config['cuda'] = False 
            
            if not os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK_UCF):
                 logger.error(f"Dlib shape predictor NOT FOUND at: {DLIB_PREDICTOR_FILE_PATH_FOR_CHECK_UCF}")
            else:
                logger.info(f"Dlib shape predictor found at: {DLIB_PREDICTOR_FILE_PATH_FOR_CHECK_UCF}")

            if UCF_DETECTOR_KEY_IN_REGISTRY not in DETECTOR.data:
                 raise RuntimeError(f"UCF detector ('{UCF_DETECTOR_KEY_IN_REGISTRY}') not registered.")
            
            ucf_model_class = DETECTOR[UCF_DETECTOR_KEY_IN_REGISTRY] 
            _model = ucf_model_class(config) 

            if not os.path.exists(UCF_WEIGHTS_PATH):
                raise FileNotFoundError(f"UCF model weights not found: {UCF_WEIGHTS_PATH}")

            checkpoint = torch.load(UCF_WEIGHTS_PATH, map_location=DEVICE) 
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                _model.load_state_dict(checkpoint['state_dict'], strict=True)
            elif isinstance(checkpoint, dict) and 'model' in checkpoint: 
                 _model.load_state_dict(checkpoint['model'], strict=True)
            else: 
                _model.load_state_dict(checkpoint, strict=True)
            
            _model.to(DEVICE) 
            _model.eval()
            
            model_instance_ucf = _model
            ucf_image_preprocessor_fn = build_ucf_inference_preprocessor()
            last_used_time_ucf = time.time()
            logger.info(f"[{MODEL_NAME_DISPLAY}] UCF Model and preprocessor loaded successfully on CPU.")
            
        except Exception as e:
            logger.exception(f"[{MODEL_NAME_DISPLAY}] Failed to load UCF model: {e}")
            model_instance_ucf, ucf_image_preprocessor_fn = None, None
            raise
        finally:
            gc.collect()

def ensure_model_loaded_ucf(): 
    if model_instance_ucf is None or ucf_image_preprocessor_fn is None:
        load_model_internal_ucf()
    else:
        global last_used_time_ucf
        last_used_time_ucf = time.time()

def unload_model_if_idle_ucf(): 
    global model_instance_ucf, last_used_time_ucf, ucf_image_preprocessor_fn
    if model_instance_ucf is None or PRELOAD_MODEL: return
    with model_lock_ucf:
        if model_instance_ucf is not None and (time.time() - last_used_time_ucf > MODEL_TIMEOUT):
            logger.info(f"[{MODEL_NAME_DISPLAY}] Unloading model due to inactivity.")
            del model_instance_ucf; del ucf_image_preprocessor_fn
            model_instance_ucf, ucf_image_preprocessor_fn = None, None
            gc.collect(); logger.info(f"[{MODEL_NAME_DISPLAY}] Model unloaded.")

@app.on_event("startup")
async def startup_event_handler_ucf(): 
    if PRELOAD_MODEL:
        logger.info(f"[{MODEL_NAME_DISPLAY}] Preloading model at startup.")
        try: load_model_internal_ucf()
        except Exception as e: logger.error(f"[{MODEL_NAME_DISPLAY}] Preloading error: {e}", exc_info=True)
    else:
        logger.info(f"[{MODEL_NAME_DISPLAY}] Lazy loading enabled for UCF model.")
    
    active_timer_ucf = None 
    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:
        def periodic_unload_check_runner_ucf():
            nonlocal active_timer_ucf 
            unload_model_if_idle_ucf()
            if model_instance_ucf is not None or PRELOAD_MODEL: 
                active_timer_ucf = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check_runner_ucf)
                active_timer_ucf.daemon = True 
                active_timer_ucf.start()
            else:
                logger.info(f"[{MODEL_NAME_DISPLAY}] Model unloaded, stopping idle check timer.")
        
        active_timer_ucf = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_unload_check_runner_ucf)
        active_timer_ucf.daemon = True
        active_timer_ucf.start()
        logger.info(f"[{MODEL_NAME_DISPLAY}] Idle check timer initiated.")

@app.get("/")
async def get_root_ucf(): 
    return {
        "model_name": MODEL_NAME_DISPLAY,
        "description": "UCF Deepfake Detection Model (via DeepfakeBench Registry)",
        "device_used": str(DEVICE),
        "model_loaded_status": model_instance_ucf is not None and ucf_image_preprocessor_fn is not None,
    }

@app.get("/health")
async def get_health_ucf(): 
    model_files_ok = (
        os.path.exists(UCF_WEIGHTS_PATH) and
        os.path.exists(UCF_CONFIG_PATH) and
        os.path.exists(XCEPTION_PRETRAINED_PATH_UCF) and
        os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK_UCF) 
    )
    
    model_and_preprocessor_loaded = model_instance_ucf is not None and ucf_image_preprocessor_fn is not None
    current_status = "healthy"
    message = "UCF service operational."

    if not torch.cuda.is_available(): 
        message += " (Running in CPU-only mode as configured by patch)."
    
    if not model_files_ok:
        current_status = "error_missing_files"
        missing_items = []
        if not os.path.exists(UCF_WEIGHTS_PATH): missing_items.append("UCF weights")
        if not os.path.exists(UCF_CONFIG_PATH): missing_items.append("UCF config")
        if not os.path.exists(XCEPTION_PRETRAINED_PATH_UCF): missing_items.append("Xception backbone weights")
        if not os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK_UCF): missing_items.append(f"Dlib predictor")
        message = f"Critical files missing: {', '.join(missing_items)}."
    elif not model_and_preprocessor_loaded and not PRELOAD_MODEL:
        current_status = "degraded_not_loaded"
        message = "UCF model ready for lazy loading."
    elif not model_and_preprocessor_loaded and PRELOAD_MODEL:
        current_status = "error_preload_failed"
        message = "UCF model was set to preload but failed. Check startup logs."
        
    return {
        "status": current_status,
        "model_name": MODEL_NAME_DISPLAY,
        "message": message,
        "torch_cuda_is_available_effective": torch.cuda.is_available(), 
        "details": {
            "ucf_weights_found": os.path.exists(UCF_WEIGHTS_PATH),
            "ucf_config_found": os.path.exists(UCF_CONFIG_PATH),
            "xception_backbone_weights_found": os.path.exists(XCEPTION_PRETRAINED_PATH_UCF),
            "dlib_shape_predictor_found": os.path.exists(DLIB_PREDICTOR_FILE_PATH_FOR_CHECK_UCF),
            "model_and_preprocessor_currently_loaded": model_and_preprocessor_loaded,
        }
    }

@app.post("/predict")
async def post_predict_ucf(input_data: ImageInput) -> Dict[str, Any]:
    try:
        ensure_model_loaded_ucf()
        if model_instance_ucf is None or ucf_image_preprocessor_fn is None:
            raise HTTPException(status_code=503, detail="UCF Model/Preprocessor not available.")

        start_time = time.time()
        image_bytes = base64.b64decode(input_data.image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        image_tensor_rgb_norm = ucf_image_preprocessor_fn(pil_image).unsqueeze(0).to(DEVICE)
        dummy_label_tensor = torch.tensor([0], device=DEVICE) 
        
        data_dict_for_model_forward = {
            'image': image_tensor_rgb_norm,
            'label': dummy_label_tensor 
        }

        with torch.no_grad():
            prediction_output_dict = model_instance_ucf.forward(data_dict_for_model_forward, inference=True)
        
        # --- Check if 'prob' key exists, if not, calculate from 'cls' ---
        if 'prob' in prediction_output_dict and prediction_output_dict['prob'] is not None:
            probability_fake = prediction_output_dict['prob'].item()
        elif 'cls' in prediction_output_dict and prediction_output_dict['cls'] is not None:
            logger.warning(f"[{MODEL_NAME_DISPLAY}] 'prob' key not found in UCFDetector output, calculating from 'cls' (logits).")
            logits = prediction_output_dict['cls'] # Assuming 'cls' contains logits [batch_size, num_classes]
            probabilities = F.softmax(logits, dim=1)
            probability_fake = probabilities[0, 1].item() # Probability of being FAKE (class 1)
        else:
            logger.error(f"[{MODEL_NAME_DISPLAY}] Neither 'prob' nor 'cls' key found in UCFDetector output for probability calculation.")
            raise ValueError("Could not determine fake probability from model output.")
            
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
        logger.error(f"[{MODEL_NAME_DISPLAY}] File not found error during UCF prediction: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Model resources missing for UCF: {e}")
    except HTTPException:
        raise 
    except Exception as e:
        logger.exception(f"[{MODEL_NAME_DISPLAY}] UCF Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during UCF prediction: {e}")

@app.post("/unload", include_in_schema=True)
async def post_unload_ucf(): 
    global model_instance_ucf, ucf_image_preprocessor_fn
    if model_instance_ucf is None: return {"status": "not_loaded", "message": "UCF model not loaded."}
    with model_lock_ucf:
        if model_instance_ucf is not None:
            del model_instance_ucf; del ucf_image_preprocessor_fn
            model_instance_ucf, ucf_image_preprocessor_fn = None, None
            gc.collect(); logger.info(f"[{MODEL_NAME_DISPLAY}] UCF Model manually unloaded.")
            return {"status": "unloaded", "message": "UCF model unloaded."}
    return {"status": "error", "message": "UCF unload failed."}

if __name__ == "__main__":
    if not logger.hasHandlers(): 
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True 
        )
        logger = logging.getLogger(__name__) 

    port = int(os.environ.get("MODEL_PORT", 5007)) 
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port} (CPU, CUDA patched)")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)