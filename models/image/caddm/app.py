"""
CADDM (Convolutional Artifact Detection for DeepFakes Module)
This service loads the CADDM model and exposes an API endpoint to analyze images
"""
import os
import io
import sys
import base64
import torch
import logging
import time
import glob
import threading
import gc
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from typing import Dict, Any, Optional
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Make sure all necessary modules are in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'caddm'))

# Initialize FastAPI app
app = FastAPI(
    title="CADDM Model Service",
    description="Service for detecting deepfake images using CADDM model",
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
# Try both backbones - sometimes one might work better than the other
available_backbones = ["resnet34", "efficientnet-b3", "efficientnet-b4"]
MODEL_BACKBONE = os.environ.get("MODEL_BACKBONE", "resnet34")
if MODEL_BACKBONE not in available_backbones:
    logger.warning(f"Specified backbone {MODEL_BACKBONE} not in available backbones. Using resnet34.")
    MODEL_BACKBONE = "resnet34"

CHECKPOINTS_DIR = "/app/checkpoints"
USE_GPU = torch.cuda.is_available() and os.environ.get("USE_CPU", "false").lower() != "true"
DEVICE = 'cuda' if USE_GPU else 'cpu'
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "5"))  # Seconds to keep model loaded

# Define config for test
CONFIG = {
    'crop_face': {
        'face_width': 80,
        'output_size': 224,
        'scale': 0.9
    },
    'adm_det': {
        'min_dim': 224,
        'aspect_ratios': [[1], [1], [1], [1]],
        'feature_maps': [7, 5, 3, 1],
        'steps': [32, 45, 75, 224],
        'min_sizes': [40, 80, 120, 224],
        'max_sizes': [80, 120, 160, 224],
        'clip': True,
        'variance': [0.1],
        'name': 'deepfake'
    },
    'sliding_win': {
        'prior_bbox': [[40, 80], [80, 120], [120, 160], [224, 224]]
    },
    'model': {
        'backbone': MODEL_BACKBONE
    },
    'test': {
        'batch_size': 1
    }
}

# Define request model to match other deepfake detectors
class ImageInput(BaseModel):
    image: str  # Base64 encoded image
    threshold: Optional[float] = 0.35  # Lower threshold for CADDM - seems more appropriate

# Global variable for the model
model = None
model_lock = threading.Lock()
last_used_time = 0

# Face detector for better preprocessing
face_detector = None
face_cascade = None

def init_face_detector():
    """Initialize the face detector for better preprocessing."""
    global face_cascade
    try:
        # Try to use OpenCV's built-in face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("Initialized OpenCV face detector")
        return True
    except Exception as e:
        logger.error(f"Error initializing face detector: {str(e)}")
        return False

def detect_and_align_face(image, face_size=224):
    """Detect face in the image and align/crop it for the model."""
    global face_cascade
    
    # If face detector not initialized, initialize it
    if face_cascade is None:
        init_face_detector()
    
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If faces detected
    if len(faces) > 0:
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Add a margin around the face (20% of face size)
        margin = int(0.2 * w)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop face
        face = image[y:y+h, x:x+w]
        
        # Resize to the expected size
        face = cv2.resize(face, (face_size, face_size))
        
        logger.info(f"Face detected and cropped")
        return face
    else:
        # If no face detected, just resize the whole image
        logger.warning("No face detected, using the whole image")
        return cv2.resize(image, (face_size, face_size))

def find_model_weights():
    """Find model weights for the specified backbone or any available weights."""
    # First try to find the exact backbone model
    specific_model = os.path.join(CHECKPOINTS_DIR, f"{MODEL_BACKBONE}.pkl")
    if os.path.exists(specific_model):
        logger.info(f"Found exact model file: {specific_model}")
        return specific_model
        
    # Next try to find any .pkl file in the checkpoints directory
    pkl_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pkl"))
    if pkl_files:
        logger.info(f"Found model file: {pkl_files[0]}")
        return pkl_files[0]
        
    # If not found in the root directory, search subdirectories
    logger.info(f"Searching subdirectories in {CHECKPOINTS_DIR} for model files")
    pkl_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "**", "*.pkl"), recursive=True)
    if pkl_files:
        logger.info(f"Found model file: {pkl_files[0]}")
        return pkl_files[0]
    
    # If still not found, raise an error
    logger.error(f"No model weights found in {CHECKPOINTS_DIR}. Directory contents:")
    for root, dirs, files in os.walk(CHECKPOINTS_DIR):
        logger.error(f"Directory: {root}")
        logger.error(f"Files: {files}")
    raise FileNotFoundError(f"No model weights found in {CHECKPOINTS_DIR}")

def load_model():
    """Load the CADDM model."""
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
            logger.info(f"Loading CADDM model with backbone {MODEL_BACKBONE} on {DEVICE}")
            
            # Patch EfficientNet's pretrained weights loading
            # If we're using EfficientNet, we need to modify the pretrained weights path
            if MODEL_BACKBONE.startswith('efficientnet'):
                # Import and modify the utils.load_pretrained_weights function
                import backbones.efficientnet_pytorch.utils as effnet_utils
                
                # Override the load_pretrained_weights function to use our weights
                original_load_pretrained = effnet_utils.load_pretrained_weights
                
                def patched_load_pretrained(model, model_name, load_fc=True, advprop=False):
                    # Use our own checkpoint instead
                    logger.info("Using custom EfficientNet loading logic")
                    # Find weights file
                    model_path = find_model_weights()
                    # Load state dict
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    # Handle different state dict formats
                    if 'network' in state_dict:
                        state_dict = state_dict['network']
                    
                    # Load the state dict
                    if load_fc:
                        model.load_state_dict(state_dict)
                    else:
                        # Filter out fully connected layer parameters
                        for k in list(state_dict.keys()):
                            if k.startswith('_fc'):
                                del state_dict[k]
                        res = model.load_state_dict(state_dict, strict=False)
                        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
                    
                    logger.info(f"Successfully loaded pretrained weights for {model_name}")
                    
                # Replace the function
                effnet_utils.load_pretrained_weights = patched_load_pretrained
            
            # Import model creator function
            from backbones.caddm import CADDM
            
            # Create the model
            model = CADDM(num_classes=2, backbone=MODEL_BACKBONE)
            
            # Find model weights
            model_path = find_model_weights()
            logger.info(f"Loading weights from {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # If the checkpoint contains 'network' key, extract it
            if 'network' in checkpoint:
                state_dict = checkpoint['network']
            else:
                state_dict = checkpoint
                
            try:
                model.load_state_dict(state_dict)
                logger.info("Successfully loaded state dict directly")
            except Exception as e:
                logger.warning(f"Could not load state dict directly: {str(e)}")
                logger.info("Trying to load with prefix 'module.'")
                # Some models are saved with 'module.' prefix from DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k if k.startswith('module.') else 'module.' + k
                    new_state_dict[name] = v
                
                try:
                    model.load_state_dict(new_state_dict)
                    logger.info("Successfully loaded state dict with 'module.' prefix")
                except Exception as e2:
                    logger.warning(f"Could not load state dict with 'module.' prefix: {str(e2)}")
                    logger.info("Trying to remove 'module.' prefix")
                    
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    
                    model.load_state_dict(new_state_dict)
                    logger.info("Successfully loaded state dict with 'module.' prefix removed")
            
            # Move model to GPU if available
            model = model.to(DEVICE)
            model.eval()
            
            # Initialize the face detector
            init_face_detector()
            
            # Update last used time
            last_used_time = time.time()
            
            logger.info(f"Model loaded successfully on {DEVICE}")
            
            # Clear CUDA cache to free up memory
            if USE_GPU:
                torch.cuda.empty_cache()
            gc.collect()
            
            return model
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        numpy_image = np.array(pil_image)
        
        # Convert from RGB to BGR (OpenCV format)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        
        # Detect and align face - better preprocessing than just resizing
        processed_image = detect_and_align_face(numpy_image)
        
        # Convert to torch tensor and normalize
        img_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).float() / 255.0
        
        # Apply normalization expected by the model (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(DEVICE)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "model": "CADDM",
        "description": "Convolutional Artifact Detection for DeepFakes Module",
        "backbone": MODEL_BACKBONE,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "lazy_loading": not PRELOAD_MODEL
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    global model
    
    # For health check, we don't need to load the model
    # Just check if files exist
    try:
        # Check if weights file exists
        model_path = find_model_weights()
        model_file_exists = os.path.exists(model_path)
        
        return {
            "status": "healthy" if model_file_exists else "missing_weights",
            "device": DEVICE,
            "model_loaded": model is not None,
            "lazy_loading": not PRELOAD_MODEL
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "device": DEVICE
        }

@app.post("/predict")
async def predict_image(input_data: ImageInput) -> Dict[str, Any]:
    """
    Detect if an image is a deepfake using the CADDM model.
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
        img_tensor = preprocess_image(image_bytes)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Get probabilities with softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # Probability of being fake (class 1)
            fake_probability = float(probabilities[1])
            
            # Apply threshold for classification
            prediction = 1 if fake_probability >= input_data.threshold else 0
            prediction_class = "fake" if prediction == 1 else "real"
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.4f} seconds. Predicted class: {prediction_class}, Probability: {fake_probability:.4f}")
        
        # Schedule unloading after timeout - run in background
        threading.Timer(5.0, unload_model_if_idle).start()
        
        # Return standardized format matching other models
        return {
            "model": "caddm",
            "backbone": MODEL_BACKBONE,
            "probability": fake_probability,
            "prediction": prediction,
            "class": prediction_class,
            "inference_time": inference_time
        }
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
            # Don't raise exception - allow service to start anyway
    else:
        logger.info("Model will be loaded on first request (PRELOAD_MODEL=false)")

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("MODEL_PORT", 5006))
    logger.info(f"Starting CADDM server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)