"""
Faceforensics_plus_plus Model Service
This service loads the Faceforensics++ model from HongguLiu and exposes an API endpoint to analyze images
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
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Faceforensics_plus_plus Model Service",
    description="Service for detecting deepfake images using Faceforensics++ model",
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

# Add model path to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'faceforensics_plus_plus'))

try:
    # Import after path setup
    from faceforensics_plus_plus.network.models import model_selection
    logger.info("Successfully imported model_selection")
except Exception as e:
    logger.error(f"Error importing model_selection: {str(e)}")
    raise ImportError(f"Failed to import model_selection: {str(e)}")

# Find model files by scanning the models directory
def find_model_file(pattern):
    """Find a model file matching the given pattern"""
    models_dir = "/app/models"
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if pattern in file:
                return os.path.join(root, file)
    return None

# Model paths and settings
C0_MODEL_PATH = find_model_file("c0") or find_model_file("deepfake") or "/app/models/deepfake_c0_xception.pkl"
C23_MODEL_PATH = find_model_file("c23") or "/app/models/ffpp_c23.pth"
C40_MODEL_PATH = find_model_file("c40") or "/app/models/ffpp_c40.pth"

# Default to c23 model if specified, otherwise use all models for ensemble
MODEL_TYPE = os.environ.get("MODEL_TYPE", "ensemble")
if MODEL_TYPE == "c0":
    DEFAULT_MODEL_PATH = C0_MODEL_PATH
elif MODEL_TYPE == "c23":
    DEFAULT_MODEL_PATH = C23_MODEL_PATH
elif MODEL_TYPE == "c40":
    DEFAULT_MODEL_PATH = C40_MODEL_PATH
else:  # Default to ensemble
    DEFAULT_MODEL_PATH = None
    MODEL_TYPE = "ensemble"

USE_GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU", False)
DEVICE = 'cuda' if USE_GPU else 'cpu'
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "300"))  # Seconds to keep model loaded

if MODEL_TYPE == "ensemble":
    logger.info(f"Using ensemble of all models")
else:
    logger.info(f"Using model type: {MODEL_TYPE}")

# Define request model
class ImageInput(BaseModel):
    image: str  # Base64 encoded image
    threshold: Optional[float] = 0.5  # Optional threshold for classification
    model_type: Optional[str] = None  # Optional model type to use for this prediction
    
    model_config = {
        "protected_namespaces": ()  # Disable protected namespace warning
    }

# Global variables for models
c0_model = None
c23_model = None
c40_model = None
model_locks = {"c0": threading.Lock(), "c23": threading.Lock(), "c40": threading.Lock()}
last_used_times = {"c0": 0, "c23": 0, "c40": 0}

def load_model(model_path, model_type):
    """Load a specific model from path."""
    global c0_model, c23_model, c40_model
    
    if model_type == "c0" and c0_model is not None:
        last_used_times["c0"] = time.time()
        return c0_model
    elif model_type == "c23" and c23_model is not None:
        last_used_times["c23"] = time.time()
        return c23_model
    elif model_type == "c40" and c40_model is not None:
        last_used_times["c40"] = time.time()
        return c40_model
        
    with model_locks[model_type]:  # Thread safety for concurrent requests
        # Check again after acquiring the lock
        if model_type == "c0" and c0_model is not None:
            last_used_times["c0"] = time.time()
            return c0_model
        elif model_type == "c23" and c23_model is not None:
            last_used_times["c23"] = time.time()
            return c23_model
        elif model_type == "c40" and c40_model is not None:
            last_used_times["c40"] = time.time()
            return c40_model
    
        try:
            logger.info(f"Loading {model_type} model from {model_path} on {DEVICE}")
            
            # Check if the model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            # Load the Xception model
            model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=DEVICE)
            
            # Handle various state dict formats
            if isinstance(state_dict, dict):
                # Handle the case where the state dict might be nested
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Some models might have 'module.' prefix for DataParallel
                has_module_prefix = any(['module.' in key for key in state_dict.keys()])
                
                if has_module_prefix:
                    logger.info(f"Removing 'module.' prefix from {model_type} state dict keys")
                    # Remove 'module.' prefix
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace('module.', '')
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
            
            # Try to load the state dict
            try:
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded {model_type} state dict")
            except Exception as e:
                logger.error(f"Error loading {model_type} state dict: {str(e)}")
                raise ValueError(f"Could not load state dict for {model_type} model: {str(e)}")
            
            if USE_GPU:
                model.cuda()
                logger.info(f"Model {model_type} moved to GPU")
            
            model.eval()
            
            # Save to global variable
            if model_type == "c0":
                c0_model = model
            elif model_type == "c23":
                c23_model = model
            elif model_type == "c40":
                c40_model = model
                
            # Update last used time
            last_used_times[model_type] = time.time()
            
            # Clear CUDA cache to free up memory
            if USE_GPU:
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Model {model_type} loaded successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {str(e)}")
            raise

def unload_model_if_idle(model_type):
    """Unload model if it's been idle for too long."""
    global c0_model, c23_model, c40_model
    
    if model_type == "c0" and c0_model is None:
        return
    elif model_type == "c23" and c23_model is None:
        return
    elif model_type == "c40" and c40_model is None:
        return
        
    if time.time() - last_used_times[model_type] > MODEL_TIMEOUT:
        with model_locks[model_type]:
            if model_type == "c0" and c0_model is not None and time.time() - last_used_times[model_type] > MODEL_TIMEOUT:
                logger.info(f"Unloading {model_type} model after {MODEL_TIMEOUT} seconds of inactivity")
                # Delete model and clear memory
                del c0_model
                c0_model = None
                # Clear CUDA cache
                if USE_GPU:
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"Model {model_type} unloaded and memory cleared")
            elif model_type == "c23" and c23_model is not None and time.time() - last_used_times[model_type] > MODEL_TIMEOUT:
                logger.info(f"Unloading {model_type} model after {MODEL_TIMEOUT} seconds of inactivity")
                # Delete model and clear memory
                del c23_model
                c23_model = None
                # Clear CUDA cache
                if USE_GPU:
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"Model {model_type} unloaded and memory cleared")
            elif model_type == "c40" and c40_model is not None and time.time() - last_used_times[model_type] > MODEL_TIMEOUT:
                logger.info(f"Unloading {model_type} model after {MODEL_TIMEOUT} seconds of inactivity")
                # Delete model and clear memory
                del c40_model
                c40_model = None
                # Clear CUDA cache
                if USE_GPU:
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"Model {model_type} unloaded and memory cleared")

def load_all_models():
    """Load all three models."""
    global c0_model, c23_model, c40_model
    
    # Track which models loaded successfully
    success = []
    
    # Load c0 model
    try:
        load_model(C0_MODEL_PATH, "c0")
        success.append("c0")
    except Exception as e:
        logger.error(f"Failed to load c0 model: {str(e)}")
    
    # Load c23 model
    try:
        load_model(C23_MODEL_PATH, "c23")
        success.append("c23")
    except Exception as e:
        logger.error(f"Failed to load c23 model: {str(e)}")
    
    # Load c40 model
    try:
        load_model(C40_MODEL_PATH, "c40")
        success.append("c40")
    except Exception as e:
        logger.error(f"Failed to load c40 model: {str(e)}")
    
    # Check if at least one model was loaded
    if not success:
        raise ValueError("Failed to load any models")
    
    logger.info(f"Successfully loaded models: {', '.join(success)}")
    
    return success

def preprocess_image(image_bytes):
    """Preprocess the image for the model."""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply the same preprocessing as in the original code
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        return preprocess(image).unsqueeze(0)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_with_model(model, image_tensor, model_type, threshold=0.5):
    """Run prediction with a specific model."""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                # Some models return multiple outputs
                outputs = outputs[0]
            
            # Ensure outputs is the right shape
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            
            # Get predictions
            if outputs.shape[1] >= 2:  # Binary classification
                _, preds = torch.max(outputs.data, 1)
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probability = probabilities[0][1].item()  # Probability of fake class
            else:  # Single output (sigmoid)
                probability = torch.sigmoid(outputs[0]).item()
                preds = torch.tensor([1 if probability >= threshold else 0])
            
            # Apply threshold
            prediction = 1 if probability >= threshold else 0
            
            return {
                "model_type": model_type,
                "probability": float(probability),
                "prediction": int(prediction),
                "class": "fake" if prediction == 1 else "real"
            }
    except Exception as e:
        logger.error(f"Error during prediction with {model_type} model: {str(e)}")
        raise ValueError(f"Prediction failed for {model_type} model: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    # Check which model files are available
    model_files = {
        "c0": os.path.exists(C0_MODEL_PATH),
        "c23": os.path.exists(C23_MODEL_PATH),
        "c40": os.path.exists(C40_MODEL_PATH)
    }
    
    # Get model paths
    model_paths = {
        "c0": C0_MODEL_PATH,
        "c23": C23_MODEL_PATH,
        "c40": C40_MODEL_PATH
    }
    
    # Get loaded models
    loaded_models = []
    if c0_model is not None:
        loaded_models.append("c0")
    if c23_model is not None:
        loaded_models.append("c23")
    if c40_model is not None:
        loaded_models.append("c40")
    
    return {
        "model": "Faceforensics_plus_plus",
        "description": "Xception-based deepfake image detection from Faceforensics++",
        "mode": MODEL_TYPE,
        "available_models": model_files,
        "loaded_models": loaded_models,
        "model_paths": model_paths,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "lazy_loading": not PRELOAD_MODEL
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check which model files are available
    model_files = {
        "c0": os.path.exists(C0_MODEL_PATH),
        "c23": os.path.exists(C23_MODEL_PATH),
        "c40": os.path.exists(C40_MODEL_PATH)
    }
    
    # Check which models are loaded
    loaded_models = []
    if c0_model is not None:
        loaded_models.append("c0")
    if c23_model is not None:
        loaded_models.append("c23")
    if c40_model is not None:
        loaded_models.append("c40")
    
    # Check if all required model files exist
    all_models_exist = all(model_files.values())
    
    return {
        "status": "healthy" if all_models_exist else "missing_weights", 
        "device": DEVICE,
        "mode": MODEL_TYPE,
        "available_models": model_files,
        "loaded_models": loaded_models,
        "lazy_loading": not PRELOAD_MODEL
    }

@app.post("/predict")
async def predict_image(input_data: ImageInput) -> Dict[str, Any]:
    """
    Detect if an image is a deepfake using the Faceforensics_plus_plus model(s).
    If mode is ensemble, uses all available models and averages the results.
    Otherwise, uses the specified model type.
    """
    global c0_model, c23_model, c40_model
    
    # Check if specific model type requested for this prediction
    request_model_type = input_data.model_type or MODEL_TYPE
    
    try:
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(input_data.image)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
        
        # Start timing
        start_time = time.time()
        
        # Optimize memory during inference
        if USE_GPU:
            torch.cuda.empty_cache()
        
        # Preprocess image
        try:
            image_tensor = preprocess_image(image_bytes)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")
        
        # Move tensor to appropriate device
        if USE_GPU and torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Run prediction
        results = []
        
        if request_model_type == "ensemble":
            # Run prediction with all available models
            
            # Try to load c0 model
            if c0_model is None:
                try:
                    c0_model = load_model(C0_MODEL_PATH, "c0")
                except Exception as e:
                    logger.error(f"Error loading c0 model: {str(e)}")
            
            # Try to run prediction with c0 model
            if c0_model is not None:
                try:
                    c0_result = predict_with_model(c0_model, image_tensor, "c0", input_data.threshold)
                    results.append(c0_result)
                    threading.Timer(5.0, lambda: unload_model_if_idle("c0")).start()
                except Exception as e:
                    logger.error(f"Error with c0 prediction: {str(e)}")
            
            # Try to load c23 model
            if c23_model is None:
                try:
                    c23_model = load_model(C23_MODEL_PATH, "c23")
                except Exception as e:
                    logger.error(f"Error loading c23 model: {str(e)}")
            
            # Try to run prediction with c23 model
            if c23_model is not None:
                try:
                    c23_result = predict_with_model(c23_model, image_tensor, "c23", input_data.threshold)
                    results.append(c23_result)
                    threading.Timer(5.0, lambda: unload_model_if_idle("c23")).start()
                except Exception as e:
                    logger.error(f"Error with c23 prediction: {str(e)}")
            
            # Try to load c40 model
            if c40_model is None:
                try:
                    c40_model = load_model(C40_MODEL_PATH, "c40")
                except Exception as e:
                    logger.error(f"Error loading c40 model: {str(e)}")
            
            # Try to run prediction with c40 model
            if c40_model is not None:
                try:
                    c40_result = predict_with_model(c40_model, image_tensor, "c40", input_data.threshold)
                    results.append(c40_result)
                    threading.Timer(5.0, lambda: unload_model_if_idle("c40")).start()
                except Exception as e:
                    logger.error(f"Error with c40 prediction: {str(e)}")
            
            # Calculate average probability
            if results:
                avg_probability = sum(r["probability"] for r in results) / len(results)
                prediction = 1 if avg_probability >= input_data.threshold else 0
                prediction_class = "fake" if prediction == 1 else "real"
            else:
                raise HTTPException(status_code=500, detail="All model predictions failed")
            
        else:
            # Run prediction with specific model
            if request_model_type == "c0":
                # Try to load c0 model if needed
                if c0_model is None:
                    c0_model = load_model(C0_MODEL_PATH, "c0")
                
                # Run prediction with c0 model
                result = predict_with_model(c0_model, image_tensor, "c0", input_data.threshold)
                results = [result]
                avg_probability = result["probability"]
                prediction = result["prediction"]
                prediction_class = result["class"]
                
                # Schedule unloading
                threading.Timer(5.0, lambda: unload_model_if_idle("c0")).start()
                
            elif request_model_type == "c23":
                # Try to load c23 model if needed
                if c23_model is None:
                    c23_model = load_model(C23_MODEL_PATH, "c23")
                
                # Run prediction with c23 model
                result = predict_with_model(c23_model, image_tensor, "c23", input_data.threshold)
                results = [result]
                avg_probability = result["probability"]
                prediction = result["prediction"]
                prediction_class = result["class"]
                
                # Schedule unloading
                threading.Timer(5.0, lambda: unload_model_if_idle("c23")).start()
                
            elif request_model_type == "c40":
                # Try to load c40 model if needed
                if c40_model is None:
                    c40_model = load_model(C40_MODEL_PATH, "c40")
                
                # Run prediction with c40 model
                result = predict_with_model(c40_model, image_tensor, "c40", input_data.threshold)
                results = [result]
                avg_probability = result["probability"]
                prediction = result["prediction"]
                prediction_class = result["class"]
                
                # Schedule unloading
                threading.Timer(5.0, lambda: unload_model_if_idle("c40")).start()
                
            else:
                raise HTTPException(status_code=500, detail=f"Model {request_model_type} is not supported")
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.4f} seconds")
        
        # Return standardized format without individual results
        return {
            "model": f"Faceforensics_plus_plus ({request_model_type})",
            "probability": float(avg_probability),
            "prediction": int(prediction),
            "class": prediction_class,
            "inference_time": float(inference_time)
        }
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models on startup only if PRELOAD_MODEL is true."""
    if PRELOAD_MODEL:
        logger.info("Preloading models at startup (PRELOAD_MODEL=true)")
        try:
            if MODEL_TYPE == "ensemble":
                load_all_models()
            elif MODEL_TYPE == "c0":
                load_model(C0_MODEL_PATH, "c0")
            elif MODEL_TYPE == "c23":
                load_model(C23_MODEL_PATH, "c23")
            elif MODEL_TYPE == "c40":
                load_model(C40_MODEL_PATH, "c40")
        except Exception as e:
            logger.error(f"Preloading failed: {str(e)}")
    else:
        logger.info("Models will be loaded on first request (PRELOAD_MODEL=false)")

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("MODEL_PORT", 5007))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)