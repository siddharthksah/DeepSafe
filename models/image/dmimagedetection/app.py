"""
DMimageDetection Model Service
This service loads the DMimageDetection model and exposes an API endpoint to analyze images
"""
import os
import io
import sys
import base64
import torch
import logging
import time
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from typing import Dict, Any, Optional
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add model path to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'dmimagedetection'))

# Initialize FastAPI app
app = FastAPI(
    title="DMimageDetection Model Service",
    description="Service for detecting deepfake images using DMimageDetection models",
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
WEIGHTS_DIR = os.path.join(current_dir, "dmimagedetection/weights")
USE_GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU", False)
DEVICE = 'cuda:0' if USE_GPU else 'cpu'

# Define request model
class ImageInput(BaseModel):
    image: str  # Base64 encoded image
    threshold: Optional[float] = 0.5  # Optional threshold for classification

# Global variable for the models
models = {}

def load_model(model_name):
    """Load the specified DMimageDetection model."""
    if model_name in models:
        return models[model_name]
    
    try:
        logger.info(f"Loading model {model_name} on {DEVICE}")
        
        # Import model definition
        from dmimagedetection.test_code.networks.resnet_mod import resnet50
        
        # Initialize model
        model = resnet50(num_classes=1, stride0=1, gap_size=1)
        
        # Load weights
        weight_path = os.path.join(WEIGHTS_DIR, model_name, "model_epoch_best.pth")
        
        if not os.path.exists(weight_path):
            logger.error(f"Model weights not found at: {weight_path}")
            raise FileNotFoundError(f"Model weights not found at: {weight_path}")
            
        state_dict = torch.load(weight_path, map_location=DEVICE)
        
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'])
        else:
            model.load_state_dict(state_dict)
        
        model = model.to(DEVICE)
        model.eval()
        
        # Save to global models dict
        models[model_name] = model
        
        logger.info(f"Model {model_name} loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise

def get_transform():
    """Get image transform pipeline for models."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "model": "DMimageDetection",
        "description": "Models for detecting deepfake images (diffusion models and GANs)",
        "available_models": ["Grag2021_progan", "Grag2021_latent"],
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Only check if files exist to report as loading
        progan_path = os.path.join(WEIGHTS_DIR, "Grag2021_progan", "model_epoch_best.pth")
        latent_path = os.path.join(WEIGHTS_DIR, "Grag2021_latent", "model_epoch_best.pth")
        
        if os.path.exists(progan_path) and os.path.exists(latent_path):
            # Try to load one model to verify
            _ = load_model("Grag2021_latent")
            return {"status": "healthy", "device": DEVICE}
        else:
            missing = []
            if not os.path.exists(progan_path):
                missing.append("Grag2021_progan")
            if not os.path.exists(latent_path):
                missing.append("Grag2021_latent")
            return {"status": "loading", "message": f"Model weights missing: {missing}", "device": DEVICE}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e), "device": DEVICE}

@app.post("/predict")
async def predict_image(input_data: ImageInput) -> Dict[str, Any]:
    """
    Detect if an image is a deepfake using the DMimageDetection models.
    Returns standardized output matching other models.
    """
    
    try:
        # We'll use latent model by default - better for diffusion model detection
        model_name = "Grag2021_latent"
            
        # Make sure model is loaded
        model = load_model(model_name)
            
        # Decode base64 image
        image_bytes = base64.b64decode(input_data.image)
        
        # Start timing
        start_time = time.time()
        
        # Open the image using PIL
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply the transformation pipeline
        transform = get_transform()
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        
        # Run prediction
        with torch.no_grad():
            output = model(image_tensor)
            
            # Get the probability
            logit = output.cpu().numpy()
            
            # For multi-dimensional output, take mean over spatial dimensions
            while len(logit.shape) > 1:
                logit = np.mean(logit, axis=-1)
            
            # Convert logit to probability using sigmoid
            probability = float(1.0 / (1.0 + np.exp(-logit)))
        
        # Apply threshold
        prediction = 1 if probability >= input_data.threshold else 0
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.4f} seconds")
        
        # Return standardized format
        return {
            "model": "dmimagedetection",
            "probability": float(probability),
            "prediction": int(prediction),
            "class": "fake" if prediction == 1 else "real",
            "inference_time_seconds": float(inference_time)
        }
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("MODEL_PORT", 5005))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)