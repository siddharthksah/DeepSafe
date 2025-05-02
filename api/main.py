"""
DeepSafe Main API
Integrates all deepfake detection models and provides a unified interface
"""
import os
import time
import base64
import requests
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model endpoints configuration
MODEL_ENDPOINTS = {
    "cnndetection": os.environ.get("CNNDETECTION_URL", "http://cnndetection:5000/predict"),
    "ganimagedetection": os.environ.get("GANIMAGEDETECTION_URL", "http://ganimagedetection:5001/predict"),
    "universalfakedetect": os.environ.get("UNIVERSALFAKEDETECT_URL", "http://universalfakedetect:5002/predict")
}

# Health endpoints
HEALTH_ENDPOINTS = {model: endpoint.replace("/predict", "/health") for model, endpoint in MODEL_ENDPOINTS.items()}

# Initialize FastAPI app
app = FastAPI(
    title="DeepSafe API",
    description="Unified API for deepfake detection using multiple models",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class ImageInput(BaseModel):
    image: str  # Base64 encoded image
    models: Optional[List[str]] = None  # Optional list of models to use
    threshold: Optional[float] = 0.5  # Optional threshold for classification
    ensemble_method: Optional[str] = "voting"  # voting or average

class ModelResult(BaseModel):
    model: str
    probability: float
    prediction: int
    class_label: str
    inference_time: float

class EnsembleResult(BaseModel):
    verdict: str
    confidence: float
    fake_votes: int
    real_votes: int
    total_votes: int
    inference_time: float
    model_results: Dict[str, Any]

def validate_base64_image(base64_str: str) -> bool:
    """Validate if the string is a valid base64 encoded image."""
    try:
        # Simple validation - decode a small part to check format
        base64.b64decode(base64_str[:100])
        return True
    except Exception:
        return False

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DeepSafe API",
        "description": "Unified API for deepfake detection",
        "version": "1.0.0",
        "available_models": list(MODEL_ENDPOINTS.keys()),
        "ensemble_methods": ["voting", "average"]
    }

@app.get("/health")
async def health():
    """Health check endpoint for all models."""
    status = {}
    all_healthy = True
    
    for model_name, endpoint in HEALTH_ENDPOINTS.items():
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                model_status = response.json()
                status[model_name] = {
                    "status": model_status.get("status", "unknown"),
                    "device": model_status.get("device", "unknown")
                }
                if model_status.get("status") != "healthy":
                    all_healthy = False
            else:
                status[model_name] = {"status": "error", "message": f"Status code: {response.status_code}"}
                all_healthy = False
        except Exception as e:
            status[model_name] = {"status": "error", "message": str(e)}
            all_healthy = False
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "models": status
    }

@app.post("/predict")
async def predict_image(input_data: ImageInput):
    """
    Detect if an image is a deepfake using multiple models.
    Returns individual model results and an ensemble verdict.
    """
    # Validate input
    if not input_data.image:
        raise HTTPException(status_code=400, detail="Missing image data")
    
    if not validate_base64_image(input_data.image):
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    
    # Determine which models to use
    models_to_use = input_data.models or list(MODEL_ENDPOINTS.keys())
    for model in models_to_use:
        if model not in MODEL_ENDPOINTS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    
    # Start timing
    start_time = time.time()
    
    # Query each model
    results = {}
    for model_name in models_to_use:
        try:
            logger.info(f"Querying model: {model_name}")
            
            response = requests.post(
                MODEL_ENDPOINTS[model_name],
                json={"image": input_data.image, "threshold": input_data.threshold},
                timeout=30
            )
            
            if response.status_code == 200:
                results[model_name] = response.json()
            else:
                error_msg = f"Model returned status code {response.status_code}"
                logger.error(error_msg)
                results[model_name] = {"error": error_msg}
        except Exception as e:
            logger.error(f"Error querying {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    # Check if we got any valid results
    valid_results = {k: v for k, v in results.items() 
                    if isinstance(v, dict) and "error" not in v}
    
    if not valid_results:
        raise HTTPException(status_code=500, detail="All models failed to process the image")
    
    # Calculate ensemble verdict
    ensemble_method = input_data.ensemble_method or "voting"
    
    if ensemble_method == "voting":
        # Count votes
        fake_votes = sum(1 for model, result in valid_results.items() 
                        if "prediction" in result and result["prediction"] == 1)
        
        real_votes = sum(1 for model, result in valid_results.items() 
                        if "prediction" in result and result["prediction"] == 0)
        
        total_votes = fake_votes + real_votes
        
        # Determine verdict
        if total_votes > 0:
            verdict = "fake" if fake_votes > real_votes else "real" if real_votes > fake_votes else "undetermined"
            confidence = max(fake_votes, real_votes) / total_votes
        else:
            verdict = "undetermined"
            confidence = 0.0
    
    elif ensemble_method == "average":
        # Calculate average probability
        probabilities = [result["probability"] for model, result in valid_results.items() 
                        if "probability" in result]
        
        if probabilities:
            avg_probability = sum(probabilities) / len(probabilities)
            verdict = "fake" if avg_probability >= input_data.threshold else "real"
            # Calculate how far from threshold (normalized to 0-1)
            confidence = abs(avg_probability - input_data.threshold) + 0.5
            # Constrain to 0-1 range
            confidence = min(1.0, max(0.0, confidence))
            
            # Set vote counts for consistency
            fake_votes = sum(1 for p in probabilities if p >= input_data.threshold)
            real_votes = len(probabilities) - fake_votes
            total_votes = len(probabilities)
        else:
            verdict = "undetermined"
            confidence = 0.0
            fake_votes = 0
            real_votes = 0
            total_votes = 0
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown ensemble method: {ensemble_method}")
    
    # Calculate total inference time
    inference_time = time.time() - start_time
    
    # Prepare response
    ensemble_result = {
        "verdict": verdict,
        "confidence": float(confidence),
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "total_votes": total_votes,
        "inference_time": float(inference_time),
        "ensemble_method": ensemble_method,
        "model_results": results
    }
    
    return ensemble_result

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)