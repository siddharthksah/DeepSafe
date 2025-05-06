"""
DeepSafe API Server
A robust, production-ready API for deepfake detection using multiple models running on CPU.
"""
import os
import time
import base64
import requests
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from PIL import Image
import io
import uuid

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment configuration with fallbacks
def get_environment_variable(name: str, default: str) -> str:
    """Get environment variable with fallback to default value"""
    value = os.environ.get(name)
    if not value:
        logger.warning(f"Environment variable {name} not set, using default: {default}")
        return default
    return value

# Model endpoints configuration with validation
MODEL_ENDPOINTS = {
    "cnndetection": get_environment_variable("CNNDETECTION_URL", "http://cnndetection:5008/predict"),
    "ganimagedetection": get_environment_variable("GANIMAGEDETECTION_URL", "http://ganimagedetection:5001/predict"),
    "universalfakedetect": get_environment_variable("UNIVERSALFAKEDETECT_URL", "http://universalfakedetect:5002/predict"),
    "hifi_ifdl": get_environment_variable("HIFI_IFDL_URL", "http://hifi_ifdl:5003/predict"),
    "npr_deepfakedetection": get_environment_variable("NPR_DEEPFAKEDETECTION_URL", "http://npr_deepfakedetection:5004/predict"),
    "dmimagedetection": get_environment_variable("DMIMAGEDETECTION_URL", "http://dmimagedetection:5005/predict"),
    "caddm": get_environment_variable("CADDM_URL", "http://caddm:5006/predict"),
    "faceforensics_plus_plus": get_environment_variable("FACEFORENSICS_PLUS_PLUS_URL", "http://faceforensics_plus_plus:5007/predict"),
}

# Health endpoints
HEALTH_ENDPOINTS = {model: endpoint.replace("/predict", "/health") for model, endpoint in MODEL_ENDPOINTS.items()}

# Constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = ["image/jpeg", "image/png", "image/webp"]
DEFAULT_TIMEOUT = 1200  # seconds - increased for CPU processing
MAX_RETRIES = 2

# Initialize FastAPI app with more metadata
app = FastAPI(
    title="DeepSafe API",
    description="Enterprise-grade API for deepfake detection using an ensemble of state-of-the-art models running on CPU",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Improved input models with validation
class ImageInput(BaseModel):
    """Input model for JSON payload"""
    image: str = Field(..., description="Base64 encoded image")
    models: Optional[List[str]] = Field(None, description="List of models to use (defaults to all)")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold (0.0 to 1.0)")
    ensemble_method: str = Field("voting", description="Ensemble method (voting or average)")
    
    class Config:
        arbitrary_types_allowed = True

    @validator('image')
    def validate_image(cls, v):
        """Validate if image is properly encoded in base64"""
        try:
            # Check if it's a valid base64 string
            decoded = base64.b64decode(v)
            
            # Check if it's a valid image
            Image.open(io.BytesIO(decoded))
            
            # Check size limit
            if len(decoded) > MAX_IMAGE_SIZE:
                raise ValueError(f"Image size exceeds maximum limit of {MAX_IMAGE_SIZE/1024/1024}MB")
                
            return v
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

    @validator('models')
    def validate_models(cls, v):
        """Validate selected models"""
        if v is not None:
            for model in v:
                if model not in MODEL_ENDPOINTS:
                    raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_ENDPOINTS.keys())}")
        return v
    
    @validator('ensemble_method')
    def validate_ensemble_method(cls, v):
        """Validate ensemble method"""
        if v not in ["voting", "average"]:
            raise ValueError(f"Unsupported ensemble method: {v}. Use 'voting' or 'average'.")
        return v

# Error handling middleware
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    """Global exception handler middleware"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "error_type": type(e).__name__}
        )

def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())

# Helper functions for model interaction
def check_model_health(model_name: str) -> Dict[str, Any]:
    """Check health of a specific model"""
    try:
        response = requests.get(
            HEALTH_ENDPOINTS[model_name], 
            timeout=600
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def query_model(model_name: str, image_data: str, threshold: float) -> Dict[str, Any]:
    """Query a specific model with retries"""
    logger.info(f"Querying model: {model_name}")
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                MODEL_ENDPOINTS[model_name],
                json={"image": image_data, "threshold": threshold},
                timeout=DEFAULT_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Model {model_name} returned successfully")
                return result
            else:
                error_msg = f"Model returned status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                
                # Only retry on specific status codes that might be temporary
                if response.status_code in [429, 502, 503, 504] and attempt < MAX_RETRIES:
                    retry_delay = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"Retrying {model_name} in {retry_delay}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(retry_delay)
                    continue
                    
                return {"error": error_msg}
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout querying {model_name}")
            if attempt < MAX_RETRIES:
                retry_delay = (attempt + 1) * 2
                logger.info(f"Retrying {model_name} in {retry_delay}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(retry_delay)
                continue
            return {"error": "Request timed out"}
            
        except Exception as e:
            logger.exception(f"Error querying {model_name}: {str(e)}")
            return {"error": str(e)}
    
    return {"error": "Maximum retries exceeded"}

def calculate_ensemble_verdict(results: Dict[str, Dict], threshold: float, method: str) -> Tuple[str, float, int, int]:
    """Calculate ensemble verdict using specified method"""
    # Filter to only valid results
    valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and "error" not in v}
    
    if not valid_results:
        return "undetermined", 0.0, 0, 0
    
    if method == "voting":
        # Count votes
        fake_votes = sum(1 for model, result in valid_results.items() 
                        if "prediction" in result and result["prediction"] == 1)
        
        real_votes = sum(1 for model, result in valid_results.items() 
                        if "prediction" in result and result["prediction"] == 0)
        
        total_votes = fake_votes + real_votes
        
        # Determine verdict
        if total_votes > 0:
            verdict = "fake" if fake_votes > real_votes else "real" if real_votes > fake_votes else "undetermined"
            confidence = max(fake_votes, real_votes) / total_votes if total_votes > 0 else 0.0
        else:
            verdict = "undetermined"
            confidence = 0.0
    
    elif method == "average":
        # Calculate average probability
        probabilities = [result["probability"] for model, result in valid_results.items() 
                        if "probability" in result]
        
        if probabilities:
            avg_probability = sum(probabilities) / len(probabilities)
            verdict = "fake" if avg_probability >= threshold else "real"
            # Calculate confidence as distance from threshold (normalized to 0-1)
            confidence = abs(avg_probability - 0.5) * 2
            confidence = min(1.0, max(0.0, confidence))
            
            # Set vote counts for consistency in the API response
            fake_votes = sum(1 for p in probabilities if p >= threshold)
            real_votes = len(probabilities) - fake_votes
        else:
            verdict = "undetermined"
            confidence = 0.0
            fake_votes = 0
            real_votes = 0
    
    else:
        # Should never happen due to validation
        logger.error(f"Unknown ensemble method: {method}")
        verdict = "undetermined"
        confidence = 0.0
        fake_votes = 0
        real_votes = 0
    
    return verdict, confidence, fake_votes, real_votes

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "DeepSafe API",
        "description": "Enterprise-grade API for deepfake detection using CPU-based models",
        "version": "1.0.0",
        "available_models": list(MODEL_ENDPOINTS.keys()),
        "ensemble_methods": ["voting", "average"],
        "documentation": "/docs",
        "processing_mode": "CPU-only"
    }

@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint for all models"""
    status = {}
    all_healthy = True
    
    # Check each model in sequence (not parallel)
    for model_name, endpoint in HEALTH_ENDPOINTS.items():
        model_health = check_model_health(model_name)
        status[model_name] = model_health
        
        # Consider system degraded if any model isn't healthy
        if model_health.get("status") != "healthy":
            all_healthy = False
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "models": status,
        "processing_mode": "CPU-only"
    }

@app.post("/predict", tags=["Detection"])
async def predict_image(input_data: ImageInput):
    """
    Detect if an image is a deepfake using multiple models running on CPU.
    Returns individual model results and an ensemble verdict.
    Note: CPU processing may take longer than GPU-based processing.
    """
    request_id = generate_request_id()
    logger.info(f"Processing prediction request {request_id}")
    
    # Determine which models to use
    models_to_use = input_data.models or list(MODEL_ENDPOINTS.keys())
    
    # Start timing
    start_time = time.time()
    
    # Query each model sequentially
    results = {}
    for model_name in models_to_use:
        # Skip models not in our configuration (should not happen due to validation)
        if model_name not in MODEL_ENDPOINTS:
            logger.warning(f"Request {request_id}: Skipping unknown model: {model_name}")
            continue
            
        model_result = query_model(model_name, input_data.image, input_data.threshold)
        results[model_name] = model_result
    
    # Check if we got any valid results
    valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and "error" not in v}
    
    if not valid_results:
        logger.error(f"Request {request_id}: All models failed to process the image")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="All models failed to process the image"
        )
    
    # Calculate ensemble verdict
    verdict, confidence, fake_votes, real_votes = calculate_ensemble_verdict(
        results, 
        input_data.threshold, 
        input_data.ensemble_method
    )
    
    # Calculate total inference time
    inference_time = time.time() - start_time
    
    # Prepare response
    response = {
        "request_id": request_id,
        "verdict": verdict,
        "confidence": float(confidence),
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "total_votes": fake_votes + real_votes,
        "inference_time": float(inference_time),
        "ensemble_method": input_data.ensemble_method,
        "model_results": results,
        "processing_mode": "CPU-only"
    }
    
    logger.info(f"Request {request_id} completed in {inference_time:.2f}s with verdict: {verdict}")
    return response


@app.post("/detect", tags=["Web UI"])
async def detect(
    file: UploadFile = File(...), 
    threshold: float = Form(0.5), 
    ensemble_method: str = Form("voting"),
    models: str = Form(None)
):
    """
    Form-based endpoint for file uploads from web UI.
    Provides a simplified response format for frontend integration.
    Note: CPU processing may take longer than GPU-based processing.
    """
    request_id = generate_request_id()
    logger.info(f"Processing web UI detection request {request_id}")
    
    try:
        # Validate file size
        file_size = 0
        contents = io.BytesIO()
        chunk = await file.read(1024)
        while chunk:
            file_size += len(chunk)
            if file_size > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large (max {MAX_IMAGE_SIZE/1024/1024}MB)"
                )
            contents.write(chunk)
            chunk = await file.read(1024)
        
        # Reset file pointer and read full content
        contents.seek(0)
        image_bytes = contents.read()
        
        # Validate image format
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_format = img.format
            
            # Additional image validation could be done here
            if not img_format or img_format.lower() not in ["jpeg", "jpg", "png", "webp"]:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported image format. Please upload JPEG, PNG, or WebP"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Convert to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Parse models if provided
        model_list = None
        if models:
            try:
                model_list = [m.strip() for m in models.split(',') if m.strip()]
                # Validate models
                for model in model_list:
                    if model not in MODEL_ENDPOINTS:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unknown model: {model}"
                        )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Error parsing models parameter: {str(e)}")
                model_list = None
        
        # Validate ensemble method
        if ensemble_method not in ["voting", "average"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported ensemble method: {ensemble_method}. Use 'voting' or 'average'."
            )
        
        # Create ImageInput object
        input_data = ImageInput(
            image=base64_image,
            threshold=float(threshold),
            ensemble_method=ensemble_method,
            models=model_list
        )
        
        # Use the JSON endpoint
        result = await predict_image(input_data)
        
        # Simplify response for web UI
        is_likely_deepfake = result["verdict"] == "fake"
        
        return {
            "request_id": result["request_id"],
            "is_likely_deepfake": is_likely_deepfake,
            "deepfake_probability": float(result["confidence"] if is_likely_deepfake else 1.0 - result["confidence"]),
            "model_count": result["total_votes"],
            "fake_votes": result["fake_votes"],
            "real_votes": result["real_votes"],
            "response_time": result["inference_time"],
            "processing_mode": "CPU-only"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting DeepSafe API server on port {port} in CPU-only mode")
    
    # Uvicorn configuration
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        log_level="info",
        workers=1  # Single worker to ensure sequential processing
    )