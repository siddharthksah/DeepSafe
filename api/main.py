#!/usr/bin/env python3
"""
DeepSafe API Gateway
====================

The central orchestration layer for the DeepSafe platform. This service acts as a unified gateway
that routes detection requests to appropriate microservices based on media type (Image, Video, Audio).

Architectural Overview:
- **Orchestration**: Manages the lifecycle of a detection request, from validation to result aggregation.
- **Ensemble Logic**: Implements the decision fusion layer, supporting Voting, Averaging, and Stacking strategies.
- **Meta-Learning**: Dynamically loads and applies modality-specific stacking models (meta-learners) to improve prediction accuracy.
- **Microservice Communication**: Dispatches parallel requests to isolated model containers, ensuring fault isolation and scalability.

Configuration:
Driven by `deepsafe_config.json`, allowing for dynamic registration of new model endpoints without code changes.
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

# Updated Pydantic imports for V2
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
import uvicorn
from PIL import Image, UnidentifiedImageError
import io
import uuid
import joblib
import numpy as np
import pandas as pd
import json

import sys

from rich.console import Console as RichConsole
from rich.table import Table as RichTable
from rich.text import Text as RichText
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends, status
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import init_db, get_db, AnalysisHistory

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
rich_console = RichConsole(width=120)

# --- Constants for Payload Keys and Media Handling ---
MEDIA_TYPE_PAYLOAD_KEYS: Dict[str, str] = {
    "image": "image_data",
    "video": "video_data",
    "audio": "audio_data",
}

MAX_IMAGE_SIZE_MB: int = 100
MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_GENERAL_PAYLOAD_SIZE_BYTES: int = (MAX_IMAGE_SIZE_MB + 15) * 1024 * 1024

CONTENT_TYPE_TO_MEDIA_TYPE_MAP: Dict[str, str] = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "video/mp4": "video",
    "video/x-m4v": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/x-matroska": "video",
    "audio/wav": "audio",
    "audio/mpeg": "audio",
    "audio/flac": "audio",
    "audio/ogg": "audio",
    "audio/x-m4a": "audio",
}


# --- Environment Variable Handling ---
def get_environment_variable(
    name: str, default: Optional[str] = None, required: bool = False
) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        if required:
            logger.error(f"FATAL: Required environment variable {name} not set.")
            raise ValueError(f"Missing required environment variable: {name}")
        if default is not None:
            logger.warning(
                f"Environment variable '{name}' not set, using default: '{default[:50]}...'"
            )
            return default
        return None
    return value


# --- Configuration Loading ---
ALL_MODEL_CONFIGS: Dict[str, Any] = {}
SUPPORTED_MEDIA_TYPES: List[str] = []

CONFIG_FILE_PATH_FROM_ENV = get_environment_variable("DEEPSAFE_CONFIG_FILE_PATH")

if CONFIG_FILE_PATH_FROM_ENV and os.path.exists(CONFIG_FILE_PATH_FROM_ENV):
    logger.info(f"Loading configuration from: {CONFIG_FILE_PATH_FROM_ENV}")
    try:
        with open(CONFIG_FILE_PATH_FROM_ENV, "r") as f_config:
            loaded_json = json.load(f_config)
        ALL_MODEL_CONFIGS = loaded_json
        SUPPORTED_MEDIA_TYPES = list(ALL_MODEL_CONFIGS.get("media_types", {}).keys())
        if not SUPPORTED_MEDIA_TYPES:
            logger.error(
                f"FATAL: 'media_types' key missing or empty in {CONFIG_FILE_PATH_FROM_ENV}."
            )
            ALL_MODEL_CONFIGS = {"media_types": {}}
        else:
            logger.info(f"Active media types: {SUPPORTED_MEDIA_TYPES}")
    except json.JSONDecodeError as e:
        logger.error(f"FATAL: Malformed JSON in config file: {e}.")
        ALL_MODEL_CONFIGS = {"media_types": {}}
    except Exception as e:
        logger.error(f"FATAL: Config load failure: {e}.")
        ALL_MODEL_CONFIGS = {"media_types": {}}
else:
    logger.error(
        f"FATAL: Configuration file not found at '{CONFIG_FILE_PATH_FROM_ENV}'. Service cannot start."
    )
    ALL_MODEL_CONFIGS = {"media_types": {}}

DEFAULT_TIMEOUT: int = int(ALL_MODEL_CONFIGS.get("default_api_timeout_seconds", 1200))
MAX_RETRIES: int = int(ALL_MODEL_CONFIGS.get("default_max_retries", 1))
META_MODEL_ARTIFACTS_DIR: str = get_environment_variable(
    "META_MODEL_ARTIFACTS_DIR", "/app/meta_model_artifacts"
)

# --- Global Variables for Meta-Learners ---
meta_learners: Dict[str, Any] = {}
meta_scalers: Dict[str, Any] = {}
meta_imputers: Dict[str, Any] = {}
meta_feature_columns_map: Dict[str, List[str]] = {}

# --- Auth Configuration ---
SECRET_KEY = get_environment_variable(
    "SECRET_KEY", "deepsafe_super_secret_key_change_me"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock Database for Demo Purposes
fake_users_db = {}


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user


# --- FastAPI Application ---
app = FastAPI(
    title="DeepSafe API",
    description="Enterprise-grade API for deepfake detection using an ensemble of state-of-the-art models.",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.on_event("startup")
async def startup_event_api():
    """
    Initializes the API service.

    Critical tasks:
    1. Validates supported media types.
    2. Hydrates the meta-learner registry by loading serialized artifacts (models, scalers, imputers)
       from the shared volume. This enables the 'stacking' ensemble method.
    """
    # Initialize database
    init_db()
    logger.info("Database initialized successfully")

    global meta_learners, meta_scalers, meta_imputers, meta_feature_columns_map

    if not SUPPORTED_MEDIA_TYPES:
        logger.error(
            "Initialization Warning: No media types configured. Meta-learners will not be loaded."
        )
        return

    logger.info(f"Initializing meta-learner artifacts for: {SUPPORTED_MEDIA_TYPES}")
    for m_type in SUPPORTED_MEDIA_TYPES:
        media_type_artifact_subdir = os.path.join(META_MODEL_ARTIFACTS_DIR, m_type)
        try:
            # Artifacts are expected to follow a standard naming convention within their media-type directories.
            model_path = os.path.join(
                media_type_artifact_subdir, "deepsafe_meta_learner.joblib"
            )
            scaler_path = os.path.join(
                media_type_artifact_subdir, "deepsafe_meta_scaler.joblib"
            )
            imputer_path = os.path.join(
                media_type_artifact_subdir, "deepsafe_meta_imputer.joblib"
            )
            cols_path = os.path.join(
                media_type_artifact_subdir, "deepsafe_meta_feature_columns.json"
            )

            required_artifact_paths = [model_path, scaler_path, imputer_path, cols_path]

            # Pre-flight check for artifact existence to avoid partial loading states.
            if not os.path.isdir(media_type_artifact_subdir):
                logger.warning(
                    f"Artifact subdirectory '{media_type_artifact_subdir}' not found for media type '{m_type}'. Stacking will be unavailable for it."
                )
                (
                    meta_learners[m_type],
                    meta_scalers[m_type],
                    meta_imputers[m_type],
                    meta_feature_columns_map[m_type],
                ) = (None, None, None, None)
                continue  # Move to next media type

            missing_artifacts = [
                p for p in required_artifact_paths if not os.path.exists(p)
            ]

            if not missing_artifacts:
                meta_learners[m_type] = joblib.load(model_path)
                meta_scalers[m_type] = joblib.load(scaler_path)
                meta_imputers[m_type] = joblib.load(imputer_path)
                with open(cols_path, "r") as f:
                    meta_feature_columns_map[m_type] = json.load(f)
                logger.info(
                    f"Stacking meta-learner and preprocessors for '{m_type}' loaded successfully from '{media_type_artifact_subdir}'."
                )
            else:
                logger.warning(
                    f"Meta-learner artifacts not fully found in '{media_type_artifact_subdir}' for media type '{m_type}'. Missing: {missing_artifacts}. Stacking will be unavailable for it."
                )
                (
                    meta_learners[m_type],
                    meta_scalers[m_type],
                    meta_imputers[m_type],
                    meta_feature_columns_map[m_type],
                ) = (None, None, None, None)
        except Exception as e:
            logger.error(
                f"Error loading meta-learner artifacts for '{m_type}' from '{media_type_artifact_subdir}': {e}",
                exc_info=True,
            )
            (
                meta_learners[m_type],
                meta_scalers[m_type],
                meta_imputers[m_type],
                meta_feature_columns_map[m_type],
            ) = (None, None, None, None)

    loaded_summary = {
        mt: (learner is not None) for mt, learner in meta_learners.items()
    }
    logger.info(f"Meta-learner loading summary (True if loaded): {loaded_summary}")


# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_size_limit_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    if request.method == "POST" and request.url.path == "/predict":
        content_length_str = request.headers.get("content-length")
        if content_length_str:
            try:
                content_length = int(content_length_str)
                if content_length > MAX_GENERAL_PAYLOAD_SIZE_BYTES:
                    logger.warning(
                        f"Request {request.state.request_id}: Payload size {content_length} exceeds limit {MAX_GENERAL_PAYLOAD_SIZE_BYTES} for /predict."
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request payload too large. Max size: {MAX_GENERAL_PAYLOAD_SIZE_BYTES / (1024*1024):.1f}MB",
                            "request_id": request.state.request_id,
                        },
                    )
            except ValueError:
                logger.warning(
                    f"Request {request.state.request_id}: Invalid Content-Length header: {content_length_str}"
                )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    if not hasattr(request.state, "request_id"):
        request.state.request_id = str(uuid.uuid4())
    try:
        return await call_next(request)
    except HTTPException as e:
        logger.warning(
            f"Request {request.state.request_id}: HTTPException raised: Status {e.status_code}, Detail: {e.detail}"
        )
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail, "request_id": request.state.request_id},
        )
    except Exception as e:
        logger.exception(
            f"Request {request.state.request_id}: Unhandled internal exception: {str(e)}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred.",
                "request_id": request.state.request_id,
                "error_type": type(e).__name__,
            },
        )


# --- Pydantic Models ---
class PredictInput(BaseModel):
    media_type: str = Field(
        ...,
        description=f"Type of media. Must be one of: {SUPPORTED_MEDIA_TYPES if SUPPORTED_MEDIA_TYPES else ['image', 'video', 'audio']}",
    )
    image_data: Optional[str] = Field(
        None, description="Base64 encoded image data (if media_type is 'image')"
    )
    video_data: Optional[str] = Field(
        None, description="Base64 encoded video data (if media_type is 'video')"
    )
    audio_data: Optional[str] = Field(
        None, description="Base64 encoded audio data (if media_type is 'audio')"
    )

    models: Optional[List[str]] = Field(
        None, description="List of specific models to use for this media_type."
    )
    threshold: float = Field(
        default_factory=lambda: (
            ALL_MODEL_CONFIGS.get("default_threshold", 0.5)
            if ALL_MODEL_CONFIGS
            else 0.5
        ),
        ge=0.0,
        le=1.0,
    )
    ensemble_method: str = Field(
        default_factory=lambda: (
            ALL_MODEL_CONFIGS.get("default_ensemble_method", "stacking")
            if ALL_MODEL_CONFIGS
            else "stacking"
        ),
        pattern="^(voting|average|stacking)$",
    )

    @model_validator(mode="after")
    def check_media_data_consistency(self) -> "PredictInput":
        media_type = self.media_type

        expected_payload_key = MEDIA_TYPE_PAYLOAD_KEYS.get(media_type)
        if not expected_payload_key:
            raise ValueError(
                f"Internal error: No payload key defined for media_type '{media_type}'."
            )

        if not getattr(self, expected_payload_key, None):
            raise ValueError(
                f"Field '{expected_payload_key}' is required and must not be empty when media_type is '{media_type}'."
            )

        for current_mt, data_key in MEDIA_TYPE_PAYLOAD_KEYS.items():
            if current_mt != media_type and getattr(self, data_key, None) is not None:
                raise ValueError(
                    f"If media_type is '{media_type}', field '{data_key}' (for {current_mt}) must be null or absent."
                )
        return self

    @field_validator("media_type")
    @classmethod
    def validate_media_type_is_supported(cls, v_media_type: str) -> str:
        if not SUPPORTED_MEDIA_TYPES:
            logger.warning(
                "SUPPORTED_MEDIA_TYPES is empty due to config load issue, cannot validate media_type against it. Allowing type through."
            )
            return v_media_type
        if v_media_type not in SUPPORTED_MEDIA_TYPES:
            raise ValueError(
                f"Unsupported media_type: '{v_media_type}'. Supported types are: {SUPPORTED_MEDIA_TYPES}"
            )
        return v_media_type

    @field_validator("models")
    @classmethod
    def validate_models_are_configured_for_media_type(
        cls, v_models: Optional[List[str]], info: ValidationInfo
    ) -> Optional[List[str]]:
        if "media_type" not in info.data:
            return v_models

        media_type = info.data.get("media_type")

        if v_models is not None and media_type:
            if not v_models:
                return None

            media_type_config = (
                ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
            ).get(media_type, {})
            available_models_for_type = list(
                media_type_config.get("model_endpoints", {}).keys()
            )

            for model_name in v_models:
                if model_name not in available_models_for_type:
                    raise ValueError(
                        f"Unknown model '{model_name}' specified for media_type '{media_type}'. Available models for '{media_type}': {available_models_for_type}"
                    )
        return v_models

    @field_validator("image_data", "video_data", "audio_data", mode="before")
    @classmethod
    def validate_encoded_media_data_size(
        cls, v_media_data: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        field_name = info.field_name
        if v_media_data is not None:
            if not isinstance(v_media_data, str) or not v_media_data.strip():
                raise ValueError(
                    f"Field '{field_name}' must be a non-empty base64 string if provided."
                )
            try:
                approx_original_data_len = (len(v_media_data) * 3) / 4
                if approx_original_data_len > MAX_GENERAL_PAYLOAD_SIZE_BYTES:
                    raise ValueError(
                        f"Encoded {field_name} data (approx {approx_original_data_len / (1024*1024):.1f}MB) exceeds general API payload size limit ({MAX_GENERAL_PAYLOAD_SIZE_BYTES / (1024*1024):.1f}MB)."
                    )
            except TypeError:
                raise ValueError(f"Field '{field_name}' must be a string if provided.")
        return v_media_data


# --- Helper Functions for Model Interaction and Ensembling ---
def check_model_health_api(model_name: str, media_type: str) -> Dict[str, Any]:
    media_type_config = ALL_MODEL_CONFIGS.get("media_types", {}).get(media_type, {})
    health_endpoints_for_type = media_type_config.get("health_endpoints", {})
    if model_name not in health_endpoints_for_type:
        return {
            "status": "error",
            "message": f"No health endpoint configured for model '{model_name}' of type '{media_type}'.",
        }

    health_url = health_endpoints_for_type[model_name]
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(
            f"Health check failed for {model_name} ({media_type}) at {health_url}: {e}"
        )
        return {"status": "unreachable", "message": str(e)}
    except json.JSONDecodeError:
        logger.warning(
            f"Health check for {model_name} ({media_type}) returned non-JSON: {response.text[:100]}"
        )
        return {"status": "invalid_response", "message": "Non-JSON health response"}


def query_model_api(
    model_name: str,
    media_type: str,
    encoded_media_data: str,
    threshold: float,
    request_id: str,
) -> Dict[str, Any]:
    media_type_config = ALL_MODEL_CONFIGS.get("media_types", {}).get(media_type, {})
    model_endpoints_for_type = media_type_config.get("model_endpoints", {})

    if model_name not in model_endpoints_for_type:
        logger.error(
            f"Request {request_id}: Model '{model_name}' not configured for media type '{media_type}'."
        )
        return {
            "error": f"Model '{model_name}' not configured for media type '{media_type}'."
        }

    model_predict_url = model_endpoints_for_type[model_name]
    logger.info(
        f"Request {request_id}: Querying model '{model_name}' ({media_type}) at {model_predict_url}."
    )

    payload_key = MEDIA_TYPE_PAYLOAD_KEYS.get(media_type)
    if not payload_key:
        logger.error(
            f"Request {request_id}: No payload key defined for media type '{media_type}' for model '{model_name}'."
        )
        return {
            "error": f"Internal configuration error: Payload key not defined for media type '{media_type}'."
        }

    payload = {payload_key: encoded_media_data, "threshold": threshold}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                model_predict_url, json=payload, timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                f"Request {request_id}: Model '{model_name}' ({media_type}) responded successfully (attempt {attempt+1})."
            )
            return result
        except requests.exceptions.Timeout:
            logger.warning(
                f"Request {request_id}: Timeout querying '{model_name}' ({media_type}) (attempt {attempt+1}/{MAX_RETRIES+1})."
            )
            if attempt == MAX_RETRIES:
                return {
                    "error": f"Request to '{model_name}' timed out after {MAX_RETRIES+1} attempts."
                }
        except requests.exceptions.HTTPError as e:
            error_text = e.response.text[:200] if e.response else "No response text."
            logger.error(
                f"Request {request_id}: HTTPError from '{model_name}' ({media_type}): {e.response.status_code} - {error_text} (attempt {attempt+1})."
            )
            if attempt < MAX_RETRIES and e.response.status_code in [429, 502, 503, 504]:
                pass
            else:
                return {
                    "error": f"Model '{model_name}' returned HTTP {e.response.status_code}: {error_text}"
                }
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Request {request_id}: Network error querying '{model_name}' ({media_type}): {str(e)} (attempt {attempt+1})."
            )
            if attempt == MAX_RETRIES:
                return {
                    "error": f"Network error querying model '{model_name}': {str(e)}"
                }
        except json.JSONDecodeError:
            logger.error(
                f"Request {request_id}: Model '{model_name}' ({media_type}) returned non-JSON response: {response.text[:100]} (attempt {attempt+1})."
            )
            if attempt == MAX_RETRIES:
                return {"error": f"Invalid JSON response from model '{model_name}'."}

        if attempt < MAX_RETRIES:
            retry_delay = 2**attempt
            logger.info(
                f"Request {request_id}: Retrying '{model_name}' ({media_type}) in {retry_delay}s..."
            )
            time.sleep(retry_delay)

    return {
        "error": f"Maximum retries ({MAX_RETRIES+1}) exceeded for model '{model_name}' ({media_type})."
    }


def calculate_ensemble_verdict_api(
    results: Dict[str, Dict],
    threshold: float,
    method: str,
    media_type: str,
    request_id: str,
) -> Tuple[str, float, int, int, float, str]:
    valid_results = {
        k: v
        for k, v in results.items()
        if isinstance(v, dict)
        and "error" not in v
        and v.get("probability") is not None
        and v.get("prediction") is not None
    }
    base_fake_votes = sum(
        1 for r_data in valid_results.values() if r_data.get("prediction") == 1
    )
    base_real_votes = sum(
        1 for r_data in valid_results.values() if r_data.get("prediction") == 0
    )
    total_valid_models = len(valid_results)

    if total_valid_models == 0:
        logger.warning(
            f"Request {request_id} ({media_type}): No valid base model results for ensemble calculation."
        )
        return "undetermined", 0.0, 0, 0, 0.5, method

    actual_method_used = method
    ensemble_prob_fake_score: float = 0.5

    if method == "stacking":
        learner = meta_learners.get(media_type)
        scaler = meta_scalers.get(media_type)
        imputer = meta_imputers.get(media_type)
        feature_cols = meta_feature_columns_map.get(media_type)

        if learner and scaler and imputer and feature_cols:
            feature_vector_values = []
            for model_col_name_from_training in feature_cols:
                base_model_name = model_col_name_from_training.replace("_prob", "")
                model_res = valid_results.get(base_model_name)
                if model_res and model_res.get("probability") is not None:
                    feature_vector_values.append(model_res["probability"])
                else:
                    feature_vector_values.append(np.nan)

            feature_df = pd.DataFrame([feature_vector_values], columns=feature_cols)

            imputed_features = imputer.transform(feature_df)
            scaled_features = scaler.transform(imputed_features)

            ensemble_prob_fake_score = float(
                learner.predict_proba(scaled_features)[0, 1]
            )
        else:
            logger.warning(
                f"Request {request_id}: Stacking ensemble for '{media_type}' requested, but artifacts not loaded. Falling back to 'voting'."
            )
            actual_method_used = "voting"

    if actual_method_used == "voting":
        if total_valid_models > 0:
            ensemble_prob_fake_score = float(base_fake_votes / total_valid_models)
        else:
            ensemble_prob_fake_score = 0.5
    elif actual_method_used == "average":
        probabilities = [
            r_data["probability"]
            for r_data in valid_results.values()
            if r_data.get("probability") is not None
        ]
        if probabilities:
            ensemble_prob_fake_score = float(sum(probabilities) / len(probabilities))
        else:
            ensemble_prob_fake_score = 0.5

    verdict = "fake" if ensemble_prob_fake_score >= threshold else "real"
    confidence_in_verdict = float(
        ensemble_prob_fake_score
        if verdict == "fake"
        else (1.0 - ensemble_prob_fake_score)
    )

    logger.info(
        f"Request {request_id} ({media_type}): Ensemble method '{actual_method_used}' "
        f"-> P(Fake)={ensemble_prob_fake_score:.4f}, Verdict='{verdict}'"
    )
    return (
        verdict,
        confidence_in_verdict,
        base_fake_votes,
        base_real_votes,
        ensemble_prob_fake_score,
        actual_method_used,
    )


# --- API Endpoints ---
@app.get("/", tags=["Info"])
async def root_api():
    media_types_config = (
        ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
    )
    configured_media_types = list(media_types_config.keys())
    model_summary = {
        mt: list(media_types_config.get(mt, {}).get("model_endpoints", {}).keys())
        for mt in configured_media_types
    }
    stacking_status = {
        mt: (meta_learners.get(mt) is not None) for mt in configured_media_types
    }

    return {
        "name": "DeepSafe API",
        "version": app.version,
        "message": "Welcome to the DeepSafe multimodal deepfake detection API.",
        "configured_media_types": configured_media_types,
        "model_endpoints_summary": model_summary,
        "stacking_ensemble_loaded_status": stacking_status,
        "documentation_urls": {"swagger_ui": "/docs", "redoc": "/redoc"},
    }


@app.get("/health", tags=["System"])
async def health_check_api_endpoint(request: Request):
    req_id = request.state.request_id
    logger.info(f"Request {req_id}: Received main API health check.")

    system_health_report: Dict[str, Any] = {
        "overall_api_status": "healthy",
        "media_type_details": {},
    }
    overall_system_is_healthy = True

    media_types_in_config = (
        list(ALL_MODEL_CONFIGS.get("media_types", {}).keys())
        if ALL_MODEL_CONFIGS
        else []
    )

    for m_type in media_types_in_config:
        type_specific_status: Dict[str, Any] = {"status": "healthy", "models": {}}
        all_models_for_type_healthy = True

        current_media_type_config = (
            ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
        ).get(m_type, {})
        model_endpoints_for_this_type = current_media_type_config.get(
            "model_endpoints", {}
        )

        if not model_endpoints_for_this_type:
            type_specific_status["status"] = "no_models_configured"
        else:
            for model_name in model_endpoints_for_this_type.keys():
                model_health_info = check_model_health_api(model_name, m_type)
                type_specific_status["models"][model_name] = model_health_info
                if model_health_info.get("status") != "healthy":
                    all_models_for_type_healthy = False

            if not all_models_for_type_healthy:
                type_specific_status["status"] = "degraded_models"
                overall_system_is_healthy = False

        type_specific_status["stacking_ensemble_loaded"] = (
            meta_learners.get(m_type) is not None
        )
        default_ensemble = (
            ALL_MODEL_CONFIGS.get("default_ensemble_method", "stacking")
            if ALL_MODEL_CONFIGS
            else "stacking"
        )
        if (
            not type_specific_status["stacking_ensemble_loaded"]
            and default_ensemble == "stacking"
        ):
            if type_specific_status["status"] == "healthy":
                type_specific_status["status"] = "degraded_stacking_unavailable"
            overall_system_is_healthy = False
            logger.warning(
                f"Request {req_id}: Stacking (default method) for '{m_type}' is unavailable. System component for this media type is degraded."
            )

        system_health_report["media_type_details"][m_type] = type_specific_status

    if not overall_system_is_healthy:
        system_health_report["overall_api_status"] = "degraded"

    system_health_report["request_id"] = req_id
    system_health_report["processing_mode"] = "CPU-only"
    return system_health_report


def print_results_summary_table_api(
    request_id: str,
    media_type: str,
    ensemble_method_used: str,
    ensemble_verdict: str,
    ensemble_prob_fake_score: float,
    model_query_results: Dict[str, Dict],
    threshold_used: float,
):
    table = RichTable(
        title=f"DeepSafe API Analysis (Req ID: {request_id}, Media: {media_type.upper()})",
        show_lines=True,
    )
    table.add_column("Component", style="cyan", min_width=25, overflow="fold")
    table.add_column("P(Fake)", style="magenta", justify="right")
    table.add_column("Pred Class", style="blue", justify="center")
    table.add_column("Verdict", style="green", justify="center")
    table.add_column(
        "Time (s) / Details", style="yellow", min_width=15, overflow="fold"
    )

    ens_pred_binary = 1 if ensemble_prob_fake_score >= threshold_used else 0
    ens_verdict_str = "fake" if ens_pred_binary == 1 else "real"
    ens_style = "bold red" if ens_verdict_str == "fake" else "bold green"

    table.add_row(
        f"Ensemble ({ensemble_method_used.capitalize()})",
        f"{ensemble_prob_fake_score:.4f}",
        str(ens_pred_binary),
        RichText(ens_verdict_str.upper(), style=ens_style),
        f"Thresh: {threshold_used:.2f}",
    )
    table.add_section()

    for model_name, res in sorted(model_query_results.items()):
        if isinstance(res, dict) and "error" not in res:
            prob = res.get("probability")
            pred_b = res.get("prediction")
            pred_c_str = res.get("class", "N/A")
            inf_t = res.get("inference_time", res.get("total_request_time"))
            prob_txt = (
                f"{prob:.4f}" if isinstance(prob, (float, np.floating)) else "N/A"
            )
            pred_b_txt = str(pred_b) if pred_b is not None else "N/A"
            time_txt = (
                f"{inf_t:.2f}s" if isinstance(inf_t, (float, np.floating)) else "N/A"
            )
            m_style = (
                "red"
                if pred_c_str == "fake"
                else "green" if pred_c_str == "real" else "default"
            )
            table.add_row(
                model_name,
                prob_txt,
                pred_b_txt,
                RichText(pred_c_str.upper(), style=m_style),
                time_txt,
            )
        elif isinstance(res, dict) and "error" in res:
            table.add_row(
                model_name,
                RichText("ERROR", style="bold red"),
                "-",
                "-",
                RichText(str(res.get("error", "?"))[:50] + "...", style="dim red"),
            )
        else:
            table.add_row(model_name, "N/A", "N/A", "N/A", "Invalid result")
    rich_console.print(table)


@app.post("/predict", tags=["Detection"], response_model_exclude_none=True)
async def predict_media_endpoint_api(request: Request, input_data: PredictInput):
    req_id = request.state.request_id
    media_type = input_data.media_type

    logger.info(
        f"Request {req_id}: Prediction received for media_type='{media_type}'. Ensemble='{input_data.ensemble_method}', Threshold='{input_data.threshold}', Models='{input_data.models or 'all configured'}'."
    )

    media_type_config = (
        ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
    ).get(media_type, {})
    model_endpoints_for_type = media_type_config.get("model_endpoints", {})

    models_to_use_names = (
        input_data.models
        if (input_data.models and len(input_data.models) > 0)
        else list(model_endpoints_for_type.keys())
    )
    if not models_to_use_names:
        logger.error(
            f"Request {req_id}: No models available or specified for media type '{media_type}'. Cannot proceed."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No models available or specified for media type '{media_type}'.",
        )

    payload_key_for_media = MEDIA_TYPE_PAYLOAD_KEYS.get(media_type)
    if not payload_key_for_media:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal configuration error: media payload key mapping missing.",
        )
    encoded_media_content = getattr(input_data, payload_key_for_media, None)
    if not encoded_media_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing media data in field '{payload_key_for_media}' for media_type '{media_type}'.",
        )

    start_overall_time = time.time()
    model_query_results: Dict[str, Dict] = {}
    for model_name in models_to_use_names:
        if model_name not in model_endpoints_for_type:
            logger.warning(
                f"Request {req_id}: Model '{model_name}' was requested but is not configured for media_type '{media_type}'. Skipping."
            )
            model_query_results[model_name] = {
                "error": f"Model '{model_name}' not configured for media_type '{media_type}'."
            }
            continue
        model_query_results[model_name] = query_model_api(
            model_name, media_type, encoded_media_content, input_data.threshold, req_id
        )

    if not any("error" not in r_data for r_data in model_query_results.values()):
        logger.error(
            f"Request {req_id}: All base model queries failed for media_type '{media_type}'."
        )
        print_results_summary_table_api(
            req_id,
            media_type,
            input_data.ensemble_method,
            "undetermined",
            0.5,
            model_query_results,
            input_data.threshold,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"All base models for '{media_type}' failed to process the request.",
        )

    (
        verdict,
        confidence,
        fake_votes,
        real_votes,
        ensemble_prob_fake,
        actual_method_used,
    ) = calculate_ensemble_verdict_api(
        model_query_results,
        input_data.threshold,
        input_data.ensemble_method,
        media_type,
        req_id,
    )
    total_processing_time = time.time() - start_overall_time
    response_payload = {
        "request_id": req_id,
        "media_type_processed": media_type,
        "verdict": verdict,
        "confidence_in_verdict": confidence,
        "ensemble_score_is_fake": ensemble_prob_fake,
        "base_model_fake_votes": fake_votes,
        "base_model_real_votes": real_votes,
        "base_model_total_votes": fake_votes + real_votes,
        "total_inference_time_seconds": total_processing_time,
        "ensemble_method_requested": input_data.ensemble_method,
        "ensemble_method_used": actual_method_used,
        "model_results": model_query_results,
        "processing_mode": "CPU-only",
    }

    # Save to database history
    try:
        db = SessionLocal()
        history_record = AnalysisHistory(
            request_id=req_id,
            username=None,  # TODO: Add current user if using auth on this endpoint
            media_type=media_type,
            media_name=None,  # Not available in this endpoint
            verdict=verdict,
            confidence=confidence,
            ensemble_method=actual_method_used,
            ensemble_score=ensemble_prob_fake,
            inference_time=total_processing_time,
            full_response=json.dumps(response_payload),
        )
        db.add(history_record)
        db.commit()
        db.close()
    except Exception as db_err:
        logger.warning(f"Request {req_id}: Failed to save to database: {db_err}")

    logger.info(
        f"Request {req_id} ({media_type}): Prediction complete in {total_processing_time:.2f}s. Verdict: '{verdict}', P(Fake): {ensemble_prob_fake:.4f} (Method: '{actual_method_used}')"
    )
    print_results_summary_table_api(
        req_id,
        media_type,
        actual_method_used,
        verdict,
        ensemble_prob_fake,
        model_query_results,
        input_data.threshold,
    )
    return response_payload

    print_results_summary_table_api(
        req_id,
        media_type,
        actual_method_used,
        verdict,
        ensemble_prob_fake,
        model_query_results,
        input_data.threshold,
    )
    return response_payload


# --- Auth Endpoints ---


@app.post("/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register", response_model=Token, tags=["Auth"])
async def register_user(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(form_data.password)
    fake_users_db[form_data.username] = {
        "username": form_data.username,
        "hashed_password": hashed_password,
        "email": "user@example.com",
        "full_name": "DeepSafe User",
        "disabled": False,
    }

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=User, tags=["Auth"])
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/detect", tags=["Web UI"], response_model_exclude_none=True)
async def detect_media_endpoint_api_form(
    request: Request,
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    ensemble_method: Optional[str] = Form(None),
    models: Optional[str] = Form(None),
):
    req_id = request.state.request_id
    content_type = file.content_type
    inferred_media_type = CONTENT_TYPE_TO_MEDIA_TYPE_MAP.get(content_type)

    if not inferred_media_type:
        logger.warning(
            f"Request {req_id}: Unsupported content_type '{content_type}' from file '{file.filename}'."
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Please upload a supported format.",
        )

    final_threshold = (
        threshold
        if threshold is not None
        else (
            ALL_MODEL_CONFIGS.get("default_threshold", 0.5)
            if ALL_MODEL_CONFIGS
            else 0.5
        )
    )
    final_ensemble_method = (
        ensemble_method
        if ensemble_method is not None
        else (
            ALL_MODEL_CONFIGS.get("default_ensemble_method", "stacking")
            if ALL_MODEL_CONFIGS
            else "stacking"
        )
    )

    logger.info(
        f"Request {req_id}: Web UI ({inferred_media_type}) detection. File: '{file.filename}', Ensemble: '{final_ensemble_method}', Models: '{models or 'all'}', Threshold: {final_threshold}"
    )

    try:
        file_contents = await file.read()
        if not file_contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        media_specific_config = (
            ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
        ).get(inferred_media_type, {})
        max_size_for_type = media_specific_config.get(
            "max_upload_size_bytes", MAX_GENERAL_PAYLOAD_SIZE_BYTES
        )

        if len(file_contents) > max_size_for_type:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file.filename}' too large for type '{inferred_media_type}' (max {max_size_for_type/(1024*1024):.1f}MB).",
            )

        if inferred_media_type == "image":
            try:
                img = Image.open(io.BytesIO(file_contents))
                img.verify()
                img = Image.open(io.BytesIO(file_contents))
                if img.width < 32 or img.height < 32:
                    raise ValueError("Image dimensions are too small (minimum 32x32).")
            except UnidentifiedImageError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot identify image file '{file.filename}'. It might be corrupt or an unsupported image format.",
                )
            except ValueError as e_img_val:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid image file '{file.filename}': {e_img_val}",
                )
            except Exception as e_img_gen:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error processing image file '{file.filename}': {e_img_gen}",
                )

        base64_media = base64.b64encode(file_contents).decode("utf-8")
        parsed_models_list = (
            [m.strip() for m in models.split(",") if m.strip()] if models else None
        )

        predict_payload_data = {
            "media_type": inferred_media_type,
            "threshold": final_threshold,
            "ensemble_method": final_ensemble_method,
            "models": parsed_models_list,
        }
        payload_key_for_media_data = MEDIA_TYPE_PAYLOAD_KEYS.get(inferred_media_type)
        if not payload_key_for_media_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error: media type key mapping failed.",
            )
        predict_payload_data[payload_key_for_media_data] = base64_media

        predict_input_object = PredictInput(**predict_payload_data)
        full_prediction_result = await predict_media_endpoint_api(
            request, predict_input_object
        )

        num_models_contributed = full_prediction_result.get("base_model_total_votes", 0)
        if num_models_contributed == 0 and full_prediction_result.get("model_results"):
            num_models_contributed = len(
                [
                    r
                    for r in full_prediction_result["model_results"].values()
                    if isinstance(r, dict) and "error" not in r
                ]
            )

        ui_response = {
            "request_id": req_id,
            "is_likely_deepfake": full_prediction_result["verdict"] == "fake",
            "deepfake_probability": full_prediction_result.get(
                "ensemble_score_is_fake", 0.5
            ),
            "model_count": num_models_contributed,
            "fake_votes": full_prediction_result.get("base_model_fake_votes", 0),
            "real_votes": full_prediction_result.get("base_model_real_votes", 0),
            "response_time": full_prediction_result.get(
                "total_inference_time_seconds", 0.0
            ),
            "ensemble_method_used": full_prediction_result.get(
                "ensemble_method_used", final_ensemble_method
            ),
            "model_results": full_prediction_result.get("model_results"),
            "processing_mode": "CPU-only",
            "media_type_processed": inferred_media_type,
            "filename": file.filename,
        }
        return ui_response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Request {req_id}: Unhandled error in /detect endpoint for file '{file.filename}': {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred processing your file: {str(e)}",
        )
    finally:
        if "file" in locals() and file:
            await file.close()


# --- History Endpoints ---
@app.get("/history", tags=["History"])
async def get_analysis_history(
    limit: int = 100,
    offset: int = 0,
    media_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve analysis history with pagination and filtering."""
    query = db.query(AnalysisHistory)
    if media_type:
        query = query.filter(AnalysisHistory.media_type == media_type)

    total = query.count()
    records = (
        query.order_by(AnalysisHistory.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": [
            {
                "id": r.id,
                "request_id": r.request_id,
                "media_type": r.media_type,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in records
        ],
    }


@app.get("/history/{request_id}", tags=["History"])
async def get_analysis_by_id(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve a specific analysis result by request ID."""
    record = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.request_id == request_id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "id": record.id,
        "request_id": record.request_id,
        "media_type": record.media_type,
        "verdict": record.verdict,
        "confidence": record.confidence,
        "ensemble_method": record.ensemble_method,
        "timestamp": record.timestamp.isoformat(),
        "full_response": (
            json.loads(record.full_response) if record.full_response else None
        ),
    }


if __name__ == "__main__":
    if (
        not CONFIG_FILE_PATH_FROM_ENV
        or not os.path.exists(CONFIG_FILE_PATH_FROM_ENV)
        or not ALL_MODEL_CONFIGS.get("media_types")
    ):
        logger.critical(
            "FATAL: API configuration (DEEPSAFE_CONFIG_FILE_PATH) is missing, invalid, or does not define 'media_types'. API cannot start meaningfully."
        )
        sys.exit(1)

    port = int(get_environment_variable("PORT", "8000"))
    workers = int(get_environment_variable("WORKERS", "1"))
    log_level = get_environment_variable("LOG_LEVEL", "info").lower()

    logger.info(
        f"Starting DeepSafe API (v{app.version}) on port {port} with {workers} worker(s). Log level: {log_level}"
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )
