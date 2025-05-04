#!/usr/bin/env python3
"""
Test client for DeepSafe deepfake detection models
"""

import requests
import base64
import time
import json
import argparse
import os
from typing import Optional, Dict, List
import sys

# Model endpoints configuration - ADDED caddm

MODEL_ENDPOINTS = {
    "cnndetection": "http://localhost:5000/predict",
    "ganimagedetection": "http://localhost:5001/predict",
    "universalfakedetect": "http://localhost:5002/predict",
    "hifi_ifdl": "http://localhost:5003/predict",
    "npr_deepfakedetection": "http://localhost:5004/predict",
    "dmimagedetection": "http://localhost:5005/predict",
    "caddm": "http://localhost:5006/predict",
    "faceforensics_plus_plus": "http://localhost:5007/predict",
}

def encode_image(image_path: str) -> str:
    """Read an image file and encode it as base64."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def test_model(image_path: str, api_url: str = "http://localhost:5000/predict", threshold: float = 0.5) -> Optional[dict]:
    """Test a deepfake detection model with an image."""
    model_name = "unknown"
    for name, url in MODEL_ENDPOINTS.items():
        if url == api_url: # Check for exact match first
            model_name = name
            break
        elif name in api_url: # Fallback check if name is in URL (e.g., for health check base URL)
            model_name = name
            break

    print(f"\n===== Testing {model_name.upper()} =====")
    print(f"URL: {api_url}")
    print(f"Image: {image_path}")

    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist")
            return None

        # Encode the image
        encoded_image = encode_image(image_path)

        # Create request payload
        payload = {
            "image": encoded_image,
            "threshold": threshold
        }

        # Make request with timing
        print("Sending request...")
        start_time = time.time()
        # Use longer timeout for potentially slower models like UniversalFakeDetect or HiFi_IFDL
        timeout = 120 if "universal" in model_name.lower() or "hifi" in model_name.lower() else 60
        response = requests.post(api_url, json=payload, timeout=timeout)
        total_time = time.time() - start_time

        # Print results
        print(f"Status code: {response.status_code}")
        print(f"Total request time: {total_time:.4f} seconds")

        if response.status_code == 200:
            result = response.json()
            print("Result:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Error: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"Error: Request timed out after {timeout} seconds. The model may still be loading or processing took too long.")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API at {api_url}. Ensure the service is running.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

def test_model_health(api_url: str = "http://localhost:5000/health", retry: bool = True) -> Optional[dict]:
    """Test the health endpoint of a model service with optional retry for loading state."""
    model_name = "unknown"
    base_url_predict = api_url # Assume input is predict URL
    if "/predict" not in api_url: # If base URL/health URL provided
        base_url_predict = api_url.replace("/health", "") + "/predict"

    for name, url in MODEL_ENDPOINTS.items():
        if url == base_url_predict:
            model_name = name
            break

    print(f"\n===== Health Check: {model_name.upper()} =====")

    # Construct health URL from predict URL or base URL
    base_url = base_url_predict.replace("/predict", "")
    health_url = f"{base_url}/health"
    print(f"Health URL: {health_url}")

    max_retries = 5 if retry else 1
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=15) # Increased timeout slightly
            print(f"Health check attempt {attempt+1}/{max_retries} -> Status code: {response.status_code}")

            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get('status', 'unknown')
                print(f"Health status: {status}")

                if 'device' in health_data:
                    print(f"Running on: {health_data['device']}")

                if status == "loading" and retry and attempt < max_retries - 1:
                    print(f"Model is still loading. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue

                # If status is healthy or loading (and no more retries), return data
                return health_data
            else:
                print(f"Health check failed: {response.text}")
                if retry and attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return None # Failed after retries or if no retry

        except requests.exceptions.Timeout:
             print(f"Health check attempt {attempt+1}/{max_retries} -> Request timed out.")
             if retry and attempt < max_retries - 1:
                 print(f"Retrying in {retry_delay} seconds...")
                 time.sleep(retry_delay)
                 continue
             return None
        except requests.exceptions.ConnectionError:
             print(f"Health check attempt {attempt+1}/{max_retries} -> Connection Error. Service might not be up.")
             if retry and attempt < max_retries - 1:
                 print(f"Retrying in {retry_delay} seconds...")
                 time.sleep(retry_delay)
                 continue
             return None # Failed to connect after retries or if no retry
        except Exception as e:
            print(f"Health check attempt {attempt+1}/{max_retries} -> Error: {str(e)}")
            if retry and attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return None # Failed after retries or if no retry

    print(f"Max retries ({max_retries}) reached for health check. Model '{model_name}' did not report healthy.")
    return None

def test_all_models(image_path: str, check_health: bool = True, threshold: float = 0.5) -> Dict[str, Optional[dict]]:
    """Test all individual models with the same image and return results."""
    results = {}

    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return results

    # Sort models to test in a consistent order (optional)
    model_items = sorted(MODEL_ENDPOINTS.items())

    for model_name, api_url in model_items:
        is_healthy = False
        if check_health:
            health_info = test_model_health(api_url, retry=True) # Retry health check
            if health_info and health_info.get('status') == 'healthy':
                is_healthy = True
            else:
                 print(f"Warning: {model_name} health check did not return 'healthy' status after retries. Skipping prediction.")

        if not check_health or is_healthy: # Proceed if health check passed or was skipped
            result = test_model(image_path, api_url, threshold)
            results[model_name] = result
        else:
             results[model_name] = None # Mark as failed due to health check failure

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DeepSafe detection models')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--url', type=str, default=None, help='URL of a specific model API predict endpoint to test (e.g., http://localhost:5003/predict)')
    parser.add_argument('--all', action='store_true', help='Test all configured models')
    parser.add_argument('--health', action='store_true', help='Check model health endpoint(s) before prediction (retries if loading)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (0.0 to 1.0)')

    args = parser.parse_args()

    if not os.path.exists(args.image):
         print(f"Error: Image file specified does not exist: {args.image}")
         sys.exit(1)

    if args.all:
        # Test all individual models
        results = test_all_models(args.image, args.health, args.threshold)

        # Print summary
        print("\n===== Prediction Summary =====")
        for model, result in results.items():
            if result:
                prediction_class = result.get("class", "unknown")
                probability = result.get("probability", -1) # Use -1 to indicate missing probability
                inf_time = result.get("inference_time_seconds", -1)
                if inf_time == -1:
                    inf_time = result.get("inference_time", -1)  # Try alternative key
                prob_str = f"{probability:.4f}" if probability != -1 else "N/A"
                time_str = f"{inf_time:.3f}s" if inf_time != -1 else "N/A"
                print(f"{model.ljust(20)}: {prediction_class.ljust(5)} (Prob Fake: {prob_str}, Time: {time_str})")
            else:
                print(f"{model.ljust(20)}: Failed or Skipped")

    elif args.url:
        # Test specific model URL provided by user
        # Validate if URL is one of the known predict endpoints
        if args.url not in MODEL_ENDPOINTS.values():
             print(f"Warning: Provided URL '{args.url}' is not in the configured MODEL_ENDPOINTS.")
             # Attempt to test anyway

        if args.health:
            # Perform health check for the specific URL
             health_info = test_model_health(args.url, retry=True)
             if not health_info or health_info.get('status') != 'healthy':
                  print(f"Health check failed for {args.url}. Aborting prediction test.")
                  sys.exit(1) # Exit if health check fails for specific test

        # Proceed with prediction test
        test_model(args.image, args.url, args.threshold)

    else:
        # Default behaviour: Test only the first model in the list if --all or --url not specified
        default_model_name = next(iter(MODEL_ENDPOINTS))
        default_model_url = MODEL_ENDPOINTS[default_model_name]
        print(f"No specific model or --all flag provided. Testing default model: {default_model_name}")

        if args.health:
            # Health check for the default model
            health_info = test_model_health(default_model_url, retry=True)
            if not health_info or health_info.get('status') != 'healthy':
                 print(f"Health check failed for default model {default_model_name}. Aborting prediction test.")
                 sys.exit(1)

        # Proceed with prediction test for the default model
        test_model(args.image, default_model_url, args.threshold)

    print("\n===== Test Client Finished =====")