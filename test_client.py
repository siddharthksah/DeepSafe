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

# Model endpoints configuration
MODEL_ENDPOINTS = {
    "cnndetection": "http://localhost:5000/predict",
    "ganimagedetection": "http://localhost:5001/predict",
    "universalfakedetect": "http://localhost:5002/predict"
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
        if url in api_url:
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
        # Use longer timeout for UniversalFakeDetect
        timeout = 120 if "universal" in model_name.lower() else 30
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
        print(f"Error: Request timed out. The model may still be loading.")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API at {api_url}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def test_model_health(api_url: str = "http://localhost:5000/health", retry: bool = True) -> Optional[dict]:
    """Test the health endpoint of a model service with optional retry for loading state."""
    model_name = "unknown"
    for name, url in MODEL_ENDPOINTS.items():
        if name in api_url or url.replace("/predict", "") in api_url:
            model_name = name
            break
    
    print(f"\n===== Health Check: {model_name.upper()} =====")
    
    base_url = "/".join(api_url.split("/")[:-1]) if "/predict" in api_url else api_url
    health_url = f"{base_url}/health"
    print(f"Health URL: {health_url}")
    
    max_retries = 5 if retry else 1
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=10)
            print(f"Health check status code: {response.status_code}")
            
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get('status', 'unknown')
                print(f"Health status: {status}")
                
                if 'device' in health_data:
                    print(f"Running on: {health_data['device']}")
                
                if status == "loading" and retry and attempt < max_retries - 1:
                    print(f"Model is still loading. Retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                return health_data
            else:
                print(f"Health check failed: {response.text}")
                if retry and attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return None
                
        except Exception as e:
            print(f"Health check error: {str(e)}")
            if retry and attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            return None
    
    print("Max retries reached. Health check failed.")
    return None

def test_all_models(image_path: str, check_health: bool = True, threshold: float = 0.5) -> Dict[str, Optional[dict]]:
    """Test all individual models with the same image and return results."""
    results = {}
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return results
    
    for model_name, api_url in MODEL_ENDPOINTS.items():
        if check_health:
            health = test_model_health(api_url, retry=True)
            if health and health.get('status') != 'healthy':
                print(f"Warning: {model_name} health check did not return 'healthy' status")
        
        result = test_model(image_path, api_url, threshold)
        results[model_name] = result
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DeepSafe detection models')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--url', type=str, default=None, help='URL of a specific model API to test')
    parser.add_argument('--all', action='store_true', help='Test all individual models')
    parser.add_argument('--health', action='store_true', help='Check model health before prediction')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    if args.all:
        # Test all individual models
        results = test_all_models(args.image, args.health, args.threshold)
        
        # Print summary
        print("\n===== SUMMARY =====")
        for model, result in results.items():
            if result:
                prediction_class = result.get("class", "unknown")
                probability = result.get("probability", 0)
                print(f"{model.ljust(20)}: {prediction_class} ({probability:.4f})")
            else:
                print(f"{model.ljust(20)}: Failed")
    
    elif args.url:
        # Test specific model
        if args.health:
            test_model_health(args.url, retry=True)
        
        test_model(args.image, args.url, args.threshold)
    
    else:
        # Default to testing CNNDetection
        if args.health:
            test_model_health(MODEL_ENDPOINTS["cnndetection"])
        
        test_model(args.image, MODEL_ENDPOINTS["cnndetection"], args.threshold)