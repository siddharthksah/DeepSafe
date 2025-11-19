import requests
import base64
import os
import json
import time

API_URL = "http://localhost:8000"
IMAGE_PATH = "test_samples/sample_image.jpg"
VIDEO_PATH = "test_samples/sample_video.mp4"

def encode_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_health():
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Health check failed: {e}")

def test_image_prediction():
    print("\nTesting /predict with IMAGE...")
    if not os.path.exists(IMAGE_PATH):
        print(f"Image file not found at {IMAGE_PATH}")
        return

    encoded_image = encode_file(IMAGE_PATH)
    payload = {
        "media_type": "image",
        "image_data": encoded_image,
        "ensemble_method": "voting", # Using voting as stacking might not be fully loaded
        "threshold": 0.5
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=payload)
        duration = time.time() - start_time
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(response.text)
        print(f"Time taken: {duration:.2f}s")
    except Exception as e:
        print(f"Image prediction failed: {e}")

def test_video_prediction():
    print("\nTesting /predict with VIDEO...")
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file not found at {VIDEO_PATH}")
        return

    encoded_video = encode_file(VIDEO_PATH)
    payload = {
        "media_type": "video",
        "video_data": encoded_video,
        "ensemble_method": "voting",
        "threshold": 0.5
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=payload)
        duration = time.time() - start_time
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(response.text)
        print(f"Time taken: {duration:.2f}s")
    except Exception as e:
        print(f"Video prediction failed: {e}")

if __name__ == "__main__":
    test_health()
    test_image_prediction()
    test_video_prediction()
