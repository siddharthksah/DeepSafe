# Integrating New Models into DeepSafe

DeepSafe is designed to be **model-agnostic**. You can integrate *any* deepfake detection model (PyTorch, TensorFlow, etc.) as long as it can be containerized and expose a simple HTTP API.

## Overview
To add a new model, you need to:
1.  **Containerize** your model using Docker.
2.  **Expose** a standardized HTTP API (Predict & Health).
3.  **Register** the model in `docker-compose.yml` and `deepsafe_config.json`.

---

## Step 1: Containerize Your Model

Create a directory for your model in `models/<media_type>/<model_name>`.
Example: `models/image/my_awesome_model/`

### `Dockerfile`
Create a `Dockerfile` that installs your dependencies. We recommend using `uv` for fast builds.

```dockerfile
FROM python:3.9-slim

# Install system dependencies (e.g., for OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy model code and weights
COPY . .

# Expose the port (e.g., 5005)
EXPOSE 5005

# Run the service
CMD ["python", "api.py"]
```

---

## Step 2: Expose HTTP API

Your model **MUST** expose two endpoints:

### 1. `GET /health`
Returns JSON indicating the model is ready.
```json
{ "status": "healthy" }
```

### 2. `POST /predict`
Accepts a JSON payload with the media data and returns a probability.

**Request Payload:**
```json
{
  "image_data": "base64_encoded_string...",
  "threshold": 0.5
}
```
*(Note: Use `video_data` or `audio_data` keys for other media types)*

**Response Payload:**
```json
{
  "probability": 0.95,      // Float 0.0 - 1.0 (Probability of being FAKE)
  "prediction": 1,          // 1 = Fake, 0 = Real
  "class": "fake",          // "fake" or "real"
  "inference_time": 0.12    // Seconds
}
```

### Example `api.py` (Flask)
```python
from flask import Flask, request, jsonify
import model_inference  # Your inference logic

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_b64 = data.get("image_data")
    
    # Run your model
    prob = model_inference.predict_from_base64(image_b64)
    
    return jsonify({
        "probability": prob,
        "prediction": 1 if prob >= 0.5 else 0,
        "class": "fake" if prob >= 0.5 else "real"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
```

---

## Step 3: Register the Model

### 1. Update `docker-compose.yml`
Add your new service.

```yaml
  my_awesome_model:
    build: ./models/image/my_awesome_model
    container_name: deepsafe-my-awesome-model
    ports:
      - "5005:5005"
    networks:
      - deepsafe-network
```

### 2. Update `config/deepsafe_config.json`
Tell the API Gateway about your new model.

```json
{
  "media_types": {
    "image": {
      "model_endpoints": {
        "npr_deepfakedetection": "http://npr_deepfakedetection:5001/predict",
        "my_awesome_model": "http://my_awesome_model:5005/predict"  <-- ADD THIS
      },
      "health_endpoints": {
        "npr_deepfakedetection": "http://npr_deepfakedetection:5001/health",
        "my_awesome_model": "http://my_awesome_model:5005/health"   <-- ADD THIS
      }
    }
  }
}
```

---

## Step 4: Verify
1.  Rebuild: `make start` (or `docker-compose up -d --build`).
2.  Check logs: `docker logs deepsafe-api`.
3.  Test: The new model will automatically be included in the ensemble!
