# DeepSafe: CPU-Optimized Deepfake Detection System

DeepSafe is an enterprise-grade system designed for detecting deepfake images. It utilizes an ensemble of multiple state-of-the-art deep learning models to provide robust and accurate detection capabilities. The system is built with a microservice architecture, orchestrated using Docker Compose, and is **specifically configured for CPU-only inference** in this setup.

It includes:

*   A central REST API for programmatic access.
*   Individual model microservices, each running a specific detection model.
*   A web-based frontend for easy image uploads and analysis.
*   Command-line client tools for direct interaction and batch processing.

**Note:** This configuration runs all deep learning models on the CPU. While this increases accessibility (no GPU required), inference times will be significantly longer compared to a GPU-accelerated setup.

## Features

*   **Ensemble Detection:** Combines results from multiple diverse models for improved accuracy and robustness.
*   **Multiple Models:** Integrates various SOTA deepfake detection models targeting different generation techniques (GANs, Diffusion Models, Face Manipulation, etc.).
*   **CPU Optimized:** Configured via Docker Compose and model settings to run entirely on CPU resources.
*   **Microservice Architecture:** Scalable and maintainable design using individual services for API, frontend, and each model.
*   **RESTful API:** Easy integration with other systems (`/predict`, `/health`).
*   **Web Interface:** User-friendly frontend for uploading images and viewing results.
*   **Command-Line Clients:**
    *   `deepsafe_client.py`: Interact with the main API for health checks and detection.
    *   `model_test_client.py`: Diagnose and test individual model endpoints directly.
    *   `deepsafe_batch_test.py`: Perform batch testing on image folders, optimized for sequential CPU execution.
*   **Configurable:** Settings like detection threshold and ensemble method can be adjusted.
*   **Dockerized:** Easily deployable using Docker and Docker Compose.
*   **Health Monitoring:** Endpoints to check the status of the main API and individual models.

## Architecture

DeepSafe employs a microservice architecture orchestrated by Docker Compose:

## Docker Network (deepsafe-network)

1.  **Frontend (`frontend` service):** A React application served by Nginx. Users interact with this interface. It runs on port 80.
2.  **API (`api` service):** The central FastAPI application. It receives requests from the frontend or clients, forwards them to the relevant model services, aggregates results, and calculates the ensemble verdict. It runs on port 8000.
3.  **Model Services (`cnndetection`, `ganimagedetection`, etc.):** Each service runs a specific deepfake detection model within its own FastAPI application. They expose `/predict` and `/health` endpoints and perform inference **on the CPU**. They run on ports 5001-5008.
4.  **Docker Network (`deepsafe-network`):** Allows services to communicate with each other using their service names (e.g., `http://api:8000`, `http://cnndetection:5008`).

## Prerequisites

*   **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
*   **Docker Compose:** [Install Docker Compose](https://docs.docker.com/compose/install/) (Usually included with Docker Desktop)
*   **Git:** For cloning the repository.
*   **Python 3.8+ (Optional):** Required only if you want to run the client scripts (`deepsafe_client.py`, `model_test_client.py`, `deepsafe_batch_test.py`) directly on your host machine.
*   **Internet Connection:** To download base Docker images, Python packages, and potentially model weights during the build process.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd DeepSafe
    ```

2.  **Build and Run with Docker Compose:**
    This is the primary method for running the entire system. It will build the Docker images for the API, frontend, and all model services, then start the containers.

    ```bash
    docker-compose up --build -d
    ```

    *   `--build`: Forces Docker Compose to build the images based on the Dockerfiles.
    *   `-d`: Runs the containers in detached mode (in the background).

    **Note:** The initial build process might take a significant amount of time, especially downloading base images and model weights. Subsequent builds will be faster due to Docker's layer caching.

3.  **Wait for Services to Initialize:**
    After starting, the model services need time to load their respective models into memory (especially on CPU). This can take several minutes per model. You can monitor the logs:
    ```bash
    docker-compose logs -f
    # Or for a specific service:
    # docker-compose logs -f cnndetection
    ```
    Look for messages indicating successful model loading in each service's log output. The system is fully ready when the `/health` endpoint reports all models as healthy or loaded.

4.  **(Optional) Setup for Client Scripts:**
    If you want to run the client scripts directly on your host:
    *   Create a Python virtual environment (recommended):
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Install the required Python packages for the clients:
        ```bash
        pip install -r requirements.txt
        ```

## Running the System

*   **Start Services:**
    ```bash
    docker-compose up -d
    ```
*   **Stop Services:**
    ```bash
    docker-compose down
    ```
*   **View Logs:**
    ```bash
    docker-compose logs -f                  # View logs for all services
    docker-compose logs -f api              # View logs for the API service
    docker-compose logs -f cnndetection   # View logs for a specific model service
    ```
*   **Check Running Containers:**
    ```bash
    docker-compose ps
    ```

## Usage

### 1. Web Frontend

*   Access the web interface by navigating to `http://localhost` in your web browser (or `http://<your-server-ip>` if running on a remote machine).
*   Upload an image file (JPEG, PNG, WebP).
*   Click "Analyze Image".
*   View the detection results, including the overall verdict, confidence score, and model votes.
*   Use the "Settings" button to configure the detection threshold, ensemble method, and optionally select specific models to use.

### 2. REST API (`http://localhost:8000`)

The API service provides endpoints for interaction:

*   **`GET /`**: Basic API information.
*   **`GET /health`**: Check the health status of the API and all underlying model services. Returns the status of each model (`healthy`, `loading`, `error`, `missing_weights`).
*   **`POST /predict`** (Content-Type: `application/json`):
    *   **Request Body:**
        ```json
        {
          "image": "base64_encoded_image_string",
          "threshold": 0.5, // Optional, default: 0.5
          "ensemble_method": "voting", // Optional, default: "voting" ('voting' or 'average')
          "models": ["cnndetection", "universalfakedetect"] // Optional, list of models to use, defaults to all
        }
        ```
    *   **Response Body:**
        ```json
        {
          "request_id": "unique_id",
          "verdict": "fake", // "fake", "real", or "undetermined"
          "confidence": 0.85, // Confidence score (0.0 to 1.0)
          "fake_votes": 6,
          "real_votes": 2,
          "total_votes": 8,
          "inference_time": 15.234, // Total time in seconds
          "ensemble_method": "voting",
          "model_results": {
            "cnndetection": { "model": "CNNDetection", "probability": 0.9, "prediction": 1, "class": "fake", "inference_time": 2.1 },
            "universalfakedetect": { "model": "UniversalFakeDetect", "probability": 0.2, "prediction": 0, "class": "real", "inference_time": 3.5 },
            // ... other model results or {"error": "reason"}
          },
          "processing_mode": "CPU-only"
        }
        ```
*   **`POST /detect`** (Content-Type: `multipart/form-data`): Designed for direct file uploads from forms (like the web UI).
    *   **Form Fields:**
        *   `file`: The image file.
        *   `threshold` (optional, float): Detection threshold.
        *   `ensemble_method` (optional, string): 'voting' or 'average'.
        *   `models` (optional, string): Comma-separated list of models.
    *   **Response Body (Simplified):**
        ```json
        {
          "request_id": "unique_id",
          "is_likely_deepfake": true,
          "deepfake_probability": 0.85, // Combined probability/confidence
          "model_count": 8,
          "fake_votes": 6,
          "real_votes": 2,
          "response_time": 15.234,
          "processing_mode": "CPU-only"
        }
        ```

### 3. DeepSafe Client (`deepsafe_client.py`)

This script allows interacting with the main DeepSafe API from the command line.

*   **Check System Health:**
    ```bash
    ./deepsafe_client.py health [--api http://localhost:8000]
    ```
*   **Detect Deepfake in an Image:**
    ```bash
    # Basic usage
    ./deepsafe_client.py detect --image path/to/your/image.jpg

    # Specify API URL, models, threshold, and method
    ./deepsafe_client.py detect --image img.png --api http://localhost:8000 --models cnndetection,universalfakedetect --threshold 0.6 --method average

    # Save results to a JSON file
    ./deepsafe_client.py detect --image img.webp --output results.json
    ```
    *(Ensure the script is executable: `chmod +x deepsafe_client.py`)*

### 4. Model Test Client (`model_test_client.py`)

Use this script for diagnosing and testing **individual** model microservices directly.

*   **Check Health of a Specific Model:**
    ```bash
    ./model_test_client.py --url http://localhost:5008/predict --health
    # OR use the health endpoint directly
    # ./model_test_client.py --url http://localhost:5008/health --health
    ```
*   **Test a Specific Model with an Image:**
    ```bash
    ./model_test_client.py --image path/to/image.jpg --url http://localhost:5008/predict [--threshold 0.7]
    ```
*   **Test All Models Sequentially with an Image:**
    ```bash
    ./model_test_client.py --image path/to/image.jpg --all [--health] [--threshold 0.6] [--output all_models_results.json]
    ```
    *(Ensure the script is executable: `chmod +x model_test_client.py`)*

### 5. Batch Test Tool (`deepsafe_batch_test.py`)

This tool is designed for testing multiple images against the models sequentially, optimized for CPU usage with memory clearing between runs.

*   **Test All Images in a Folder with Individual Models:**
    ```bash
    ./deepsafe_batch_test.py --assets path/to/image_folder [--threshold 0.5] [--output batch_results.json]
    ```
*   **Test All Images using the Main API (Ensemble):**
    ```bash
    ./deepsafe_batch_test.py --assets path/to/image_folder --api-only [--method average] [--threshold 0.6] [--output api_batch_results.json]
    ```
    *(Ensure the script is executable: `chmod +x deepsafe_batch_test.py`)*

## Configuration

The system configuration is primarily managed through:

1.  **`docker-compose.yml`:**
    *   Defines services, ports, networks, and dependencies.
    *   Sets environment variables for each service. Key variables include:
        *   `USE_GPU=false`: **Forces CPU usage for all model services.**
        *   `MODEL_PORT`: Port for the specific model service.
        *   `PRELOAD_MODEL`: (Default: `false`) Whether to load the model on service startup (`true`) or on the first request (`false`). Lazy loading saves initial memory but adds delay to the first prediction.
        *   `MODEL_TIMEOUT`: (Default: `600`) Seconds of inactivity before a model is unloaded from memory (if `PRELOAD_MODEL=false`).
        *   `MODEL_PATH`: Path to the model weights file within the container (used by some models like `cnndetection`).
        *   `MODEL_BACKBONE`: Specific backbone architecture for models like `caddm`.
        *   API Service URLs: Environment variables like `CNNDETECTION_URL`, `GANIMAGEDETECTION_URL`, etc., tell the main API how to reach each model service.
2.  **Model `app.py` files:** Some models might have internal defaults or further logic based on environment variables set in `docker-compose.yml`. Several explicitly confirm `USE_GPU=False` and `DEVICE='cpu'`.
3.  **Client Scripts:** Accept command-line arguments for API URL, threshold, models, etc.

## Models Included

This DeepSafe instance includes the following detection models, each running as a separate microservice:

1.  **`cnndetection`** (Port 5008): Based on [PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection) - General CNN-based detection.
2.  **`ganimagedetection`** (Port 5001): Based on [grip-unina/GANimageDetection](https://github.com/grip-unina/GANimageDetection) - Specialized in detecting images generated by GANs (e.g., StyleGAN).
3.  **`universalfakedetect`** (Port 5002): Based on [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) - Aims for generalization across different generative models using CLIP features.
4.  **`hifi_ifdl`** (Port 5003): Based on [CHELSEA234/HiFi_IFDL](https://github.com/CHELSEA234/HiFi_IFDL) - Focuses on high-frequency inconsistencies for detecting manipulation.
5.  **`npr_deepfakedetection`** (Port 5004): Based on [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) - Uses Neural Pattern Recognition techniques.
6.  **`dmimagedetection`** (Port 5005): Based on [grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection) - Designed to detect images generated by Diffusion Models.
7.  **`caddm`** (Port 5006): Based on [megvii-research/CADDM](https://github.com/megvii-research/CADDM) - Convolutional Artifact Detection Module.
8.  **`faceforensics_plus_plus`** (Port 5007): Based on [HongguLiu/Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection) - Uses XceptionNet trained on the FaceForensics++ dataset, primarily for face manipulations.

## Technology Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **Machine Learning:** PyTorch
*   **Frontend:** React, Tailwind CSS, Nginx
*   **Containerization:** Docker, Docker Compose
*   **CLI:** Python, Rich, Requests
*   **Image Processing:** Pillow, OpenCV (within some models)

## Contributing

Contributions are welcome! Please follow standard practices like creating issues for bugs or feature requests and submitting pull requests for changes. (Further contribution guidelines can be added here).

## License

Contact me for the commercial usage.