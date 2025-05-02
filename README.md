# DeepFake Detection System

A production-ready system for detecting deepfake images using the CNNDetection model.

## Overview

This project provides a complete solution for deepfake detection with:

- **Model Service**: A containerized CNNDetection model service
- **API Backend**: FastAPI service handling client requests
- **Web Frontend**: React-based user interface for easy interaction
- **Docker Deployment**: Complete Docker setup for easy deployment

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────┐
│                 │     │              Backend                │
│  Web Frontend   │◄───►│  ┌─────────┐    ┌──────────────┐    │
│  (React)        │     │  │ API     │◄──►│CNNDetection  │    │
│                 │     │  │(FastAPI)│    │Model Service │    │
└─────────────────┘     │  └─────────┘    └──────────────┘    │
                        └─────────────────────────────────────┘
```

## Prerequisites

- Docker and Docker Compose
- Git
- WSL2 if running on Windows

## Setup for Windows with WSL

1. Enable WSL2 on Windows:
   ```
   wsl --install
   ```

2. Install Ubuntu from Microsoft Store

3. Install Docker Desktop for Windows with WSL2 integration

4. Install VS Code with Remote - WSL extension

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://your-repo-url/deepfake-detector.git
   cd deepfake-detector
   ```

2. Copy the CNNDetection model code:
   ```bash
   # Create the directory structure
   mkdir -p model/cnn_detection
   
   # Copy CNNDetection code into model/cnn_detection
   # You can clone the repository or copy from an existing location
   cp -r /path/to/CNNDetection/* model/cnn_detection/
   ```

3. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Web UI: http://localhost:3000
   - API: http://localhost:8000
   - Model Service: http://localhost:5000

## Usage

1. Open the web UI at http://localhost:3000
2. Upload an image you want to analyze
3. Click "Analyze Image"
4. View the results showing the probability of the image being a deepfake

## API Endpoints

### Main API (FastAPI)

- `GET /health` - Health check
- `POST /detect` - Upload and analyze an image

### Model Service

- `GET /health` - Health check
- `POST /predict` - Internal endpoint for image analysis

## Development

### Project Structure

```
deepfake-detector/
├── api/                      # FastAPI backend 
│   ├── main.py               # API entry point
│   ├── Dockerfile            # API container definition
│   └── requirements.txt      # API dependencies
├── model/                    # CNNDetection model service
│   ├── app.py                # Model service code
│   ├── Dockerfile            # Model container definition
│   ├── requirements.txt      # Model dependencies
│   └── cnn_detection/        # CNNDetection code (copied from repo)
├── frontend/                 # React frontend
│   ├── src/                  # React source code
│   ├── public/               # Public assets
│   ├── Dockerfile            # Frontend container definition
│   └── package.json          # Frontend dependencies
├── docker-compose.yml        # Docker Compose configuration
└── README.md                 # Project documentation
```

### Local Development

For frontend development:
```bash
cd frontend
npm install
npm start
```

For API development:
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

## Extending the System

To add more deepfake detection models:

1. Create a new model service similar to the existing one
2. Update the API to call multiple model services
3. Implement an ensemble method to combine results
4. Update the UI to display multiple model results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [CNNDetection](https://github.com/PeterWang512/CNNDetection) for the deepfake detection model