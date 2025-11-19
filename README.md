# DeepSafe: Advanced Deepfake Detection Platform

![DeepSafe Banner](https://via.placeholder.com/1200x300?text=DeepSafe+Deepfake+Detection+Platform)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/yourusername/DeepSafe/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/DeepSafe/actions)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**DeepSafe** is a modular, extensible, and containerized platform for detecting deepfakes in images and videos. It aggregates state-of-the-art detection models into a unified API and provides a user-friendly web interface for analysis.

## 🚀 Key Features

-   **Multi-Model Ensemble**: Combines predictions from multiple state-of-the-art models (NPR, UniversalFakeDetect, CrossEfficientViT, etc.) for higher accuracy.
-   **Meta-Learning**: Uses a stacking ensemble approach to learn the best combination of model outputs.
-   **Modular Architecture**: Each model runs in its own Docker container, allowing for easy addition or removal of detectors.
-   **Video & Image Support**: Specialized pipelines for both media types.
-   **Modern UI**: React-based frontend for easy interaction and visualization of results.
-   **REST API**: Fully documented FastAPI backend for integration into other systems.

## 🏗️ Architecture

DeepSafe uses a microservices architecture orchestrated by Docker Compose:

```mermaid
graph TD
    Client[Web Client / User] --> Nginx[Nginx / Frontend]
    Nginx --> API[DeepSafe API (FastAPI)]
    API --> Meta[Meta-Learner / Ensemble]
    
    subgraph "Image Detectors"
        Meta --> NPR[NPR Deepfake]
        Meta --> UFD[Universal Fake Detect]
        Meta --> WClip[Wavelet CLIP]
    end
    
    subgraph "Video Detectors"
        Meta --> CEV[Cross Efficient ViT]
    end
```

## 🛠️ Quick Start

### Prerequisites

-   Docker & Docker Compose
-   Git

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/DeepSafe.git
    cd DeepSafe
    ```

2.  **Start the application**:
    ```bash
    make start
    ```
    *This command builds all necessary containers and starts the services. It may take a while on the first run.*

3.  **Access the Dashboard**:
    Open your browser and navigate to `http://localhost:80`.

4.  **Access the API Docs**:
    Navigate to `http://localhost:8000/docs` for the Swagger UI.

## 🧪 Testing

We provide a built-in test suite to verify the system's health and prediction capabilities.

```bash
make test
```

This command will:
1.  Check the health of all active services.
2.  Run prediction tests on sample image and video files.
3.  Report the results in the terminal.

## 📦 Available Models

| Model | Type | Status | Description |
| :--- | :--- | :--- | :--- |
| **NPR Deepfake** | Image | ✅ Active | Neural Pattern Recognition for image deepfakes. |
| **Universal Fake Detect** | Image | ✅ Active | Generalizable deepfake detection. |
| **Cross Efficient ViT** | Video | ✅ Active | Efficient video deepfake detection. |
| **Wavelet CLIP** | Image | 🚧 Optional | Requires manual weight download. |
| **TruFor** | Image | 🚧 Disabled | Transformer for image forgery detection. |

*To enable disabled models, uncomment them in `docker-compose.yml`.*

## 🗺️ Roadmap

- [x] Core API and Frontend
- [x] Docker Containerization
- [x] Image Detection Pipeline
- [x] Video Detection Pipeline
- [ ] Real-time Webcam Analysis
- [ ] Audio Deepfake Detection
- [ ] Browser Extension

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgements

-   [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) for model implementations.
-   [FastAPI](https://fastapi.tiangolo.com/) for the high-performance API.
-   [React](https://reactjs.org/) for the frontend library.
