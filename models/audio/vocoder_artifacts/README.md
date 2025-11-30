# Vocoder Artifacts Audio Detection Model

## Overview
This directory contains the integration of the **Vocoder Artifacts** audio deepfake detection model into DeepSafe.

**Source**: [csun22/Synthetic-Voice-Detection-Vocoder-Artifacts](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts)

**Paper**: [AI-Synthesized Voice Detection Using Neural Vocoder Artifacts (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.html)

## Model Architecture
- **Base**: Modified RawNet2 with multi-loss training
- **Input**: Audio waveform (16kHz, ~4 seconds)
- **Output**: Binary classification (Real/Fake)
- **Performance**: 4.54% EER on ASVspoof dataset

## Downloading Model Weights
Due to file size (~100MB), model weights are NOT included in the Docker image by default.

You must download them separately:

```bash
# Option 1: Direct download (requires Google Drive authentication)
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&id=15qOi26czvZddIbKP_SOR8SLQFZK8cf8E" -O best_model.pth

# Option 2: Manual download
# Visit: https://drive.google.com/file/d/15qOi26czvZddIbKP_SOR8SLQFZK8cf8E/view
# Download and save to: models/audio/vocoder_artifacts/models/best_model.pth
```

## Directory Structure
```
models/audio/vocoder_artifacts/
├── Dockerfile          # Container definition
├── api.py              # Flask API wrapper
├── requirements.txt    # Python dependencies
├── temp_repo/          # Cloned original repository (model code)
├── models/             # Model weights directory
│   └── best_model.pth  # Pretrained weights (download separately)
└── README.md           # This file
```

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "model": "Vocoder Artifacts Detection", "device": "cpu"}
```

### Prediction
```
POST /predict
Body: {
  "audio_data": "base64_encoded_wav_file",
  "threshold": 0.5
}

Response: {
  "probability": 0.87,
  "prediction": 1,
  "class": "fake",
  "inference_time": 0.12,
  "sample_rate": 16000
}
```

## Testing
```bash
# Start the service
docker-compose up -d vocoder_artifacts

# Check health
curl http://localhost:8001/health

# Test prediction (requires sample audio)
python deepsafe_test.py test --media-type audio --input test_samples/sample_audio.wav
```
