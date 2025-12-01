#!/bin/bash

# Script to download model weights for DeepSafe
# Usage: ./scripts/download_weights.sh

echo "Downloading DeepSafe Model Weights..."

# Directory setup
mkdir -p models/image/wavelet_clip_detection/model_code/weights
mkdir -p models/video/cross_efficient_vit/model_code/gdrive_weights

# Wavelet-CLIP Weights
echo "------------------------------------------------"
echo "Checking Wavelet-CLIP weights..."
if [ -f "models/image/wavelet_clip_detection/model_code/weights/clip_wavelet_best.pth" ]; then
    echo "Wavelet-CLIP weights already exist."
else
    echo "Please download 'clip_wavelet_best.pth' manually from the official repository or Google Drive."
    echo "Link: https://drive.google.com/drive/folders/1Z7pH9KPQbx1TrMqap2y9Op6OUO9SHP_D"
    echo "Place it in: models/image/wavelet_clip_detection/model_code/weights/"
fi

# CrossEfficientViT Weights
echo "------------------------------------------------"
echo "Checking CrossEfficientViT weights..."
# This is a placeholder as the Dockerfile handles gdown, but this script serves as a manual fallback guide.
echo "CrossEfficientViT weights are typically handled by the Docker build."
echo "If build fails, download from: https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1"
echo "Place in: models/video/cross_efficient_vit/model_code/gdrive_weights/"

# FakeSTormer Weights
echo "------------------------------------------------"
echo "Checking FakeSTormer weights..."
if [ -f "models/video/fake_stormer/model_code/weights/best.pth" ]; then
    echo "FakeSTormer weights already exist."
else
    echo "Please download the weights from the following Dropbox link:"
    echo "Link: https://www.dropbox.com/scl/fo/elk2szqf0du4l6zm5job9/AAdVmNH--6ywHBZGNQJlR5o?rlkey=5kde7vj4wklrx1jwdul0m6g46&e=1&st=czw4szw0&dl=0"
    echo "Place the 'best.pth' file in: models/video/fake_stormer/model_code/weights/"
fi

echo "------------------------------------------------"
echo "Download check complete."
