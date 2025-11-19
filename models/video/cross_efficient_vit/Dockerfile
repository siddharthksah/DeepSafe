# Use a Python slim base image
FROM python:3.9-slim

WORKDIR /app

# Install essential system dependencies for OpenCV headless and video processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    libgl1-mesa-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    libjpeg-dev libpng-dev libtiff-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV OPENCV_IO_ENABLE_NONFREE=0
ENV QT_QPA_PLATFORM=offscreen

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==1.11.0 \
    torchvision==0.12.0 \
    torchaudio==0.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# NumPy version will be handled by requirements.txt now
# REMOVE THIS LINE: RUN pip install --no-cache-dir numpy==1.21.6

RUN pip install --no-cache-dir opencv-python-headless==4.6.0.66

# Install other Python dependencies from requirements.txt
# This will install numpy as per the constraint in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ... (rest of your Dockerfile remains the same) ...

# Clone the new model's repository
RUN git clone https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection.git model_code

# Download model weights from Google Drive
RUN mkdir -p /app/model_code/gdrive_weights && \
    gdown --folder https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1 -O /app/model_code/gdrive_weights --remaining-ok

RUN mkdir -p /app/model_code/efficient-vit/pretrained_models && \
    mkdir -p /app/model_code/cross-efficient-vit/pretrained_models

# Move downloaded weights to their respective locations
RUN if [ -f /app/model_code/gdrive_weights/efficient_vit.pth ]; then \
        mv /app/model_code/gdrive_weights/efficient_vit.pth /app/model_code/efficient-vit/pretrained_models/efficient_vit.pth; \
    else \
        echo "Warning: efficient_vit.pth not found in gdrive_weights, assuming it might be part of the repo or handled differently." ; \
    fi && \
    if [ -f /app/model_code/gdrive_weights/cross_efficient_vit.pth ]; then \
        mv /app/model_code/gdrive_weights/cross_efficient_vit.pth /app/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth; \
    else \
        echo "Warning: cross_efficient_vit.pth not found in gdrive_weights, assuming it might be part of the repo or handled differently." ; \
    fi && \
    rm -rf /app/model_code/gdrive_weights

# Verification for critical model weights
RUN if [ ! -f /app/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth ]; then echo "ERROR: cross_efficient_vit.pth final placement failed! This is critical." && exit 1; fi

COPY app.py .

ENV PYTHONPATH=/app/model_code/cross-efficient-vit:/app/model_code/efficient-vit:/app/model_code/preprocessing:/app/model_code/cross-efficient-vit/efficient_net:${PYTHONPATH}

ENV MODEL_PORT=7001
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="900"
ENV USE_GPU="false"
ENV DEFAULT_MODEL_VARIANT="cross_efficient_vit"
ENV CROSS_EFFICIENT_VIT_MODEL_PATH="/app/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth"
ENV EFFICIENT_VIT_MODEL_PATH="/app/model_code/efficient-vit/pretrained_models/efficient_vit.pth"
ENV CROSS_EFFICIENT_VIT_CONFIG_PATH="/app/model_code/cross-efficient-vit/configs/architecture.yaml"
ENV EFFICIENT_VIT_CONFIG_PATH="/app/model_code/efficient-vit/configs/architecture.yaml"

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]