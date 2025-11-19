# Use a Python slim base image
FROM python:3.9-slim

WORKDIR /app

# Install git, wget, build-essential, cmake for dlib, and libGL for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies, ensuring CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    torch==1.12.0 \
    torchvision==0.13.0 \
    torchaudio==0.12.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Clone the DeepfakeBench repository
RUN git clone https://github.com/SCLBD/DeepfakeBench.git /app/DeepfakeBench

# Create directories for DeepfakeBench's expected pretrained models and weights
RUN mkdir -p /app/DeepfakeBench/training/pretrained && \
    mkdir -p /app/DeepfakeBench/training/weights && \
    mkdir -p /app/preprocessing/dlib_tools

# Download dlib shape_predictor_81_face_landmarks.dat
RUN wget -q https://raw.githubusercontent.com/codeniko/shape_predictor_81_face_landmarks/master/shape_predictor_81_face_landmarks.dat \
    -O /app/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat

# Download Xception backbone weights (ImageNet pretrained) - UCF uses Xception
RUN wget -q https://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth \
    -O /app/DeepfakeBench/training/pretrained/xception-b5690688.pth

# Download UCF model weights (trained on FaceForensics++)
# NOTE: The UCF paper mentions training on FF++, FF-DF, FF-F2F, FF-FS, FF-NT.
# The DeepfakeBench release page has UCF_FF++.pth for weights trained on FF++.
RUN wget -q https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/ucf_best.pth \
    -O /app/DeepfakeBench/training/weights/ucf_faceforensics++.pth


# Copy the application file
COPY app.py .

# Set PYTHONPATH to include the DeepfakeBench main and training directory
ENV PYTHONPATH=/app/DeepfakeBench:/app/DeepfakeBench/training:${PYTHONPATH}

# Environment variables for the application
ENV MODEL_PORT=5007
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="600"
ENV USE_GPU="false"

# Expose the application port
EXPOSE ${MODEL_PORT}

# Command to run the application
CMD ["python", "app.py"]