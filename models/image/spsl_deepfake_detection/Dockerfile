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
    mkdir -p /app/DeepfakeBench/training/weights

# --- Corrected Dlib shape predictor download location ---
# The DeepfakeBench code (e.g., fwa_blend.py) uses a relative path:
# 'preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
# When our app.py runs from /app, this resolves to /app/preprocessing/dlib_tools/
RUN mkdir -p /app/preprocessing/dlib_tools
RUN wget -q https://raw.githubusercontent.com/codeniko/shape_predictor_81_face_landmarks/master/shape_predictor_81_face_landmarks.dat \
    -O /app/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat

# Download Xception backbone weights (ImageNet pretrained) into DeepfakeBench's structure
RUN wget -q https://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth \
    -O /app/DeepfakeBench/training/pretrained/xception-b5690688.pth

# Download SPSL model weights into DeepfakeBench's structure
RUN wget -q https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/SPSL_best.pth \
    -O /app/DeepfakeBench/training/weights/spsl_faceforensics++.pth


# Copy the application file
COPY app.py .

# Set PYTHONPATH to include the DeepfakeBench main and training directory
# This allows `from detectors import ...` to find the modules within DeepfakeBench
ENV PYTHONPATH=/app/DeepfakeBench:/app/DeepfakeBench/training:${PYTHONPATH}

# Environment variables for the application
ENV MODEL_PORT=5006
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="600"
ENV USE_GPU="false"

# Expose the application port
EXPOSE ${MODEL_PORT}

# Command to run the application
CMD ["python", "app.py"]