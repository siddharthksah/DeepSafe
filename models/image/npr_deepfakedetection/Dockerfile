FROM python:3.9-slim

WORKDIR /app

# Install git and wget
RUN apt-get update && apt-get install -y --no-install-recommends git wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install common build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install NumPy first - use a common and stable version
RUN pip install --no-cache-dir numpy==1.23.5 

# Install PyTorch CPU version (2.0.1 from requirements.txt) and other dependencies
# Ensure torch and torchvision are for CPU
RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu
# Install other dependencies from requirements.txt
# Pip should skip torch/torchvision if already installed with the correct version.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Clone the repository
RUN git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git npr_deepfakedetection

# Download model weights
RUN mkdir -p npr_deepfakedetection/weights && \
    wget -O npr_deepfakedetection/weights/NPR.pth \
    https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth

# Set environment variables
ENV MODEL_PORT=5001
ENV USE_GPU=false
ENV PRELOAD_MODEL=false
ENV MODEL_TIMEOUT=600

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]