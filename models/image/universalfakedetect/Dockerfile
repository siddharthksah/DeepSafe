FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install git and wget
RUN apt-get update && \
    apt-get install -y git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Clone the repository
RUN git clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git universalfakedetect && \
    mkdir -p universalfakedetect/pretrained_weights

RUN python -c "import torch; import os; os.makedirs('/root/.cache/clip', exist_ok=True)" && \
    wget -O /root/.cache/clip/ViT-L-14.pt https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

# Download the weights file separately to avoid timeouts during startup
RUN wget -O universalfakedetect/pretrained_weights/fc_weights.pth \
    https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth

# Set environment variables
ENV MODEL_PORT=5004
ENV USE_GPU=false
ENV TIMEOUT=600

# Expose port
EXPOSE 5004

# Run the application with gunicorn for better timeout handling
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5004", "--timeout-keep-alive", "600"]