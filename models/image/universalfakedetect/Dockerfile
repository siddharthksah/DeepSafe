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

# Download the weights file separately to avoid timeouts during startup
RUN wget -O universalfakedetect/pretrained_weights/fc_weights.pth \
    https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth

# Set environment variables
ENV MODEL_PORT=5002
ENV USE_GPU=true
ENV TIMEOUT=600

# Expose port
EXPOSE 5002

# Run the application with gunicorn for better timeout handling
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002", "--timeout-keep-alive", "600"]