FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install git and wget
RUN apt-get update && apt-get install -y git wget && apt-get clean

# Copy requirements file
COPY requirements.txt .

# Install dependencies
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
ENV MODEL_PORT=5004
ENV USE_GPU=true

# Expose port
EXPOSE 5004

# Run the application
CMD ["python", "app.py"]